import math
import random
from tarfile import LinkFallbackError

from physics.pool import Pool
from physics.sound import Sound
from players.seeker import Seeker
from players.hider import Hider

SYMBOLS = {
    0: "▯",
    0.125: "░",
    0.25: "▒",
    0.375: "▓",
    0.5: "▓",
    0.625: "▓",
    0.75: "▓",
    0.875: "█",
    1: "█",
}

class MarcoPolo:
    """
    Game orchestrator: owns players, runs rounds, and uses physics primitives.
    """

    def __init__(self, pool_name, num_polos, diagnostics = False):
        self.pool = Pool(pool_name)
        self.num_polos = num_polos
        self.time = 0

        self.init_players()

        self.round_yell = False

        if diagnostics != False:
            self.diagnostics = diagnostics

    def init_players(self):
        available_positions = []
        for i in range(len(self.pool.grid)):
            for j in range(len(self.pool.grid[0])):
                if self.pool.grid[i][j] == 0:
                    available_positions.append((i, j))
        random.shuffle(available_positions)

        # seeker starts at first available position
        self.marco = Seeker(available_positions[0][0], available_positions[0][1], self.pool)
        self.marco.game = self

        # hiders start at subsequent positions
        self.polos = []
        for i in range(self.num_polos):
            polo = Hider(available_positions[i + 1][0], available_positions[i + 1][1], self.pool)
            polo.game = self
            self.polos.append(polo)

    def has_won(self):
        for polo in self.polos:
            if polo.pos == self.marco.pos:
                return True
        return False

    def simulate_polo_action(self, polo):
        sound = None

        dist = math.hypot(polo.pos[0] - self.marco.pos[0], polo.pos[1] - self.marco.pos[1])

        if self.round_yell and dist >= 2:
            sound = self.pool.get_action_sound(polo.pos, "yell")
        else:
            (dx, dy) = polo.choose_action()
            polo.pos = (polo.pos[0] + dx, polo.pos[1] + dy)
            sound = self.pool.get_action_sound(polo.pos, (dx, dy))

        loudness = sound.observed_sound_loudness(self.marco.pos)
        polo.beliefGrid = polo.get_updated_belief_grid([(sound.pos[0], sound.pos[1], loudness)])
        
        return sound

    def simulate_marco_action(self):
        action = self.marco.choose_action()
        print(f"Marco chose action: {action}")
        print(f"Marco current pos: {self.marco.pos}")
        
        if action == "yell":
            self.round_yell = True
        else:
            (dx, dy) = action
            new_x = self.marco.pos[0] + dx
            new_y = self.marco.pos[1] + dy
            
            print(f"New position would be: ({new_x}, {new_y})")
            print(f"in_bounds check: {self.pool.in_bounds(new_x, new_y)}")
            
            if self.pool.in_bounds(new_x, new_y):
                self.marco.pos = (new_x, new_y)
                print(f"Moved to: {self.marco.pos}")
            else:
                print("Move blocked!")

    def iterate_round(self):
        """
        Advance the game by one round.
        Returns True if Marco catches a polo, False otherwise.
        """
        self.pool.time += 1

        self.simulate_marco_action()

        if self.has_won():
            return True

        sounds = []
        for polo in self.polos:
            sounds.append(self.simulate_polo_action(polo))

        observations = []
        for sound in sounds:
            (pos, loudness) = sound.observed_sound(self.marco.pos)
            print(f"Observation: {pos}, {loudness}")

            x = max(0, min(pos[0], len(self.pool.grid)-1))
            y = max(0, min(pos[1], len(self.pool.grid[0])-1))

            observations.append((x, y, loudness))
        self.marco.beliefGrid = self.marco.get_updated_belief_grid(observations)

        self.round_yell = False

        self.time += 1

        return False

    def render(self):
        """Render the current game state."""
        self.pool.render(self.marco, self.polos)
    
    def display_diagnostics(self):

        print(f"Time: {self.time}")
        if self.diagnostics[0]:
            
            marco_action_rewards = [(reward, action) for action, reward in self.marco.lastActionRewardPairs.items()]
            marco_action_rewards.sort(key=lambda x: x[0], reverse=True)
            
            print("Marco Best Action: " + str(marco_action_rewards[0][1]) + " with reward: " + str(marco_action_rewards[0][0]))
            print("Marco Second Best Action: " + str(marco_action_rewards[1][1]) + " with reward: " + str(marco_action_rewards[1][0]))
        
        for i in range(1, len(self.diagnostics)):
            if self.diagnostics[i]:
                if len(self.polos[i].lastActionRewardPairs.items()) > 0:
                    polo_action_rewards = [(reward, action) for action, reward in self.polos[i].lastActionRewardPairs.items()]
                    polo_action_rewards.sort(key=lambda x: x[0], reverse=True)

                    print(f"Polo {i} Best Action: {polo_action_rewards[0][1]} with reward: {polo_action_rewards[0][0]}")
                    print(f"Polo {i} Second Best Action: {polo_action_rewards[1][1]} with reward: {polo_action_rewards[1][0]}")
                else:
                    print(f"Polo {i} has no actions")

        self.display_belief_grid(self.marco)

    def display_belief_grid(self, player):
        for row in player.beliefGrid:
            line = "".join(
                SYMBOLS[round(cell * 8) / 8]
                for cell in row
            )
            print(line)
            
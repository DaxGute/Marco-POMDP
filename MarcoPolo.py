import math
import random

from physics.pool import Pool
from physics.sound import Sound
from players.seeker import Seeker
from players.hider import Hider

SYMBOLS = {
    0: "▯",
    0.25: "░",
    0.5: "▒",
    0.75: "▓",
    1: "█",
}

class MarcoPolo:

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
        
        if action == "yell":
            self.round_yell = True
        else:
            (dx, dy) = action
            new_x = self.marco.pos[0] + dx
            new_y = self.marco.pos[1] + dy
            
            self.marco.pos = (new_x, new_y)
          

    def iterate_round(self):
        self.pool.time += 1

        self.simulate_marco_action()

        if self.has_won():
            return True

        observations = []
        for polo in self.polos:
            sound = self.simulate_polo_action(polo)

            observation = [(sound.pos[0], sound.pos[1], sound.observed_sound_loudness(self.marco.pos))]
            polo.beliefGrid = polo.get_updated_belief_grid(observation)

            (pos, loudness) = sound.observed_sound(self.marco.pos)

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
            print(marco_action_rewards)
        
        # for i in range(1, len(self.diagnostics)):
        #     if self.diagnostics[i]:
        #         if len(self.polos[i].lastActionRewardPairs.items()) > 0:
        #             polo_action_rewards = [(reward, action) for action, reward in self.polos[i].lastActionRewardPairs.items()]
        #             polo_action_rewards.sort(key=lambda x: x[0], reverse=True)

        #             print(f"Polo {i} Best Action: {polo_action_rewards[0][1]} with reward: {polo_action_rewards[0][0]}")
        #             print(f"Polo {i} Second Best Action: {polo_action_rewards[1][1]} with reward: {polo_action_rewards[1][0]}")
        #         else:
        #             print(f"Polo {i} has no actions")

        print("Marco's belief grid:")
        self.display_belief_grid(self.marco.beliefGrid)

        closest_polo = None
        closest_distance = float('inf')
        for polo in self.polos:
            distance = math.hypot(polo.pos[0] - self.marco.pos[0], polo.pos[1] - self.marco.pos[1])
            if distance < closest_distance:
                closest_distance = distance
                closest_polo = polo
        print(f"Closest polo: {closest_polo.pos} with distance: {closest_distance}")
        self.display_belief_grid(closest_polo.beliefGrid)


    def display_belief_grid(self, beliefGrid):
        grid = beliefGrid
    
        eps = 1e-12

        logs = [[math.log(max(cell, eps)) for cell in row] for row in grid]

        min_log = min(min(row) for row in logs)
        max_log = max(max(row) for row in logs)
        span = max_log - min_log

        for row in logs:
            line = ""
            for v in row:
                z = (v - min_log) / span
                
                level = round(z * 4) / 4
                line += SYMBOLS[level]
            print(line)
            
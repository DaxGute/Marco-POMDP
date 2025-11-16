import math
import random

from physics.pool import Pool
from physics.sound import Sound
from players.seeker import Seeker
from players.hider import Hider


class MarcoPolo:
    """
    Game orchestrator: owns players, runs rounds, and uses physics primitives.
    """

    def __init__(self, pool_name: str, num_polos: int):
        self.pool = Pool(pool_name)
        self.num_polos = num_polos

        self.init_players()

        self.last_round_yell = False

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

    def iterate_round(self):
        """
        Advance the game by one round.
        Returns True if Marco catches a polo, False otherwise.
        """
        self.pool.time += 1

        sounds = []
        for polo in self.polos:
            sound = None

            dist = math.hypot(polo.pos[0] - self.marco.pos[0], polo.pos[1] - self.marco.pos[1])

            if not self.last_round_yell or dist < 2:
                (dx, dy) = polo.choose_action()
                polo.pos = (polo.pos[0] + dx, polo.pos[1] + dy)
                sound = self.pool.get_action_sound(polo.pos, (dx, dy))
            else:
                sound = self.pool.get_action_sound(polo.pos, "yell")

            loudness = sound.observed_sound_loudness(self.marco.pos)
            polo.beliefGrid = polo.get_updated_belief_grid([(sound.pos[0], sound.pos[1], loudness)])
            sounds.append(sound)

        self.last_round_yell = False

        observations = []
        for sound in sounds:
            (pos, loudness) = sound.observed_sound(self.marco.pos)
            observations.append((pos[0], pos[1], loudness))
        self.marco.beliefGrid = self.marco.get_updated_belief_grid(observations)

        action = self.marco.choose_action()
        if action == "yell":
            self.last_round_yell = True
        else:
            (dx, dy) = action
            self.marco.pos = (self.marco.pos[0] + dx, self.marco.pos[1] + dy)

        for polo in self.polos:
            if polo.pos == self.marco.pos:
                return True
        return False

    def render(self):
        """Render the current game state."""
        self.pool.render(self.marco, self.polos)
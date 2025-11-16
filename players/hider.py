from players.player import Player
from physics.pool import Pool
import math


class Hider(Player):
    def __init__(self, x: int, y: int, pool: Pool):
        super().__init__(x, y, pool)

    def choose_action(self):
        actions = self.get_actions()
        best_action = actions[0]
        best_reward = -float("inf")
        for action in actions:
            dx, dy = action

            origin = (self.pos[0] + dx, self.pos[1] + dy)
            sound = self.pool.get_action_sound(origin, (dx, dy))

            loudness = sound.observed_sound_loudness(self.game.marco.pos)
            newBeliefGrid = self.get_updated_belief_grid([(origin[0], origin[1], loudness)])

            reward = -self.get_reward(newBeliefGrid)
            if reward <= best_reward:
                best_reward = reward
                best_action = action

        return best_action



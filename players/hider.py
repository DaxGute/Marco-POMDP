from players.player import Player
from physics.pool import Pool
import math
from copy import deepcopy
from physics.sound import get_perceived_likelihood_grid


class Hider(Player):
    def __init__(self, x: int, y: int, pool: Pool):
        super().__init__(x, y, pool)
        self.beliefGrid = self.initialize_belief_grid()


    def get_actions(self):
        otherPlayers = [polo for polo in self.game.polos if polo != self]
        otherPlayers.append(self.game.marco)

        actions = deepcopy(super().get_actions())
        for action in super().get_actions():
            pos = (self.pos[0] + action[0], self.pos[1] + action[1])

            for player in otherPlayers:
                if player.pos == pos:
                    actions.remove(action)
                    break
            

        return actions


    def choose_action(self):
        actions = self.get_actions()
        best_action = actions[0]
        best_reward = -float("inf")

        self.lastActionRewardPairs = {}
        for action in actions:
            dx, dy = action

            origin = (self.pos[0] + dx, self.pos[1] + dy)
            sound = self.pool.get_action_sound(origin, (dx, dy))

            loudness = sound.observed_sound_loudness(self.game.marco.pos)
            newBeliefGrid = self.get_updated_belief_grid(self.beliefGrid, (origin[0], origin[1], loudness))
                
            reward = -self.get_reward(newBeliefGrid, self.game.marco.pos)
            self.lastActionRewardPairs[(action)] = reward
            if reward >= best_reward:
                best_reward = reward
                best_action = action

        return best_action



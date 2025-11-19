from players.player import Player
from physics.pool import Pool
import math
from copy import deepcopy


class Hider(Player):
    def __init__(self, x: int, y: int, pool: Pool):
        super().__init__(x, y, pool)
        self.l1 = 1e5   # certainty (now normalized to [0,1])
        self.l2 = 1e2   # distance
        self.l3 = 1e10   # capture
        self.l4 = 1e1   # time


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

    def get_reward(self, beliefGrid, seekerPos):
        certaintyReward = 0
        distanceReward = 0
        capturedReward = 0
        timeReward = -self.game.time

        for i in range(len(beliefGrid)):
            for j in range(len(beliefGrid[0])):
                if beliefGrid[i][j] > 0:
                    certaintyReward += -beliefGrid[i][j] ** 2

        distance = math.sqrt((self.pos[0] - seekerPos[0]) ** 2 + (self.pos[1] - seekerPos[1]) ** 2)
        if distance < 1:
            capturedReward = 1
        else:
            distanceReward = (1 / distance)

        compositeReward = (
            self.l1 * certaintyReward
            + self.l2 * distanceReward
            + self.l3 * capturedReward
            + self.l4 * timeReward
        )

        return compositeReward

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
            newBeliefGrid = self.get_updated_belief_grid([(origin[0], origin[1], loudness)])
                
            reward = self.get_reward(newBeliefGrid, self.game.marco.pos)
            self.lastActionRewardPairs[(action)] = reward
            if reward >= best_reward:
                best_reward = reward
                best_action = action

        return best_action



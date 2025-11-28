import copy
import math
from physics.pool import Pool
from physics.sound import Sound, get_perceived_likelihood_grid



class Player:
    def __init__(self, x: int, y: int, pool: Pool):
        self.pos = (x, y)
        self.pool = pool
        
        self.l1 = 1e4   # certainty (now normalized to [0,1])
        self.l2 = 1e2   # distance
        self.l3 = 1e3   # capture
        self.l4 = 1e1   # time

        self.lastActionRewardPairs = {}


    def initialize_belief_grid(self):
        totalWaterCells = 0
        for i, row in enumerate(self.pool.grid):
            for j, cell in enumerate(row):
                if cell == 0 and (i, j):
                    totalWaterCells += 1

        beliefPrior = 1 / totalWaterCells 

        beliefGrid = []
        for i in range(len(self.pool.grid)):
            beliefGrid.append([])
            for j in range(len(self.pool.grid[0])):
                if self.pool.grid[i][j] == 0:
                    beliefGrid[i].append(beliefPrior)
                else:
                    beliefGrid[i].append(0)
        return beliefGrid


    def normalize_belief_grid(self, beliefGrid):
        z = 0
        for i in range(len(beliefGrid)):
            for j in range(len(beliefGrid[0])):
                z += beliefGrid[i][j]
        for i in range(len(beliefGrid)):
            for j in range(len(beliefGrid[0])):
                beliefGrid[i][j] /= z
        return beliefGrid


    def get_reward(self, beliefGrid, seekerPos):
        certaintyReward = 0
        distanceReward = 0
        capturedReward = 0
        timeReward = -self.game.time

        for i in range(len(beliefGrid)):
            for j in range(len(beliefGrid[0])):
                if beliefGrid[i][j] > 0:
                    certaintyReward += beliefGrid[i][j] ** 2

                    distance = math.sqrt((i - seekerPos[0]) ** 2 + (j - seekerPos[1]) ** 2)
                    if distance < 1:
                        capturedReward += beliefGrid[i][j]
                    else:
                        distanceReward += beliefGrid[i][j] * (1 / distance)

        compositeReward = (
            self.l1 * certaintyReward
            + self.l2 * distanceReward
            + self.l3 * capturedReward
            + self.l4 * timeReward
        )

        return compositeReward


    def get_actions(self):
        """Return all available movement actions (still, slow, fast)."""
        potential_actions = []
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                if abs(dx) + abs(dy) < 4:
                    potential_actions.append((dx, dy))

        available_actions = []
        for action in potential_actions:
            new_x = self.pos[0] + action[0]
            new_y = self.pos[1] + action[1]
            if self.pool.in_bounds(new_x, new_y):
                available_actions.append(action)

        return available_actions

    
    def get_updated_belief_grid(self, beliefGrid, observation):
        H = len(beliefGrid)
        W = len(beliefGrid[0])
        newBeliefGrid = self.get_diffused_prior_belief_grid(beliefGrid, observation[2])

        (px, py, loudness) = observation
        if loudness < 0.001:
            return newBeliefGrid

        L = get_perceived_likelihood_grid(
            (self.game.marco.pos[0], self.game.marco.pos[1]), # observer position
            (px, py), # perceived position
            loudness, # perceived loudness
            (H, W), # grid shape
        )
        
        for i in range(H):
            for j in range(W):
                newBeliefGrid[i][j] *= L[i][j]

        return self.normalize_belief_grid(newBeliefGrid)


    def get_diffused_prior_belief_grid(self, beliefGrid, loudness):
        actions_liklihoods = self.pool.get_perceived_sound_actions_liklihoods(loudness)
        newBeliefGrid = copy.deepcopy(beliefGrid)

        for i in range(len(newBeliefGrid)):
            for j in range(len(newBeliefGrid[0])):
                for action in actions_liklihoods:
                    if action == "yell":
                        continue
                    dx, dy = action
                    new_x = i - dx
                    new_y = j - dy
                    if self.pool.in_bounds(new_x, new_y):
                        newBeliefGrid[i][j] += actions_liklihoods[action] * beliefGrid[new_x][new_y]

        return self.normalize_belief_grid(newBeliefGrid)
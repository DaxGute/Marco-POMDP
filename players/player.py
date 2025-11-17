import math
from physics.pool import Pool
from physics.sound import Sound, get_perceived_likelihood_grid


class Player:
    def __init__(self, x: int, y: int, pool: Pool):
        self.pos = (x, y)
        self.pool = pool

        self.l1 = 1e5   # certainty (now normalized to [0,1])
        self.l2 = 1e5   # distance
        self.l3 = 1e10   # capture
        self.l4 = 1e1   # time

        self.diffusion_rate = 0.85

        self.beliefGrid = self.initialize_belief_grid()

        self.observation_history = []

        self.lastActionRewardPairs = {}

    def initialize_belief_grid(self):
        totalWaterCells = 0
        for i, row in enumerate(self.pool.grid):
            for j, cell in enumerate(row):
                if cell == 0 and (i, j) != (self.pos[0], self.pos[1]):
                    totalWaterCells += 1

        beliefPrior = 1 / totalWaterCells 

        beliefGrid = []
        for i in range(len(self.pool.grid)):
            beliefGrid.append([])
            for j in range(len(self.pool.grid[0])):
                if self.pool.grid[i][j] == 0 and (i, j) != (self.pos[0], self.pos[1]):
                    beliefGrid[i].append(beliefPrior)
                else:
                    beliefGrid[i].append(0)
        return beliefGrid

    def get_reward(self, beliefGrid, seekerPos):
        certaintyReward = 0
        distanceReward = 0
        capturedReward = 0
        timeReward = -self.game.time

        for i in range(len(beliefGrid)):
            for j in range(len(beliefGrid[0])):
                if beliefGrid[i][j] > 0:
                    certaintyReward += beliefGrid[i][j] * math.log(beliefGrid[i][j])

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

    def normalize_belief_grid(self, beliefGrid):
        z = 0
        for i in range(len(beliefGrid)):
            for j in range(len(beliefGrid[0])):
                z += beliefGrid[i][j]
        for i in range(len(beliefGrid)):
            for j in range(len(beliefGrid[0])):
                beliefGrid[i][j] /= z
        return beliefGrid

    
    def get_updated_belief_grid(self, observations):
        H = len(self.beliefGrid)
        W = len(self.beliefGrid[0])
        newBeliefGrid = self.diffuse_belief_grid(self.beliefGrid)

        if observations:
            for (px, py, loudness) in observations:
                L = get_perceived_likelihood_grid(
                    observer_pos=self.pos,
                    perceived_pos=(px, py),
                    perceived_loudness=loudness,
                    grid_shape=(H, W),
                )
                for i in range(H):
                    for j in range(W):
                        newBeliefGrid[i][j] = max(newBeliefGrid[i][j], L[i][j])
                        if self.pool.grid[i][j] == 1:
                            newBeliefGrid[i][j] = 0

        return self.normalize_belief_grid(newBeliefGrid)

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

    def diffuse_belief_grid(self, beliefGrid):
        H = len(beliefGrid)
        W = len(beliefGrid[0])
        newBeliefGrid = [row[:] for row in beliefGrid]

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                newBeliefGrid[i][j] = 0
                totalNeighbors = 0
                totalProbability = 0
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if self.pool.in_bounds(i + dx, j + dy):
                            totalNeighbors += 1
                            totalProbability += beliefGrid[i + dx][j + dy]

                if totalNeighbors > 0:
                    newBeliefGrid[i][j] = totalProbability / totalNeighbors
                else:
                    newBeliefGrid[i][j] = 0

        return self.normalize_belief_grid(newBeliefGrid)
                
import math
from physics.pool import Pool
from physics.sound import Sound, get_perceived_likelihood_grid



class Player:
    def __init__(self, x: int, y: int, pool: Pool):
        self.pos = (x, y)
        self.pool = pool


        self.beliefGrid = self.initialize_belief_grid()

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

    
    def get_updated_belief_grid(self, observations):
        H = len(self.beliefGrid)
        W = len(self.beliefGrid[0])
        newBeliefGrid = self.diffuse_belief_grid(self.beliefGrid)

        
        if observations:

            L_total = [[0.0 for _ in range(W)] for _ in range(H)]
            for (px, py, loudness) in observations:
                if loudness < 0.001:
                    continue
                L = get_perceived_likelihood_grid(
                    observer_pos=(self.game.marco.pos[0], self.game.marco.pos[1]),
                    perceived_pos=(px, py),
                    perceived_loudness=loudness,
                    grid_shape=(H, W),
                )
                for i in range(H):
                    for j in range(W):
                        L_total[i][j] += L[i][j]


            alpha = 0.4

            for i in range(H):
                for j in range(W):
                    if self.pool.grid[i][j] == 1:
                        newBeliefGrid[i][j] = 0
                    else:
                        newBeliefGrid[i][j] = alpha * L_total[i][j] + (1 - alpha) * newBeliefGrid[i][j]

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
                
import math
from physics.pool import Pool
from physics.sound import Sound, get_perceived_likelihood_grid


class Player:
    def __init__(self, x: int, y: int, pool: Pool):
        self.pos = (x, y)
        self.pool = pool

        self.l1 = 1e2  # certainty reward
        self.l2 = 1e1  # distance reward
        self.l3 = 1e5  # captured reward
        self.l4 = 1e1  # time reward

        self.beliefGrid = self.initialize_belief_grid()


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

    def get_reward(self, beliefGrid):
        certaintyReward = 0
        distanceReward = 0
        capturedReward = 0
        timeReward = -self.pool.time

        for i in range(len(beliefGrid)):
            for j in range(len(beliefGrid[0])):
                if beliefGrid[i][j] > 0:
                    certaintyReward += -beliefGrid[i][j] * math.log(beliefGrid[i][j])

                    distance = math.sqrt((i - self.pos[0]) ** 2 + (j - self.pos[1]) ** 2)
                    if distance <= 1:
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

    def get_updated_belief_grid(self, observations):
        """
        beliefGrid[i][j] = P(at least one hider in cell (i,j)) before hearing new sounds.
        observations = list of (perceived_x, perceived_y, perceived_loudness)
        """

        if len(observations) == 0:
            return self.beliefGrid

        H = len(self.beliefGrid)
        W = len(self.beliefGrid[0])

        liklihood_grids = []  

        for (x, y, loudness) in observations:

            L = get_perceived_likelihood_grid(
                observer_pos=self.pos,
                perceived_pos=(x, y),
                perceived_loudness=loudness,
                grid_shape=(H, W),
            )

            Z = 0.0
            for i in range(H):
                for j in range(W):
                    L[i][j] = self.beliefGrid[i][j] * L[i][j]
                    Z += L[i][j]

            # normalize to get s_k(i,j)
            L = [[L[i][j] / Z for j in range(W)] for i in range(H)]
            liklihood_grids.append(L)


        newBeliefGrid = [[0.0 for _ in range(W)] for i in range(H)]
        for i in range(H):
            for j in range(W):

                prior = self.beliefGrid[i][j] 

                no_hider_prob = 1.0
                for l in liklihood_grids:
                    no_hider_prob *= (1.0 - l[i][j])

                newBeliefGrid[i][j] = 1.0 - (1.0 - prior) * no_hider_prob

        return newBeliefGrid


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



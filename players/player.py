import copy
import math
from physics.pool import Pool
from physics.sound import Sound, get_perceived_likelihood_grid

SYMBOLS = {
    0: "▯",
    0.25: "░",
    0.5: "▒",
    0.75: "▓",
    1: "█",
}

class Player:
    def __init__(self, x: int, y: int, pool: Pool):
        self.pos = (x, y)
        self.pool = pool

        self.lastActionRewardPairs = {}

        self.beliefGrid = self.initialize_belief_grid()


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
                beliefGrid[i][j] = max(beliefGrid[i][j], 1e-100)
        return beliefGrid



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

        (px, py, loudness) = observation


        newBeliefGrid = self.get_diffused_prior_belief_grid(beliefGrid, loudness)

        L = get_perceived_likelihood_grid(
            (self.game.marco.pos[0], self.game.marco.pos[1]), # observer position
            (px, py), # perceived position
            loudness, # perceived loudness
            (H, W), # grid shape
        )
        L = self.normalize_belief_grid(L)

        for i in range(H):
            for j in range(W):
                newBeliefGrid[i][j] *= L[i][j]**3

        newBeliefGrid[self.game.marco.pos[0]][self.game.marco.pos[1]] = 0

        return self.normalize_belief_grid(newBeliefGrid)

    # def diffuse_sharp_belief_grid(self, beliefGrid):
    #     max_likelihood = max(max(row) for row in beliefGrid)
    #     if max_likelihood_at_peak > 0.8:  
    #         beliefGrid_diffused = [[0.0 for _ in range(W)] for _ in range(H)]
    #         for i in range(H):
    #             for j in range(W):
    #                 # Add neighboring cells
    #                 for di in [-1, 0, 1]:
    #                     for dj in [-1, 0, 1]:
    #                         ni, nj = i + di, j + dj
    #                         if 0 <= ni < H and 0 <= nj < W:
    #                             L_diffused[i][j] += L[ni][nj] * 0.2  # Spread 20% to neighbors
    #                 beli_diffused[i][j] += L[i][j] * 0.8  # Keep 80% at original
    #     return newBeliefGrid

    def get_diffused_prior_belief_grid(self, beliefGrid, loudness):
        newBeliefGrid = [[0.0 for _ in row] for row in beliefGrid]

        for i in range(len(newBeliefGrid)):
            for j in range(len(newBeliefGrid[0])):
                if (i, j) == self.game.marco.pos:
                    continue
                if  not self.pool.in_bounds(i,j):
                    continue

                dist = math.hypot(i - self.game.marco.pos[0], j - self.game.marco.pos[1])
                source_loudness = loudness * (dist * dist)
                actions_liklihoods = self.pool.get_perceived_sound_actions_liklihoods(source_loudness)


                for action in actions_liklihoods:

                    if action == "yell":
                        dx, dy = 0, 0
                    else:   
                        dx, dy = action

                    source_x = i - dx
                    source_y = j - dy
                    if self.pool.in_bounds(source_x, source_y):
                        newBeliefGrid[i][j] += actions_liklihoods[action] * beliefGrid[source_x][source_y]

        return self.normalize_belief_grid(newBeliefGrid)

    def doggalicious_display_belief_grid(self, beliefGrid):
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


    def display_belief_grid(self):
        grid = self.beliefGrid
    
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

        
    def display_action_rewards(self):
        if not self.lastActionRewardPairs:
            print(f"I have no action rewards.")
            return
        
        action_rewards = [(reward, action) for action, reward in self.lastActionRewardPairs.items()]
        action_rewards.sort(key=lambda x: x[0], reverse=True)
        
        print(f"Taken Action: {action_rewards[0][1]} with reward: {action_rewards[0][0]}")
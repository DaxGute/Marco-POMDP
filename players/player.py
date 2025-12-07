import copy
import math
from physics.pool import Pool
from physics.sound import Sound, get_perceived_likelihood_grid
import numpy as np

SYMBOLS = {
    0: "▯",
    0.25: "░",
    0.5: "▒",
    0.75: "▓",
    1: "█",
}

class Player:
    def __init__(self, x: int, y: int, pool: Pool, game):
        self.pos = (x, y)
        self.pool = pool
        self.game = game
        
        self.lastActionRewardPairs = {}

        # Defer belief grid initialization - it requires marco to exist
        # It will be initialized by the game class after all players are created
        self.beliefGrid = None


    def initialize_belief_grid(self):
        H, W = len(self.game.pool.grid), len(self.game.pool.grid[0])
        mx, my = self.game.marco.pos

        distances = []
        for i in range(H):
            for j in range(W):
                if self.pool.grid[i][j] == 0:
                    d = math.hypot(i - mx, j - my)
                    distances.append(d)

        d_min = min(distances)
        d_max = max(distances)
        span = max(d_max - d_min, 1e-9)

        beliefGrid = []
        total = 0.0
        for i in range(H):
            row = []
            for j in range(W):
                if self.pool.grid[i][j] == 0:
                    d = math.hypot(i - mx, j - my)
                    val = (d - d_min) / span
                    row.append(val)
                    total += val
                else:
                    row.append(0)
            beliefGrid.append(row)
        for i in range(H):
            for j in range(W):
                beliefGrid[i][j] /= total

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
        beliefGrid = np.array(beliefGrid)
        (H, W) = beliefGrid.shape

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
                newBeliefGrid[i][j] *= L[i][j]

        newBeliefGrid[self.game.marco.pos[0]][self.game.marco.pos[1]] = 0

        return self.normalize_belief_grid(newBeliefGrid)

    def get_diffused_prior_belief_grid(self, beliefGrid, loudness):
        beliefGrid = np.array(beliefGrid)
        H, W = beliefGrid.shape
        newBeliefGrid = np.zeros((H, W))
        
        marco_pos = self.game.marco.pos
    
        i_grid, j_grid = np.mgrid[0:H, 0:W]
        
        dist = np.hypot(i_grid - marco_pos[0], j_grid - marco_pos[1])
        source_loudness = loudness * (dist * dist)
        
        for i in range(H):
            for j in range(W):
                if (i, j) == marco_pos or not self.pool.in_bounds(i, j):
                    continue
                
                actions_liklihoods = self.pool.get_perceived_sound_actions_liklihoods(
                    source_loudness[i, j]
                )
                
                for action, likelihood in actions_liklihoods.items():
                    if action == "yell":
                        dx, dy = 0, 0
                    else:
                        dx, dy = action
                    
                    source_x, source_y = i - dx, j - dy
                    if self.pool.in_bounds(source_x, source_y):
                        newBeliefGrid[i, j] += likelihood * beliefGrid[source_x, source_y]
        
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
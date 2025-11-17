import math
from physics.pool import Pool
from physics.sound import Sound, get_perceived_likelihood_grid


class Player:
    def __init__(self, x: int, y: int, pool: Pool):
        self.pos = (x, y)
        self.pool = pool

        self.l1 = 1e7   # certainty (now normalized to [0,1])
        self.l2 = 1e6   # distance
        self.l3 = 1e8   # capture
        self.l4 = 1e1   # time

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
        current_time = self.game.time
        
        print(f"Time {current_time}: Received {len(observations)} observations")
        print(f"History size before: {len(self.observation_history) if hasattr(self, 'observation_history') else 'NOT INITIALIZED'}")
        
        # Add new observations
        for obs_x, obs_y, obs_loudness in observations:
            obs_pos = (obs_x, obs_y)
            self.observation_history.append({
                'pos': obs_pos,
                'loudness': obs_loudness,
                'time': current_time,
                'weight': 1.0
            })
        
        print(f"History size after adding: {len(self.observation_history)}")
        
        # Decay all observations
        decay_rate = 0.85
        for obs in self.observation_history:
            age = current_time - obs['time']
            obs['weight'] = decay_rate ** age
        
        # Remove very old/weak observations
        self.observation_history = [obs for obs in self.observation_history 
                                if obs['weight'] > 0.01]
        
        print(f"History size after pruning: {len(self.observation_history)}")
        
        # Build belief grid from weighted observations
        H, W = len(self.pool.grid), len(self.pool.grid[0])
        grid = [[0.0 for _ in range(W)] for _ in range(H)]
        
        for obs in self.observation_history:
            L = get_perceived_likelihood_grid(
                self.pos, obs['pos'], obs['loudness'], (H, W)
            )

            # In get_updated_belief_grid, add after creating L:
            print(f"Observation at {obs['pos']} loudness {obs['loudness']:.2f} weight {obs['weight']:.4f}")
            max_L = max(max(row) for row in L)
            print(f"  Max likelihood: {max_L:.6f}")
            for i in range(H):
                for j in range(W):
                    grid[i][j] += obs['weight'] * L[i][j]
        
        # Normalize
        grid[self.pos[0]][self.pos[1]] = 0.0
        Z = sum(sum(row) for row in grid)
        
        print(f"Total probability mass before normalization: {Z}")
        
        if Z > 0:
            grid = [[cell/Z for cell in row] for row in grid]
        else:
            # Uniform if no information
            print("WARNING: No probability mass, returning uniform!")
            grid = [[1.0/(H*W) if self.pool.grid[i][j] == 0 else 0.0 
                    for j in range(W)] for i in range(H)]
            grid[self.pos[0]][self.pos[1]] = 0.0
        
        return grid

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

    def diffuse_belief(self, grid, steps=1):
        H, W = len(grid), len(grid[0])
        
        # Amount of diffusion per step (tune this based on polo speed)
        diffusion_per_step = 0.15  # 15% spreads to neighbors each turn
        
        current_grid = [row[:] for row in grid]  # Copy
        
        for _ in range(min(steps, 5)):  # Cap diffusion steps to avoid over-spreading
            new_grid = [[0.0 for _ in range(W)] for _ in range(H)]
            
            for i in range(H):
                for j in range(W):
                    if self.pool.grid[i][j] == 1:  # Wall - no probability here
                        continue
                    
                    prob = current_grid[i][j]
                    if prob <= 0:
                        continue
                    
                    # Keep some probability at current cell
                    keep_amount = 1.0 - diffusion_per_step
                    new_grid[i][j] += prob * keep_amount
                    
                    # Spread remaining probability to valid neighbors
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            # Check bounds and not a wall
                            if (0 <= ni < H and 0 <= nj < W and 
                                self.pool.grid[ni][nj] != 1):
                                neighbors.append((ni, nj))
                    
                    if neighbors:
                        spread_per_neighbor = (prob * diffusion_per_step) / len(neighbors)
                        for ni, nj in neighbors:
                            new_grid[ni][nj] += spread_per_neighbor
            
            current_grid = new_grid
        
        return current_grid



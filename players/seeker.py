import copy
from players.player import Player
from physics.pool import Pool
import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import random
from physics.sound import get_perceived_likelihood_grid, Sound, get_actual_sound_likelihood


class Seeker(Player):
    def __init__(self, x: int, y: int, pool: Pool, num_polos: int):
        super().__init__(x, y, pool)

        self.beliefGrids = []
        for i in range(num_polos):
            self.beliefGrids.append(self.initialize_belief_grid())


    def get_actions(self):
        available_actions = super().get_actions()

        available_actions.append("yell")
        available_actions.remove((0,0))

        return available_actions

    def get_reward(self, beliefGrids, seekerPos):
        total_reward = 0
        for i in range(len(beliefGrids)):
            total_reward += super().get_reward(beliefGrids[i], seekerPos)

        return total_reward

    def choose_action(self):
        actions = self.get_actions()
        best_action = None
        best_reward = -float("inf")
        self.lastActionRewardPairs = {}

        for action in actions:
            if action != "yell":
                if not self.pool.in_bounds(self.pos[0] + action[0], self.pos[1] + action[1]):
                    continue

            newBeliefGrids = self.beliefGrids
            seekerPos = self.pos
            if action == "yell":
                newBeliefGrids = self.expected_yelling_belief_grid()
            else:
                seekerPos = (seekerPos[0] + action[0], seekerPos[1] + action[1])
                
            reward = self.get_reward(newBeliefGrids, seekerPos)

            self.lastActionRewardPairs[(action)] = reward
            if reward >= best_reward:
                best_reward = reward
                best_action = action

        return best_action

    def get_updated_belief_grids(self, observations):
        assignments = self.assign_observations(observations)
        newBeliefGrids = copy.deepcopy(self.beliefGrids)

        for hider_idx, obs_idx in assignments:
            newBeliefGrids[hider_idx] = super().get_updated_belief_grid(newBeliefGrids[hider_idx], observations[obs_idx])

        return newBeliefGrids

    def assign_observations(self, observations): 
        n_hiders = len(self.beliefGrids)
        n_obs = len(observations)

        cost_matrix = np.zeros((n_hiders, n_obs))

        H, W = len(self.beliefGrids[0]), len(self.beliefGrids[0][0])

        for obs_idx, (px, py, loudness) in enumerate(observations):
            L = get_perceived_likelihood_grid(
                (self.game.marco.pos[0], self.game.marco.pos[1]),
                (px, py),
                loudness,
                (H, W),
            )

            for hider_idx in range(n_hiders):
                beliefGrid = self.get_diffused_prior_belief_grid(self.beliefGrids[hider_idx], loudness)
                likelihood = np.sum(np.array(beliefGrid) * np.array(L))

                cost_matrix[hider_idx, obs_idx] = -likelihood
        
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        assignments = list(zip(row_idx, col_idx))

        return assignments


    def expected_yelling_belief_grid(self):
        H, W = len(self.beliefGrids[0]), len(self.beliefGrids[0][0])

        observations = []
        for beliefGrid in self.beliefGrids:

            coords = []
            weights = []

            for i in range(H):
                for j in range(W):
                    w = beliefGrid[i][j]
                    coords.append([i, j])
                    weights.append(w)

            total_weight = sum(weights)
            centroid_x = sum(weights[i] * coords[i][0] for i in range(len(coords))) / total_weight
            centroid_y = sum(weights[i] * coords[i][1] for i in range(len(coords))) / total_weight

            center = [int(centroid_x), int(centroid_y)]

            dist = math.hypot(center[0] - self.pos[0], center[1] - self.pos[1])
            
            if dist < 2:
                sound = self.pool.get_action_sound(center, (0, 0))
            else:
                sound = self.pool.get_action_sound(center, "yell")
            loudness = sound.observed_sound_loudness(self.pos)

            observations.append((center[0], center[1], loudness))

        newBeliefGrids = self.get_updated_belief_grids(observations)

        return newBeliefGrids
import copy
from players.player import Player
from physics.pool import Pool
from sklearn.cluster import KMeans
import math
import random
from physics.sound import get_perceived_likelihood_grid, Sound


class Seeker(Player):
    def __init__(self, x: int, y: int, pool: Pool):
        super().__init__(x, y, pool)


    def get_actions(self):
        """Seeker can move (still, slow, fast) or yell 'seeker'."""
        available_actions = super().get_actions()

        available_actions.append("yell")
        available_actions.remove((0,0))

        return available_actions

    def choose_action(self):
        actions = self.get_actions()
        best_action = None
        best_reward = -float("inf")
        self.lastActionRewardPairs = {}

        for action in actions:
            if action != "yell":
                if not self.pool.in_bounds(self.pos[0] + action[0], self.pos[1] + action[1]):
                    continue

            newBeliefGrid = self.beliefGrid
            seekerPos = self.pos
            if action == "yell":
                newBeliefGrid = self.expected_yelling_belief_grid()
            else:
                seekerPos = (seekerPos[0] + action[0], seekerPos[1] + action[1])
                
            reward = self.get_reward(newBeliefGrid, seekerPos)

            self.lastActionRewardPairs[(action)] = reward
            if reward >= best_reward:
                best_reward = reward
                best_action = action

        return best_action

    def expected_yelling_belief_grid(self):
        k = len(self.game.polos)
        coords = []
        weights = []

        for i in range(len(self.beliefGrid)):
            for j in range(len(self.beliefGrid[0])):
                w = self.beliefGrid[i][j]
                coords.append([i, j])
                weights.append(w)

        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
        )

        kmeans.fit(coords, sample_weight=weights)
        centers = kmeans.cluster_centers_
        centers = [(round(center[0]), round(center[1])) for center in centers]
        
        observations = []
        for center in centers:
            dist = math.hypot(center[0] - self.pos[0], center[1] - self.pos[1])
            if dist < 2:
                continue
            sound = self.pool.get_action_sound(center, "yell")
            loudness = sound.observed_sound_loudness(self.pos)
            observations.append((center[0], center[1], loudness))

        newBeliefGrid = self.get_updated_belief_grid(observations)
        return newBeliefGrid
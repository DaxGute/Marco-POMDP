import copy
from players.player import Player
from physics.pool import Pool
from sklearn.cluster import KMeans


class Seeker(Player):
    def __init__(self, x: int, y: int, pool: Pool):
        super().__init__(x, y, pool)


    def get_actions(self):
        """Seeker can move (still, slow, fast) or yell 'seeker'."""
        available_actions = super().get_actions()

        available_actions.append("seeker")

        return available_actions

    def choose_action(self):
        actions = self.get_actions()
        best_action = actions[0]
        best_reward = -float("inf")
        for action in actions:
            newBeliefGrid = self.beliefGrid
            if action == "yell":
                newBeliefGrid = self.expected_yelling_belief_grid()
            
                
            reward = self.get_reward(newBeliefGrid)
            if reward >= best_reward:
                best_reward = reward
                best_action = action

        return best_action

    def expected_yelling_belief_grid(self):
        k = self.pool.num_hiders

        coords = []
        weights = []

        for i in range(len(self.beliefGrid)):
            for j in range(len(self.beliefGrid[0])):
                w = self.beliefGrid[i][j]

                coords.append([i, j])
                weights.append(w)
        
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++"
        )
        kmeans.fit(coords, sample_weight=weights)

        centers = kmeans.cluster_centers_
        centers = [(round(center[0]), round(center[1])) for center in centers]

        observations = []
        for center in centers:
            observations.append(self.pool.get_action_sound(center, "yell"))
        
        newBeliefGrid = self.get_updated_belief_grid(observations)
        return newBeliefGrid



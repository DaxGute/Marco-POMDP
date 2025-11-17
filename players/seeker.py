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
        from physics.sound import get_perceived_likelihood_grid
        
        k = len(self.game.polos)
        print(f"Expected yelling: need {k} polos")
        
        # Sample from current belief grid
        coords = []
        weights = []
        for i in range(len(self.beliefGrid)):
            for j in range(len(self.beliefGrid[0])):
                w = self.beliefGrid[i][j]
                if w > 0:
                    coords.append((i, j))
                    weights.append(w)
        
        print(f"Found {len(coords)} valid cells to sample from")
        
        if not coords:
            print("No valid coords, returning current belief")
            return self.beliefGrid
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        sampled_centers = random.choices(coords, weights=normalized_weights, k=k)
        
        print(f"Sampled centers: {sampled_centers}")
        
        # Remove duplicates
        sampled_centers = list(set(sampled_centers))
        print(f"Unique centers: {sampled_centers}")
        
        # Generate expected observations from sampled positions
        observations = []
        for center in sampled_centers:
            dist = math.hypot(center[0] - self.pos[0], center[1] - self.pos[1])
            print(f"Center {center} is at distance {dist} from {self.pos}")
            if dist < 2:
                print(f"  Skipped (too close)")
                continue
            sound = self.pool.get_action_sound(center, "yell")
            loudness = sound.observed_sound_loudness(self.pos)
            print(f"  Would generate observation at {center} with loudness {loudness}")
            observations.append((center[0], center[1], loudness))
        
        print(f"Total observations: {len(observations)}")
        
        # If no observations (all too close), return current belief
        if not observations:
            print("No observations generated, returning current belief")
            return self.beliefGrid
        
        # Use existing get_updated_belief_grid method to process observations
        print("Updating belief grid with observations...")
        newBeliefGrid = self.get_updated_belief_grid(observations)
        
        # Check if belief changed
        diff = sum(abs(self.beliefGrid[i][j] - newBeliefGrid[i][j]) 
                for i in range(len(self.beliefGrid))
                for j in range(len(self.beliefGrid[0])))
        print(f"Belief changed by total difference: {diff}")
        
        return newBeliefGrid
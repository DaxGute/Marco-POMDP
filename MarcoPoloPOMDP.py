from MarcoPolo import MarcoPolo
import copy
import math

class MarcoPoloPOMDP():
    def __init__(self, pool_name, num_polos, diagnostics = False, locations = []):
        
        self.game = MarcoPolo(pool_name, num_polos, diagnostics)
        self.depth = 3
        self.num_branches = 3

    def hider_agnostic_clone(self, game):
        game = copy.copy(game)
        game.marco = copy.copy(game.marco)
        
        game.polos = [copy.copy(polo) for polo in game.polos]
        
        game.marco.lastActionRewardPairs = game.marco.lastActionRewardPairs.copy()
        for polo in game.polos:
            polo.lastActionRewardPairs = polo.lastActionRewardPairs.copy()
        
        game.marco.beliefGrids = [[row[:] for row in bg] for bg in game.marco.beliefGrids]
        
        for idx, belief_grid in enumerate(game.marco.beliefGrids):
            H, W = len(game.pool.baseGrid), len(game.pool.baseGrid[0])
            coords = []
            weights = []
            
            for i in range(H):
                for j in range(W):
                    w = belief_grid[i][j]
                    coords.append([i, j])
                    weights.append(w)
            
            centroid_x = sum(weights[i] * coords[i][0] for i in range(len(coords))) 
            centroid_y = sum(weights[i] * coords[i][1] for i in range(len(coords))) 
            centroid = (int(centroid_x), int(centroid_y))
            
            game.polos[idx].pos = centroid
            game.polos[idx].beliefGrid = [row[:] for row in belief_grid]
        
        return game

    def seeker_agnostic_clone(self, game):
        game = copy.copy(game)
        game.marco = copy.copy(game.marco)
        
        game.polos = [copy.copy(polo) for polo in game.polos]
        
        game.marco.lastActionRewardPairs = game.marco.lastActionRewardPairs.copy()
        for polo in game.polos:
            polo.lastActionRewardPairs = polo.lastActionRewardPairs.copy()
        
        for i, polo in enumerate(game.polos):
            game.polos[i].beliefGrid = [row[:] for row in polo.beliefGrid]
        
        game.marco.beliefGrids = [
            [row[:] for row in polo.beliefGrid] 
            for polo in game.polos
        ]
            
        return game
                

    def get_best_marco_action_reward(self, game, depth):
        num_branches = min(int(2**depth), self.num_branches)
        game.marco.choose_action()

        action_rewards = [(reward, action) for action, reward in game.marco.lastActionRewardPairs.items()]
        action_rewards.sort(key=lambda x: x[0], reverse=True)

        if depth == 0:
            if len(action_rewards) == 0:
                return (0, 0), game.marco.get_reward(game.marco.beliefGrids, game.marco.pos)
            return action_rewards[0][1], action_rewards[0][0]

        best_action, best_reward = None, float('-inf')
        
        for i in range(min(num_branches, len(action_rewards))):
            marco_action = None
            if game.rounds_since_yell < 3:
                marco_action = (0, 0)
            else:
                marco_action = action_rewards[i][1]
            
            temp = self.hider_agnostic_clone(game)

            if marco_action == "yell":
                temp.rounds_since_yell = 0
            else:
                temp.marco.pos = (temp.marco.pos[0] + marco_action[0], temp.marco.pos[1] + marco_action[1])

            actions = []
            for idx in range(len(temp.polos)):
                action, _ = self.get_best_hider_action_reward(temp, idx, depth - 1)
                actions.append(action)

            self.update_belief_on_polo_actions(temp, actions)
       
            _, reward = self.get_best_marco_action_reward(temp, depth - 1)

            if reward > best_reward:
                best_action = marco_action
                best_reward = reward

        return best_action, best_reward

    def get_best_hider_action_reward(self, game, polo_idx, depth):
        num_branches = min(int(2**depth), self.num_branches)

        game.polos[polo_idx].choose_action()

        action_rewards = [(reward, action) for action, reward in game.polos[polo_idx].lastActionRewardPairs.items()]
        action_rewards.sort(key=lambda x: x[0], reverse=True)

        if depth == 0:
            return action_rewards[0][1], action_rewards[0][0]

        best_action, best_reward = None, float('-inf')
        
        for i in range(min(num_branches, len(action_rewards))):
            actions = []
            if game.rounds_since_yell == 0:
                actions = [(0, 0) for _ in range(len(game.polos))]
            else:
                for idx in range(len(game.polos)):
                    if idx == polo_idx:
                        actions.append(action_rewards[i][1])
                    else:
                        action, _ = self.get_best_hider_action_reward(game, idx, depth - 1)
                        actions.append(action)

            temp = self.seeker_agnostic_clone(game)

            self.update_belief_on_polo_actions(temp, actions)

            marco_action, _ = self.get_best_marco_action_reward(temp, depth - 1)
            if marco_action == "yell":
                temp.rounds_since_yell = 0
            else:
                temp.marco.pos = (temp.marco.pos[0] + marco_action[0], temp.marco.pos[1] + marco_action[1])

            _, reward = self.get_best_hider_action_reward(temp, polo_idx, depth - 1)

            if reward > best_reward:
                best_action = actions[polo_idx]
                best_reward = reward

        return best_action, best_reward
    
    def iterate_round(self):

        if self.game.rounds_since_yell < 3:
            best_action = (0, 0)
        else:
            best_action, _ = self.get_best_marco_action_reward(self.game, self.depth)
        print("Marco action:", best_action)

        if best_action == "yell":
            self.game.rounds_since_yell = 0
        else:
            self.game.marco.pos = (self.game.marco.pos[0] + best_action[0], self.game.marco.pos[1] + best_action[1])

        if self.game.has_won():
            return True

        actions = []
        for i in range(len(self.game.polos)):
            if self.game.rounds_since_yell == 0:
                best_action = (0, 0)
            else:
                best_action, _ = self.get_best_hider_action_reward(self.game, i, self.depth)
            
            self.game.polos[i].pos = (self.game.polos[i].pos[0] + best_action[0], self.game.polos[i].pos[1] + best_action[1])
            actions.append(best_action)
            print("Polo", i+1, "action:", best_action)
        
        self.update_belief_on_polo_actions(self.game, actions)

        return False

    @staticmethod
    def update_belief_on_polo_actions(game, actions):
        marco_observations = []
        for i, action in enumerate(actions):
            sound = game.pool.get_action_sound(game.polos[i].pos, (action[0], action[1])) 
            if game.rounds_since_yell == 0:
                sound = game.pool.get_action_sound(game.polos[i].pos, "yell")

            dist = math.hypot(game.polos[i].pos[0] - game.marco.pos[0], game.polos[i].pos[1] - game.marco.pos[1])
            dist = max(dist, 1)

            polo_observation = (sound.pos[0], sound.pos[1], sound.loudness / (dist * dist))
            game.polos[i].beliefGrid = game.polos[i].get_updated_belief_grid(game.polos[i].beliefGrid, polo_observation)

            (pos, loudness) = sound.observed_sound(game.marco.pos)

            x = max(0, min(int(round(pos[0])), len(game.pool.grid)-1))
            y = max(0, min(int(round(pos[1])), len(game.pool.grid[0])-1))

            marco_observations.append((x, y, loudness))
        
        game.marco.beliefGrids = game.marco.get_updated_belief_grids(marco_observations)

        game.rounds_since_yell += 1
        game.time += 1

    def render(self):
        self.game.render()

    def display_diagnostics(self):
        self.game.display_diagnostics()

from MarcoPolo import MarcoPolo
import copy
import math

class MarcoPoloPOMDP():
    def __init__(self, pool_name, num_polos, diagnostics = False):
        
        self.game = MarcoPolo(pool_name, num_polos, diagnostics)
        self.num_branches = 2
        self.depth = 2

    def hider_agnostic_clone(self, game):
        game = copy.deepcopy(game)

        for idx, belief_grid in enumerate(game.marco.beliefGrids):
            # Calculate centroid of the belief grid
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
            game.polos[idx].beliefGrid = belief_grid
        
        return game

    def seeker_agnostic_clone(self, game):
        game = copy.deepcopy(game)

        for i, polo in enumerate(game.polos):
            game.marco.beliefGrids[i] = polo.beliefGrid

        return game
            

    def get_best_marco_action_reward(self, game, depth):
        game.marco.choose_action()

        action_rewards = [(reward, action) for action, reward in game.marco.lastActionRewardPairs.items()]
        action_rewards.sort(key=lambda x: x[0], reverse=True)

        if depth == 0:
            if len(action_rewards) == 0:
                return (0, 0), game.marco.get_reward(game.marco.beliefGrids, game.marco.pos)
            return action_rewards[0][1], action_rewards[0][0]

        best_action, best_reward = None, float('-inf')
        if game.rounds_since_yell >= 3:
            for i in range(self.num_branches):
                action = action_rewards[i][1]
                
                temp = self.hider_agnostic_clone(game)
                if action == "yell":
                    temp.rounds_since_yell = 0
                else:
                    temp.marco.pos = (temp.marco.pos[0] + action[0], temp.marco.pos[1] + action[1])

                observations = []
                for idx,polo in enumerate(temp.polos):
                    action, reward = self.get_best_hider_action_reward(temp, idx, depth - 1)
                    polo.pos = (polo.pos[0] + action[0], polo.pos[1] + action[1])
                    sound = temp.pool.get_action_sound(polo.pos, action)

                    dist = math.hypot(polo.pos[0] - temp.marco.pos[0], polo.pos[1] - temp.marco.pos[1])
                    dist = max(dist, 1)

                    observation = (sound.pos[0], sound.pos[1], sound.loudness / (dist * dist))
                    observations.append(observation)

                    polo.beliefGrid = polo.get_updated_belief_grid(polo.beliefGrid, observation)
                
                temp.marco.beliefGrids = temp.marco.get_updated_belief_grids(observations)

                temp.rounds_since_yell += 1
                temp.time += 1

                _, reward = self.get_best_marco_action_reward(temp, depth - 1)

                if reward > best_reward:
                    best_action = action
                    best_reward = reward

        else:
            best_action = (0, 0)

            temp = self.hider_agnostic_clone(game)

            observations = []
            for idx, polo in enumerate(temp.polos):
                action, reward = self.get_best_hider_action_reward(temp, idx, depth - 1)
                polo.pos = (polo.pos[0] + action[0], polo.pos[1] + action[1])
                sound = temp.pool.get_action_sound(polo.pos, action)
                
                dist = math.hypot(polo.pos[0] - temp.marco.pos[0], polo.pos[1] - temp.marco.pos[1])
                dist = max(dist, 1)

                observation = (sound.pos[0], sound.pos[1], sound.loudness / (dist * dist))
                observations.append(observation)

                polo.beliefGrid = polo.get_updated_belief_grid(polo.beliefGrid, observation)
            
            temp.marco.beliefGrids = temp.marco.get_updated_belief_grids(observations)

            temp.rounds_since_yell += 1
            temp.time += 1

            _, best_reward = self.get_best_marco_action_reward(temp, depth - 1)

        return best_action, best_reward

    def get_best_hider_action_reward(self, game, polo_idx, depth):

        game.polos[polo_idx].choose_action()

        action_rewards = [(reward, action) for action, reward in game.polos[polo_idx].lastActionRewardPairs.items()]
        action_rewards.sort(key=lambda x: x[0], reverse=True)

        if depth == 0:
            return action_rewards[0][1], action_rewards[0][0]

        best_action, best_reward = None, float('-inf')
        if game.rounds_since_yell != 0:
            for i in range(self.num_branches):
                action = action_rewards[i][1]
                temp = self.seeker_agnostic_clone(game)

                temp.polos[i].pos = (temp.polos[i].pos[0] + action[0], temp.polos[i].pos[1] + action[1])
                sound = game.pool.get_action_sound(temp.polos[i].pos, (action[0], action[1]))

                dist = math.hypot(sound.pos[0] - temp.marco.pos[0], sound.pos[1] - temp.marco.pos[1])
                dist = max(dist, 1)

                observations = []
                observation = (sound.pos[0], sound.pos[1], sound.loudness / (dist * dist))
                observations.append(observation)

                temp.polos[i].beliefGrid = temp.polos[i].get_updated_belief_grid(temp.polos[i].beliefGrid, observation)

                for idx in range(len(temp.polos)):
                    if idx != polo_idx:
                        sound = temp.simulate_polo_action(temp.polos[idx])
                        dist = math.hypot(temp.polos[idx].pos[0] - temp.marco.pos[0], temp.polos[idx].pos[1] - temp.marco.pos[1])
                        dist = max(dist, 1)
                        observation = (sound.pos[0], sound.pos[1], sound.loudness / (dist * dist))
                        observations.append(observation)
                        temp.polos[idx].beliefGrid = temp.polos[idx].get_updated_belief_grid(temp.polos[idx].beliefGrid, observation)

                temp.marco.beliefGrids = temp.marco.get_updated_belief_grids(observations)

                temp.rounds_since_yell += 1
                temp.time += 1

                marco_action, _ = self.get_best_marco_action_reward(temp, depth - 1)
                if marco_action == "yell":
                    temp.rounds_since_yell = 0
                else:
                    temp.marco.pos = (temp.marco.pos[0] + marco_action[0], temp.marco.pos[1] + marco_action[1])

                _, reward = self.get_best_hider_action_reward(temp, polo_idx, depth - 1)

                if reward > best_reward:
                    best_action = action
                    best_reward = reward
            
        else:
            best_action = (0, 0)

            temp = self.seeker_agnostic_clone(game)
            sound = temp.pool.get_action_sound(temp.polos[polo_idx].pos, "yell")

            dist = math.hypot(sound.pos[0] - temp.marco.pos[0], sound.pos[1] - temp.marco.pos[1])
            dist = max(dist, 1)

            observations = []
            observation = (sound.pos[0], sound.pos[1], sound.loudness / (dist * dist))
            observations.append(observation)

            temp.polos[polo_idx].beliefGrid = temp.polos[polo_idx].get_updated_belief_grid(temp.polos[polo_idx].beliefGrid, observation)

            for idx in range(len(temp.polos)):
                if idx != polo_idx:
                    sound = temp.pool.get_action_sound(temp.polos[idx].pos, "yell")
                    dist = math.hypot(temp.polos[idx].pos[0] - temp.marco.pos[0], temp.polos[idx].pos[1] - temp.marco.pos[1])
                    dist = max(dist, 1)
                    observation = (sound.pos[0], sound.pos[1], sound.loudness / (dist * dist))
                    observations.append(observation)
                    temp.polos[idx].beliefGrid = temp.polos[idx].get_updated_belief_grid(temp.polos[idx].beliefGrid, observation)

            temp.marco.beliefGrids = temp.marco.get_updated_belief_grids(observations)

            temp.rounds_since_yell += 1
            temp.time += 1

            action, _ = self.get_best_marco_action_reward(temp, depth - 1)
            if action == "yell":
                temp.rounds_since_yell = 0
            else:
                temp.marco.pos = (temp.marco.pos[0] + action[0], temp.marco.pos[1] + action[1])

            _, best_reward = self.get_best_hider_action_reward(temp, polo_idx, depth - 1)

        return best_action, best_reward
    
    def iterate_round(self):
        self.game.pool.time += 1

        best_action, _ = self.get_best_marco_action_reward(self.game, self.depth)
        print("Marco action:", best_action)
        if best_action == "yell":
            self.game.rounds_since_yell = 0
        else:
            self.game.marco.pos = (self.game.marco.pos[0] + best_action[0], self.game.marco.pos[1] + best_action[1])

        if self.game.has_won():
            return True

        observations = []
        for i in range(len(self.game.polos)):
            best_action, _ = self.get_best_hider_action_reward(self.game, i, self.depth)
            print("Polo", i, "action:", best_action)
            self.game.polos[i].pos = (self.game.polos[i].pos[0] + best_action[0], self.game.polos[i].pos[1] + best_action[1])
            if self.game.rounds_since_yell == 0:
                sound = self.game.pool.get_action_sound(self.game.polos[i].pos, "yell")
            else:
                sound = self.game.pool.get_action_sound(self.game.polos[i].pos, (best_action[0], best_action[1]))


            dist = math.hypot(self.game.polos[i].pos[0] - self.game.marco.pos[0], self.game.polos[i].pos[1] - self.game.marco.pos[1])
            dist = max(dist, 1)

            observation = (sound.pos[0], sound.pos[1], sound.loudness / (dist * dist))
            self.game.polos[i].beliefGrid= self.game.polos[i].get_updated_belief_grid(self.game.polos[i].beliefGrid, observation)

            (pos, loudness) = sound.observed_sound(self.game.marco.pos)

            x = max(0, min(int(round(pos[0])), len(self.game.pool.grid)-1))
            y = max(0, min(int(round(pos[1])), len(self.game.pool.grid[0])-1))

            observations.append((x, y, loudness))
            
        self.game.marco.beliefGrids = self.game.marco.get_updated_belief_grids(observations)

        self.game.rounds_since_yell += 1

        self.game.time += 1

        return False

    def render(self):
        self.game.render()

    def display_diagnostics(self):
        self.game.display_diagnostics()

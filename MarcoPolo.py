import math
import random

from physics.pool import Pool
from physics.sound import Sound
from players.seeker import Seeker
from players.hider import Hider



class MarcoPolo:

    def __init__(self, pool_name, num_polos, diagnostics = False, locations=None):
        self.pool = Pool(pool_name)
        self.num_polos = num_polos
        self.time = 0

        self.init_players(locations)

        self.rounds_since_yell = 5

        if diagnostics != False:
            self.diagnostics = diagnostics

    def init_players(self, locations):

        if locations is not None:
            marco_location = locations[len(locations) - 1]
            self.marco = Seeker(marco_location[0], marco_location[1], self.pool, self.num_polos, self)
            self.marco.beliefGrids = []
            for i in range(self.num_polos):
                self.marco.beliefGrids.append(self.marco.initialize_belief_grid())
    
            self.polos = []
            for i in range(self.num_polos):
                polo_location = locations[i]
                polo = Hider(polo_location[0], polo_location[1], self.pool, self)
                self.polos.append(polo)
                self.polos[i].beliefGrid = self.polos[i].initialize_belief_grid()
            return
        
        available_positions = []
        for i in range(len(self.pool.grid)):
            for j in range(len(self.pool.grid[0])):
                if self.pool.grid[i][j] == 0:
                    available_positions.append((i, j))
        random.shuffle(available_positions)

        self.marco = Seeker(available_positions[0][0], available_positions[0][1], self.pool, self.num_polos, self)
        self.marco.beliefGrids = []
        for i in range(self.num_polos):
            self.marco.beliefGrids.append(self.marco.initialize_belief_grid())

        self.polos = []
        for i in range(self.num_polos):
            polo = Hider(available_positions[i + 1][0], available_positions[i + 1][1], self.pool, self)
            self.polos.append(polo)
            self.polos[i].beliefGrid = self.polos[i].initialize_belief_grid()

    def has_won(self):
        for polo in self.polos:
            if polo.pos == self.marco.pos:
                return True
        return False

    def simulate_polo_action(self, polo):
        sound = None

        (dx, dy) = polo.choose_action()
        print("Polo action:", (dx, dy))
        polo.pos = (polo.pos[0] + dx, polo.pos[1] + dy)
        sound = self.pool.get_action_sound(polo.pos, (dx, dy))

        if self.rounds_since_yell == 0:  
            sound = self.pool.get_action_sound(polo.pos, "yell")
            (dx, dy) = (0, 0)

        return sound

    def simulate_marco_action(self):
        if self.rounds_since_yell < 3:
            return 

        action = self.marco.choose_action()
        print("Marco action:", action)
        
        if action == "yell":
            self.rounds_since_yell = 0
        else:
            (dx, dy) = action
            new_x = self.marco.pos[0] + dx
            new_y = self.marco.pos[1] + dy
            
            self.marco.pos = (new_x, new_y)
          

    def iterate_round(self):

        self.simulate_marco_action()

        if self.has_won():
            return True

        observations = []
        for polo in self.polos:
            sound = self.simulate_polo_action(polo)

            dist = math.hypot(polo.pos[0] - self.marco.pos[0], polo.pos[1] - self.marco.pos[1])
            dist = max(dist, 1)

            observation = (sound.pos[0], sound.pos[1], sound.loudness / (dist * dist))
            polo.beliefGrid= polo.get_updated_belief_grid(polo.beliefGrid, observation)

            (pos, loudness) = sound.observed_sound(self.marco.pos)

            x = max(0, min(int(round(pos[0])), len(self.pool.grid)-1))
            y = max(0, min(int(round(pos[1])), len(self.pool.grid[0])-1))

            observations.append((x, y, loudness))
            
        self.marco.beliefGrids = self.marco.get_updated_belief_grids(observations)

        self.rounds_since_yell += 1

        self.time += 1

        return False

    def render(self):
        """Render the current game state."""
        self.pool.render(self.marco, self.polos)
    
    def display_diagnostics(self):
        print("--------------------------------")

        print(f"Time: {self.time}")

        print("\nMarco Composite:")
        self.marco.display_belief_grid()
        # self.marco.display_action_rewards()

        for i, polo in enumerate(self.polos):
            print(f"\nPolo {i + 1}:")
            polo.display_belief_grid()
        
        # print("\nClosest Polo:")
        closest_polo = None
        closest_distance = float('inf')
        for polo in self.polos:
            distance = math.hypot(polo.pos[0] - self.marco.pos[0], polo.pos[1] - self.marco.pos[1])
            if distance < closest_distance:
                closest_distance = distance
                closest_polo = polo
        # closest_polo.display_belief_grid()
        # closest_polo.display_action_rewards()

        print("--------------------------------")



    
            
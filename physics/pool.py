import copy
import csv
import math
from physics.sound import Sound, get_actual_sound_likelihood

POOLS_DIR = "pools"

SYMBOLS = {
    0: " ",
    1: "█",
    2: "○",  # Polo
    3: "●",  # Marco
}

SOUND_SCALE = {
    "0": 1e-6,
    "1": 1e2,
    "1.5": 4e2,
    "2": 6e2,
    "2.5": 7.4e2,
    "3": 1.3e3,
    "4": 1e5,
}

SOUND_ACTIONS = {
    (0, 0): SOUND_SCALE["0"],
    (-1, 0): SOUND_SCALE["1"],
    (1, 0): SOUND_SCALE["1"],
    (0, -1): SOUND_SCALE["1"],
    (0, 1): SOUND_SCALE["1"],
    (-1, -1): SOUND_SCALE["1.5"],
    (-1, 1): SOUND_SCALE["1.5"],
    (1, -1): SOUND_SCALE["1.5"],
    (1, 1): SOUND_SCALE["1.5"],
    (2, 0): SOUND_SCALE["2"],
    (0, 2): SOUND_SCALE["2"],
    (-2, 0): SOUND_SCALE["2"],
    (0, -2): SOUND_SCALE["2"],
    (-2,-1): SOUND_SCALE["2.5"],
    (-2,1): SOUND_SCALE["2.5"],
    (2,-1): SOUND_SCALE["2.5"],
    (2,1): SOUND_SCALE["2.5"],
    (1,-2): SOUND_SCALE["2.5"],
    (1,2): SOUND_SCALE["2.5"],
    (-1,-2): SOUND_SCALE["2.5"],
    (-1,2): SOUND_SCALE["2.5"],
    (-2, -2): SOUND_SCALE["3"],
    (-2, 2): SOUND_SCALE["3"],
    (2, -2): SOUND_SCALE["3"],
    (2, 2): SOUND_SCALE["3"],
    "yell": SOUND_SCALE["4"],
}

class Pool:
    """
    Physical pool environment: loads the map and provides geometry helpers.
    Game logic (turns, win condition, etc.) lives in MarcoPolo.
    """

    def __init__(self, pool_name: str):
        self.baseGrid = self.load_pool_csv(f"{POOLS_DIR}/{pool_name}")
        self.grid = copy.deepcopy(self.baseGrid)
        self.time = 0
        
    def load_pool_csv(self, path):
        grid = []
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                grid.append([int(cell) for cell in row])
        return grid

    def in_bounds(self, x: int, y: int) -> bool:
        if x < 0 or x >= len(self.grid) or y < 0 or y >= len(self.grid[0]):
            return False
        return self.baseGrid[x][y] != 1

    def update_grid(self, marco, polos):
        """Overlay player positions onto a copy of the base grid."""
        self.grid = copy.deepcopy(self.baseGrid)
        for polo in polos:
            self.grid[polo.pos[0]][polo.pos[1]] = 2
        self.grid[marco.pos[0]][marco.pos[1]] = 3

    def render(self, marco, polos):
        """Render the pool with the given players."""
        self.update_grid(marco, polos)

        #print("\033[2J\033[H", end="")
        for row in self.grid:
            print("".join(SYMBOLS.get(cell, "?") for cell in row))
        print("\n")

    def get_action_sound(self, pos, action):
        return Sound(pos, SOUND_ACTIONS[action])

    def get_perceived_sound_actions_liklihoods(self, loudness):
        actions_log_liklihoods = {}
    
        for action in SOUND_ACTIONS:
            likelihood = get_actual_sound_likelihood(SOUND_ACTIONS[action], loudness)
            actions_log_liklihoods[action] = math.log(max(likelihood, 1e-300))
        
        # Log-sum-exp trick for normalization
        max_log = max(actions_log_liklihoods.values())
        
        actions_liklihoods = {}
        total = 0.0
        
        for action in actions_log_liklihoods:
            prob = math.exp(actions_log_liklihoods[action] - max_log)
            actions_liklihoods[action] = prob
            total += prob
        
        # Normalize
        for action in actions_liklihoods:
            actions_liklihoods[action] /= total
        
        return actions_liklihoods
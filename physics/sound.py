import math
import random
import math
from typing import List, Tuple

SIGMA_R_SCALE: float = 3.0   # radial uncertainty 
SIGMA_T_SCALE: float = 1.0   # tangential uncertainty 

def get_perceived_likelihood_grid(observer_pos, perceived_pos, perceived_loudness, grid_shape):
   
    ox, oy = observer_pos
    px, py = perceived_pos
    H, W = grid_shape

    vx = px - ox
    vy = py - oy
    dist = max(math.hypot(vx, vy), 1e-6)

    u_r = (vx / dist, vy / dist)
    u_t = (-u_r[1], u_r[0])

    # Uncertainty scale: bigger if farther, smaller if louder
    source_loudness = max(perceived_loudness, 1e-6)
    scale = dist / math.sqrt(source_loudness)

    sigma_r = max(SIGMA_R_SCALE * scale, 1e-6)
    sigma_t = max(SIGMA_T_SCALE * scale, 1e-6)

    likelihood_grid = [[0.0 for _ in range(grid_shape[1])] for _ in range(grid_shape[0])]

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            dx = i - px  
            dy = j - py  

            r = dx * u_r[0] + dy * u_r[1]
            t = dx * u_t[0] + dy * u_t[1]

            likelihood = math.exp(-0.5 * (r**2 / sigma_r**2 + t**2 / sigma_t**2))
            likelihood_grid[i][j] = likelihood

    return likelihood_grid


class Sound:

    # tangential and radial uncertainty scales
    sigma_t_scale: float = SIGMA_T_SCALE
    sigma_r_scale: float = SIGMA_R_SCALE

    def __init__(self, pos, loudness: float):

        self.pos = pos
        self.loudness = loudness

    def observed_sound_pos(self, observer_pos):
 
        ox, oy = observer_pos
        sx, sy = self.pos

        dx = sx - ox
        dy = sy - oy
        dist = math.sqrt((sx - ox) ** 2 + (sy - oy) ** 2)
        dist = max(dist, 0.0001)

        scale = dist / math.sqrt(self.loudness)
        sigma_t = max(self.sigma_t_scale * scale, 0.0001)
        sigma_r = max(self.sigma_r_scale * scale, 0.0001)

        # radial direction
        unit_r = (dx / dist, dy / dist)
        # perpendicular (tangential)
        unit_t = (-unit_r[1], unit_r[0])

        r_coord = random.gauss(dist, sigma_r)
        t_coord = random.gauss(0, sigma_t)

        x_offset = r_coord * unit_r[0] + t_coord * unit_t[0]
        y_offset = r_coord * unit_r[1] + t_coord * unit_t[1]

        x_obv = round(ox + x_offset)
        y_obv = round(oy + y_offset)

        return (x_obv, y_obv)

    def observed_sound_loudness(self, observer_pos):
        sx, sy = self.pos
        ox, oy = observer_pos

        dx = sx - ox
        dy = sy - oy
        dist = math.sqrt(dx*dx + dy*dy)
        dist = max(dist, 0.0001)

        # physical inverse-square falloff
        loudness_observed = self.loudness / (dist * dist)
        return loudness_observed

    def observed_sound(self, observer_pos):
        if self.loudness == 0:
            return (None, None)
        return (
            self.observed_sound_pos(observer_pos),
            self.observed_sound_loudness(observer_pos),
        )

    def __str__(self):
        return f"Sound(loudness={self.loudness})"

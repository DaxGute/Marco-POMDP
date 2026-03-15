# Marco-POMDP (Python)
The goal of this project is to model the children’s game Marco Polo as a multi-agent POMDP in which a blind seeker attempts to locate and tag several hiders moving in a discrete 2D pool. The core difficulty is that the seeker receives only noisy, indirect information: swimming noise and “Polo” responses. The resulting system reflects a highly non-trivial partially observable environment with simultaneous multi-agent decision making.

#### 1. Environment and Geometry (CSV-based Pool Loader)

The pool is defined by CSV maps, where 0 denotes swimmable water and 1 denotes obstacles. At load time, the system constructs a Pool object with both a baseGrid (environment) and an overlay grid used for rendering.

#### 2. Physics-Based Sound Model

Each action produces a Sound object whose perceived location and loudness degrade with distance using an anisotropic Gaussian noise model. Therefore, a hider who swims faster emits proportionally louder and more informative sound. The likelihood grid for a sound is computed by projecting every pool cell into the radial–tangential basis relative to the observed sound direction and applying a 2D Gaussian.

#### 3. Belief-State Management

Each agent maintains a full probability distribution over the pool (beliefGrid). Updates occur via a combination of:

Diffusion for modeling uncertainty in player motion

Bayesian-style likelihood integration where sound likelihood grids are blended with priors.

For the seeker’s “Marco” action, I implemented an expected observation model: the seeker runs weighted k-means++ clustering over its own belief grid to estimate the most probable hider locations, simulates the hypothetical “Polo” responses, and constructs an expected post-yell belief grid.

#### 4. Agent Reward Models and Action Selection

Both seekers and hiders perform single-step lookahead evaluation using composite rewards:

Certainty reward: encourages reducing entropy in the belief about other players.

Distance reward: encourages approaching (for the seeker) or increasing distance (for the hider).

Capture term: strongly rewards states where the seeker ends adjacent to a hider.

Time penalty: discourages wasting turns.

The tuning of these reward coefficients—and the correct coupling between belief updates and reward evaluation—is the main bottleneck moving forward.

#### 5. End-to-End Gameplay Functioning

The simulation now runs complete rounds. This establishes a fully functional testbed for policy iteration, heuristic search, or learning-based extensions.

#### 6. POMDP Lookahead Planning

Action selection is extended with a finite-horizon POMDP-style lookahead search that simulates possible future interactions between the seeker and the hiders. The planner recursively evaluates action sequences up to a fixed depth (currently depth = 3) while pruning the search tree by considering only the top-rewarded candidate actions at each step. At every node, the current game state is cloned into an “agnostic” hypothetical state, where hidden agents are replaced with point estimates derived from belief distributions (e.g., centroids of belief grids). This allows the planner to simulate outcomes without revealing true hidden positions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Follow the prompt to select a pool shape CSV from `pools/`.

## Files

- `main.py`: CLI entrypoint; lists available pool layouts and runs the interactive Marco/Polo simulation loop.
- `MarcoPoloPOMDP.py`: POMDP-style planner/wrapper around `MarcoPolo`; performs lookahead search to choose actions.
- `MarcoPolo.py`: Core game environment/state (pool, players, turn iteration, win condition, diagnostics).
- `players/player.py`: Shared `Player` base class (action space + belief grid update machinery).
- `players/seeker.py`: The seeker (“Marco”) policy and belief-grid management (includes optional observation assignment scaffolding).
- `players/hider.py`: The hider (“Polo”) policy and belief updates.
- `physics/pool.py`: Pool grid loader/renderer + movement validity + sound/action likelihood helpers.
- `physics/sound.py`: Sound observation model and likelihood functions.
- `pools/*.csv`: Pool layouts (0 = free cell, 1 = wall) loaded by `physics/pool.py`.
- `requirements.txt`: Python dependencies (`numpy`, `scipy`).

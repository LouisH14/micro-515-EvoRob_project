# Challenge 3 Theoretical Answers (Detailed Report)

This document provides a detailed theoretical analysis for Challenge 3 (co-evolution of body and controller on hilly terrain), based on the current implementation in the repository.

## Q3.0 Understanding the World

### 1) What is the genome size of the SO2-oscillator?

There are two useful notions of genome size:

- Controller-only SO2 genome size:
  - The SO2 controller defines:
    - `n_weights`: number of active oscillator/inter-oscillator weights
    - `2 * num_dofs`: initial oscillator state values
  - In the current implementation, `num_dofs = 8` and `n_weights = 22`, so:
  - `n_params = n_weights + 2 * num_dofs = 22 + 16 = 38`

- Full co-evolution genome size (controller + morphology):
  - `n_body_params = 8` (upper/lower segments for 4 legs)
  - `n_total = n_controller + n_body = 38 + 8 = 46`

Interpretation:

- 38 controls the neural oscillator dynamics.
- 8 controls morphology (leg segment lengths).

### 2) What are the minimum and maximum leg lengths?

Morphology parameters are mapped as:

- `body_params = (genotype[self.n_weights:] + 1) / 4 + 0.1`

Assuming genotype values are sampled or optimized in `[-1, 1]`:

- Minimum at `g = -1`: `(-1 + 1) / 4 + 0.1 = 0.1`
- Maximum at `g = 1`: `(1 + 1) / 4 + 0.1 = 0.6`

So each body length parameter is in:

- `[0.1, 0.6]` meters (in model units)

### 3) What are the components in the reward function?

Current reward in `AntHillEnv.step` is:

- `reward = healthy_reward + forward_reward - ctrl_cost - cfrc_cost`

Components:

- `forward_reward`: forward velocity along x-axis (progression)
- `healthy_reward`: constant survival term (currently set to `1` each step)
- `ctrl_cost`: control effort penalty, proportional to squared action norm
- `cfrc_cost`: contact force penalty

Interpretation:

- Encourages locomotion speed and survival
- Penalizes high actuation effort and potentially aggressive contacts

### 4) What are the stopping criteria for the environment simulator?

The simulation can stop due to:

- Numerical instability in MuJoCo accelerations:
  - if `qacc` contains NaN, Inf, or extremely large values (`abs > 1e6`)
- Torso height outside a healthy interval:
  - terminated when `qpos[2] < 0.2` or `qpos[2] > 1.0`
- Time-limit truncation:
  - via Gymnasium wrapper max episode steps

Interpretation:

- Prevents meaningless evaluations from unstable physics states
- Removes solutions that collapse or become physically unrealistic

## Q3.1 Parallel Evaluation

### 1) How many times do we parallelize a single individual?

In `evaluate_individual`, one genotype is evaluated across:

- `n_repeats = 10` vectorized environments

So each individual is tested in 10 parallel rollouts.

### 2) How is the final fitness computed over multiple parallel environments?

Process:

1. For each environment, rewards are accumulated over time (`sum` over steps).
2. This gives one episode return per parallel rollout.
3. Final scalar fitness is the mean return across the 10 rollouts.
4. For multi-objective, each objective is also summed over time then averaged over rollouts.

So fitness estimator is a sample mean across repeated episodes.

### 3) Why compute fitness this way?

Benefits:

- Reduces variance from stochastic reset noise and trajectory randomness
- Gives a more reliable estimate of true policy quality
- Reduces over-selection of individuals that got lucky in a single seed
- Improves evolutionary selection pressure by using more stable rankings

In short, this is a robustness-oriented estimator.

### 4) What does SO2 controller reset do?

For SO2:

- `reset_controller(batch_size=n_repeats)` replicates internal oscillator initial state for each parallel environment.
- Internal state matrix `y` becomes shape `(2 * num_dofs, batch_size)`.

This ensures each parallel rollout starts from a well-defined oscillator state.

Important detail:

- SO2 is open-loop in this implementation (it ignores observations in action generation), so reset strongly influences early phase dynamics.

## Q3.2 Terrain Parameters and Expected Effects

Current terrain generation combines:

- A global slope map
- Smoothed random bumps/noise map

with parameters `slope_deg`, `bump_scale`, `sigma`.

### 1) What is the effect of changing slope_deg, bump_scale, sigma?

- `slope_deg`:
  - Controls global incline angle via `tan(deg2rad(slope_deg))`
  - Higher value increases uphill gradient and required propulsion/stability

- `bump_scale`:
  - Scales amplitude of local terrain irregularities
  - Higher value creates larger obstacles and stronger local perturbations

- `sigma` (Gaussian filter on noise):
  - Small sigma: high-frequency rough terrain (sharp local bumps)
  - Large sigma: smoother low-frequency undulations

Practical interpretation:

- `slope_deg` tests sustained uphill capability
- `bump_scale` and `sigma` shape disturbance spectrum and foothold unpredictability

### 2) Hypothesis: does evolving on small bumps change outcomes?

Yes, likely significantly.

Expected evolutionary shifts:

- Morphology:
  - Increased stability-oriented structures (possibly altered leg proportions)
  - Better compromise between reach and support

- Control:
  - Lower-frequency, more robust gaits
  - Better disturbance rejection
  - Potentially lower nominal speed but higher consistency

This is a classic robustness vs peak-speed trade-off.

### 3) MLP trained before hill adaptation: can it walk uphill?

Theoretical expectation:

- A controller evolved on flat terrain usually generalizes only partially to slopes.
- Without slope exposure during training, success uphill is often limited and brittle.

### 4) If we make legs longer, can it walk uphill now?

Likely outcome:

- Longer legs may improve obstacle clearance and stride reach.
- But they can also increase torque demands and instability.

So performance can improve, but only within a morphology-control compatibility window. Too long can degrade gait quality.

## Q3.3 Body and Controller Evolution

### 1) What are the current termination conditions in step?

Current termination in `AntHillEnv.step`:

- Numerical invalidity in `qacc` (NaN/Inf/very large)
- Torso z-height outside `[0.2, 1.0]`

### 2) How should termination be altered for hill terrain?

Recommended additions for hill-specific realism:

- Stagnation criterion:
  - terminate if forward displacement is below threshold for many consecutive steps
- Orientation criterion:
  - terminate if torso flips upside down for a sustained interval
- Terrain-aware height criterion:
  - replace fixed global torso z-threshold by threshold relative to local terrain height

Why:

- On slopes, absolute torso height alone can be misleading
- Stuck and flip detection better represent locomotion failure modes

### 3) Can you improve the reward function?

Yes. For hill locomotion, a better objective should explicitly capture climbing quality and efficiency.

A principled shaped reward:

- positive terms:
  - forward progress (`x_velocity`)
  - uphill progress (`z_velocity` or net elevation gain)
  - survival/stability bonus
- negative terms:
  - control effort (`ctrl_cost`)
  - contact shock/slip penalties (`cfrc_cost`, optional slip term)

Example concept:

- `reward = w1 * x_velocity + w2 * max(z_velocity, 0) + w3 * healthy - w4 * ctrl_cost - w5 * cfrc_cost - w6 * slip`

This aligns reward with task success (climb robustly and efficiently).

## Search Space Dimensionality

### CMA-ES with SO2 + body

- SO2 controller params: 38
- Body params: 8
- Total: 46

### CMA-ES with MLP + body

For current MLP setting (`input=27`, `hidden=8`, `output=8`):

- Layer 1 weights: `27 * 8 = 216`
- Layer 2 weights: `8 * 8 = 64`
- Controller total: `216 + 64 = 280`
- Full genome with body: `280 + 8 = 288`

Can we reduce MLP size?

- Yes. Decrease hidden size (for example 8 -> 4).
- This reduces search dimensionality and can improve optimization speed.
- Trade-off: less representational power.

## Bonus: Adaptive Feedback (Hebbian)

### Why is Hebbian adaptation useful?

Hebbian controllers adapt weights online during rollout.

Potential advantages:

- Fast local adaptation to perturbations and terrain variations
- Better resilience to distribution shift (flat-trained vs hill-tested)
- Potentially improved robustness without explicit external memory

Risks:

- Larger search space
- Harder optimization and possible instability if adaptation dynamics are poorly tuned

### What is the dimensionality of Hebbian search space?

For the Hebbian controller:

- Base connection count = `input*hidden + hidden*output`
- With `27, 8, 8`: `216 + 64 = 280`
- Hebbian learns A, B, C, D per connection: `4 * 280 = 1120`
- Add body params 8 -> full co-evolution dimension = `1128`

This is much larger than SO2 and standard MLP setups.

## NSGA-II in this Challenge

### 1) What are the two objectives used now?

The current evaluation uses:

- Objective 1: `reward_forward + healthy_reward`
- Objective 2: `-ctrl_cost`

So NSGA-II trades off locomotion performance vs energetic efficiency.

### 2) Can we think of better objectives?

Yes. Stronger hill-specific alternatives include:

- Objective A: net elevation gain (or final height reached)
- Objective B: energy efficiency, for example progress per control cost

Optional robust objective set:

- maximize uphill progress
- minimize control effort
- maximize robustness across random seeds/terrain perturbations

This improves alignment with real locomotion goals on uneven terrain.

## Suggested Final Discussion Points (for README)

If you need a concise scientific discussion section, focus on:

- Why multi-objective optimization is relevant for body-brain co-design
- How Pareto front reveals speed vs efficiency vs robustness trade-offs
- Why morphology and controller are coupled on hills (not independently optimal)
- Why robustness-aware evaluation (multiple repeats) improves reproducibility

## Practical Note

The current code still contains TODO slots in `main()` and some reward/termination placeholders. The theoretical answers above reflect the present implementation and the intended challenge direction, so they can be used as a basis for your final report and then updated with your empirical findings.

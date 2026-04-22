import os
import xml.etree.ElementTree as xml
from os.path import join
from tempfile import TemporaryDirectory
from PIL import Image
import scipy.ndimage

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import imageio
import numpy as np
from gymnasium.vector import AsyncVectorEnv

from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import get_last_checkpoint_dir, get_project_root
from evorob.world.base import World
from evorob.world.envs.ant_flat import AntFlatEnvironment
from evorob.world.robot.controllers.mlp import NeuralNetworkController
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

"""
    Final Project: Multi-task robot evolution across 4 environments.
    Genotype encodes both controller weights and body morphology.

    Environments
    ------------
    1  Flat terrain   (ant_flat_terrain.xml)
    2  Ice terrain    (ant_ice_terrain.xml)
    3  Hilly terrain  (AntHill-v0, heightmap generated at runtime)
    4  Mystery terrain (not disclosed — commented out, unlock when released)
"""

ROOT_DIR = get_project_root()
ENV_HILL = "AntHill-v0"
MAX_EPISODE_STEPS = 1000  # DO NOT CHANGE for evaluation


# ---------------------------------------------------------------------------
# World: body + brain co-evolution across multiple terrains
# ---------------------------------------------------------------------------

class FinalWorld(World):
    """Evaluates one robot morphology+controller genotype on all terrains."""

    def __init__(self):
        action_space = 8
        state_space = 27

        # TODO: Choose your controller (SO2Controller / NeuralNetworkController /
        #       HebbianController). Switching here affects the genotype size.
        self.controller = NeuralNetworkController(
            input_size=state_space,
            output_size=action_space,
            hidden_size=action_space,
        )

        self.n_weights = self.controller.n_params
        self.n_body_params = 8          # 4 legs × (upper, lower)
        self.n_params = self.n_weights + self.n_body_params

        self.temp_dir = TemporaryDirectory()
        self.world_file = join(self.temp_dir.name, "FinalRobot.xml")
        self.base_xml_path = join(
            ROOT_DIR, "evorob", "world", "robot", "assets", "hill_world.xml"
        )

        self.joint_limits = [
            [-30, 30], [30, 70],
            [-30, 30], [-70, -30],
            [-30, 30], [-70, -30],
            [-30, 30], [30, 70],
        ]
        self.joint_axis = [
            [0, 0, 1], [-1, 1, 0],
            [0, 0, 1], [1, 1, 0],
            [0, 0, 1], [-1, 1, 0],
            [0, 0, 1], [1, 1, 0],
        ]

        self.create_terrain_file("terrain.png")

    # ------------------------------------------------------------------
    # Robot XML generation
    # ------------------------------------------------------------------

    def update_robot_xml(self, genotype: np.ndarray):
        points, connectivity_mat = self.geno2pheno(genotype)
        robot = AntRobot(
            points, connectivity_mat, self.joint_limits, self.joint_axis, verbose=False
        )
        robot.xml = robot.define_robot()
        robot.write_xml(self.temp_dir.name)

        world = xml.parse(self.base_xml_path)
        robot_env = world.getroot()
        robot_env.append(xml.Element("include", attrib={"file": "FinalRobot.xml"}))
        world_xml = xml.tostring(robot_env, encoding="unicode")
        with open(self.world_file, "w") as f:
            f.write(world_xml)

    def geno2pheno(self, genotype: np.ndarray):
        control_weights = genotype[: self.n_weights] * 0.1
        body_params = (genotype[self.n_weights :] + 1) / 4 + 0.1
        assert len(body_params) == self.n_body_params
        assert not np.any(body_params <= 0)
        self.controller.geno2pheno(control_weights)

        (
            fl_leg, fl_ankle,
            fr_leg, fr_ankle,
            bl_leg, bl_ankle,
            br_leg, br_ankle,
        ) = body_params

        def knee(hip, dx, dy, length):
            return np.array([dx * np.sqrt(0.5 * length ** 2),
                             dy * np.sqrt(0.5 * length ** 2), 0]) + hip

        fl_hip = np.array([0.2, 0.2, 0])
        fr_hip = np.array([-0.2, 0.2, 0])
        bl_hip = np.array([-0.2, -0.2, 0])
        br_hip = np.array([0.2, -0.2, 0])

        fl_knee = knee(fl_hip, 1, 1, fl_leg);  fl_toe = knee(fl_knee, 1, 1, fl_ankle)
        fr_knee = knee(fr_hip, -1, 1, fr_leg); fr_toe = knee(fr_knee, -1, 1, fr_ankle)
        bl_knee = knee(bl_hip, -1, -1, bl_leg); bl_toe = knee(bl_knee, -1, -1, bl_ankle)
        br_knee = knee(br_hip, 1, -1, br_leg); br_toe = knee(br_knee, 1, -1, br_ankle)

        points = np.vstack([
            fl_hip, fl_knee, fl_toe,
            fr_hip, fr_knee, fr_toe,
            bl_hip, bl_knee, bl_toe,
            br_hip, br_knee, br_toe,
        ])
        connectivity_mat = np.array([
            [150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        return points, connectivity_mat

    # ------------------------------------------------------------------
    # Terrain file (env 3)
    # ------------------------------------------------------------------

    def create_terrain_file(self, filename="terrain.png", width=400, depth=400):
        slope_deg = 5.0
        bump_scale = 0.1
        sigma = 3.0

        rise = np.tan(np.deg2rad(slope_deg))
        x = np.linspace(0, 1, depth)
        y = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)
        slope_map = X * rise

        rng = np.random.default_rng(42)
        noise = rng.uniform(0, 1, (width, depth))
        noise = scipy.ndimage.gaussian_filter(noise, sigma=sigma)
        noise = (noise - noise.min()) / (noise.max() - noise.min()) * np.tanh(X * 10)
        terrain = np.clip(slope_map + noise * bump_scale, 0, 1)
        terrain[-1, -1] = 1

        img = Image.fromarray((terrain * 255).astype(np.uint8), mode="L")
        img.save(os.path.join(self.temp_dir.name, filename))

    # ------------------------------------------------------------------
    # Per-environment evaluation (single-env, vectorised)
    # ------------------------------------------------------------------

    def _eval_flat_ice(self, n_repeats=4, n_steps=500):
        """Evaluate on flat (first half) and ice (second half) in one batch.

        Returns (mean_flat_reward, mean_ice_reward).
        """
        assert n_repeats % 2 == 0
        half = n_repeats // 2

        def _make(i):
            robot_path = "ant_flat_terrain.xml" if i < half else "ant_ice_terrain.xml"
            # pathlib joins: absolute path wins over prefix → works with abs world_file
            return lambda: AntFlatEnvironment(robot_path=robot_path)

        envs = AsyncVectorEnv([_make(i) for i in range(n_repeats)])
        self.controller.reset_controller(batch_size=n_repeats)
        rewards_full = np.zeros((n_steps, n_repeats))

        obs, _ = envs.reset()
        done_mask = np.zeros(n_repeats, dtype=bool)
        for step in range(n_steps):
            actions = np.where(
                done_mask[:, None], 0, self.controller.get_action(obs)
            )
            obs, rewards, dones, truncated, _ = envs.step(actions)
            rewards_full[step, ~done_mask] = rewards[~done_mask]
            done_mask = done_mask | dones | truncated
            if np.all(done_mask):
                break

        envs.close()
        totals = rewards_full.sum(axis=0)
        return float(totals[:half].mean()), float(totals[half:].mean())

    def _eval_hill(self, n_repeats=4, n_steps=500):
        """Evaluate on hilly terrain. Returns mean reward."""
        envs = AsyncVectorEnv([
            lambda i=i: gym.make(
                ENV_HILL,
                robot_path=self.world_file,
                max_episode_steps=n_steps,
            )
            for i in range(n_repeats)
        ])
        self.controller.reset_controller(batch_size=n_repeats)
        rewards_full = np.zeros((n_steps, n_repeats))

        obs, _ = envs.reset()
        done_mask = np.zeros(n_repeats, dtype=bool)
        for step in range(n_steps):
            actions = np.where(
                done_mask[:, None], 0, self.controller.get_action(obs)
            )
            obs, rewards, dones, truncated, _ = envs.step(actions)
            rewards_full[step, ~done_mask] = rewards[~done_mask]
            done_mask = done_mask | dones | truncated
            if np.all(done_mask):
                break

        envs.close()
        return float(rewards_full.sum(axis=0).mean())

    # def _eval_mystery(self, n_repeats=4, n_steps=500):
    #     """TODO: Evaluate on the mystery terrain (revealed at the end of the course).
    #
    #     Replace the environment name and any terrain-specific setup below.
    #     Return a scalar mean reward over n_repeats parallel runs.
    #     """
    #     raise NotImplementedError("TODO: implement mystery terrain evaluation")

    # ------------------------------------------------------------------
    # Combined fitness (used by NSGA-II)
    # ------------------------------------------------------------------

    def evaluate_individual(self, genotype: np.ndarray, n_repeats=4, n_steps=500):
        """Evaluate one genotype on all active environments.

        Returns a 1-D array of objective values (one per environment).
        Currently active objectives:
            [0] flat terrain reward
            [1] ice terrain reward
            [2] hill terrain reward
            # [3] mystery terrain reward  ← uncomment when env 4 is released
        """
        self.update_robot_xml(genotype)

        # TODO: Design your own reward shaping inside each _eval_* method
        obj_flat, obj_ice = self._eval_flat_ice(n_repeats=n_repeats, n_steps=n_steps)
        obj_hill = self._eval_hill(n_repeats=n_repeats, n_steps=n_steps)

        # obj_mystery = self._eval_mystery(n_repeats=n_repeats, n_steps=n_steps)

        return np.array([obj_flat, obj_ice, obj_hill])
        # return np.array([obj_flat, obj_ice, obj_hill, obj_mystery])  # ← 4-objective


# ---------------------------------------------------------------------------
# Neutral evaluation (leaderboard — uses info fields, ignores custom reward)
# ---------------------------------------------------------------------------

def _run_single_env_episodes(env, controller, n_episodes, seed):
    """Run n_episodes and return per-episode neutral rewards from info."""
    rng = np.random.default_rng(seed)
    controller.reset_controller(batch_size=1)
    episode_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        controller.reset_controller(batch_size=1)
        total = 0.0
        done = False
        while not done:
            action = controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, _, terminated, truncated, info = env.step(action)
            # TODO: adapt neutral formula if your environments expose different info keys
            neutral = (
                float(info.get("healthy_reward", 1.0))
                + float(info.get("x_position", 0.0))
                - float(info.get("ctrl_cost", 0.0))
                - float(info.get("cfrc_cost", 0.0))
            )
            total += neutral
            done = terminated or truncated
        episode_rewards.append(total)

    return episode_rewards


def _record_video(env, controller, seed, out_path, max_steps=MAX_EPISODE_STEPS):
    try:
        controller.reset_controller(batch_size=1)
        obs, _ = env.reset(seed=seed)
        frames, reward = [], 0.0
        for _ in range(max_steps):
            frames.append(env.render())
            action = controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, r, terminated, truncated, _ = env.step(action)
            reward += r
            if terminated or truncated:
                break
        imageio.mimwrite(out_path, frames, fps=20)
        print(f"  Video saved: {out_path}")
        return reward
    except Exception as e:
        print(f"  Video skipped: {e}")
        return None


def _stats(values):
    a = np.asarray(values, dtype=float)
    return dict(
        values=list(values),
        mean=float(np.mean(a)), std=float(np.std(a)),
        best=float(np.max(a)), worst=float(np.min(a)),
    )


def evaluate_checkpoint(checkpoint_dir: str, output_dir: str = "evaluation_output"):
    """Neutral leaderboard evaluation of a final-project checkpoint.

    Loads the best genotype, evaluates 256 episodes per environment using a
    fixed neutral reward formula (from info fields), records one video per
    environment, and writes evaluation_score.txt.

    DO NOT change n_episodes, max_episode_steps, or seed.
    """
    n_episodes: int = 256       # DO NOT CHANGE
    max_episode_steps: int = MAX_EPISODE_STEPS  # DO NOT CHANGE
    seed: int = 0               # DO NOT CHANGE

    # --- load checkpoint ---
    last_gen = get_last_checkpoint_dir(checkpoint_dir)

    def _load(fname):
        for d in ([last_gen] if last_gen else []) + [checkpoint_dir]:
            p = os.path.join(d, fname)
            if os.path.isfile(p):
                return np.load(p, allow_pickle=True)
        return None

    x_best = _load("x_best.npy")
    if x_best is None:
        print(f"ERROR: x_best.npy not found in '{checkpoint_dir}'.")
        return None
    print(f"Loaded x_best (shape: {x_best.shape})")

    world = FinalWorld()
    world.update_robot_xml(x_best)

    # --- environments ---
    terrains = {
        "flat": lambda: AntFlatEnvironment(robot_path="ant_flat_terrain.xml"),
        "ice":  lambda: AntFlatEnvironment(robot_path="ant_ice_terrain.xml"),
        "hill": lambda: gym.make(
            ENV_HILL, robot_path=world.world_file,
            max_episode_steps=max_episode_steps,
        ),
        # "mystery": lambda: ...,  # TODO: uncomment when env 4 is released
    }

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name, make_env in terrains.items():
        print(f"Evaluating {name} ({n_episodes} episodes)...")
        env = make_env()
        rewards = _run_single_env_episodes(
            env, world.controller, n_episodes, seed
        )
        env.close()
        results[name] = _stats(rewards)
        r = results[name]
        print(f"  {name}: mean={r['mean']:.2f} +/- {r['std']:.2f}  best={r['best']:.2f}")

        # video
        env_render = make_env() if name != "hill" else gym.make(
            ENV_HILL, robot_path=world.world_file,
            max_episode_steps=max_episode_steps, render_mode="rgb_array",
        )
        _record_video(
            env_render, world.controller, seed,
            os.path.join(output_dir, f"evaluation_{name}.mp4"),
            max_steps=max_episode_steps,
        )
        env_render.close()

    # --- score file ---
    score_path = os.path.join(output_dir, "evaluation_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Final Project - Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Controller    : {type(world.controller).__name__} ({world.controller.n_params} params)\n")
        f.write(f"Genotype size : {world.n_params}  (weights={world.n_weights}, body={world.n_body_params})\n")
        f.write(f"Checkpoint    : {checkpoint_dir}\n")
        f.write(f"Episodes/env  : {n_episodes}\n")
        f.write(f"Neutral reward: healthy_reward + x_position - ctrl_cost - cfrc_cost\n\n")

        f.write(f"{'Terrain':<10s} {'Mean':>9s} {'Std':>8s} {'Best':>9s} {'Worst':>9s}\n")
        f.write("-" * 50 + "\n")
        for name, r in results.items():
            f.write(f"{name:<10s} {r['mean']:9.2f} {r['std']:8.2f} {r['best']:9.2f} {r['worst']:9.2f}\n")
        f.write("\n")

        for name, r in results.items():
            f.write(f"--- {name.upper()} per-episode ---\n")
            for i, v in enumerate(r["values"]):
                f.write(f"  Episode {i+1:3d}: {v:10.2f}\n")
            f.write("\n")

    print(f"\nScore saved: {score_path}")
    return results


# ---------------------------------------------------------------------------
# Main: NSGA-II multi-task evolution
# ---------------------------------------------------------------------------

def run_multi_task_evolution(
    num_generations: int = 100,
    population_size: int = 100,
    n_parents: int = 100,
    n_repeats: int = 4,
    n_steps: int = 500,
    mutation_prob: float = 0.3,
    crossover_prob: float = 0.5,
    bounds: tuple = (-1, 1),
    ckpt_interval: int = 10,
    results_dir: str = None,
    random_seed: int = 42,
):
    np.random.seed(random_seed)

    world = FinalWorld()
    print(f"Genotype size : {world.n_params}  "
          f"(weights={world.n_weights}, body={world.n_body_params})")

    if results_dir is None:
        results_dir = join(ROOT_DIR, "results", "final_multi")

    # TODO: Tune NSGA-II hyperparameters for your experiment
    ea = NSGAII(
        population_size=population_size,
        n_opt_params=world.n_params,
        n_parents=n_parents,
        num_generations=num_generations,
        bounds=bounds,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        output_dir=results_dir,
    )

    print(f"\nRunning {num_generations} generations, population {population_size}")
    print("Objectives: [flat, ice, hill]")
    print(f"Checkpoints: {results_dir}\n")

    for gen in range(num_generations):
        pop = ea.ask()
        # TODO: Parallelise across individuals if running on a cluster
        fitnesses = np.empty((len(pop), 3))  # 3 objectives (4 when mystery unlocked)
        for idx, genotype in enumerate(pop):
            fitnesses[idx] = world.evaluate_individual(
                genotype, n_repeats=n_repeats, n_steps=n_steps
            )
        ea.tell(pop, fitnesses, save_checkpoint=(gen % ckpt_interval == 0))

    # Final evaluation
    evaluate_checkpoint(
        checkpoint_dir=results_dir,
        output_dir=join(results_dir, "eval"),
    )


if __name__ == "__main__":
    # Quick smoke-test (2 generations, tiny population)
    run_multi_task_evolution(
        num_generations=2,
        population_size=6,
        n_parents=6,
        n_repeats=2,
        n_steps=100,
        ckpt_interval=1,
        results_dir=join(get_project_root(), "results", "final_test"),
    )
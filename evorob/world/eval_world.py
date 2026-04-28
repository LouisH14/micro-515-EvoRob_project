import os
import shutil
import xml.etree.ElementTree as xml
from os.path import join, basename, isfile
from tempfile import TemporaryDirectory

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from evorob.utils.filesys import get_last_checkpoint_dir, get_project_root
from evorob.world.base import World
from evorob.world.robot.controllers.base import Controller
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

ROOT_DIR = get_project_root()
_EVAL_TERRAIN_XML = join(
    ROOT_DIR, "evorob", "world", "envs", "assets", "eval_terrain.xml"
)


class EvalWorld(World):
    """Loads a student's evolved robot and evaluates it on the eval terrain.

    Typical usage
    -------------
    world = EvalWorld()

    # (Optional) swap in your own controller BEFORE calling update_robot_xml:
    # world.set_controller(SO2Controller(input_size=27, output_size=8, hidden_size=8))

    # Option A – you have the robot XML saved from training:
    world.update_robot_xml("/path/to/AntRobot.xml")
    world.geno2pheno(x_best)          # loads controller weights

    # Option B – you only have the checkpoint directory:
    world.load_from_checkpoint("results/final_project")
    """

    def __init__(self):
        self.controller = self._default_controller()
        self.n_weights = self.controller.n_params
        self.n_body_params = 8          # 4 legs × (upper, lower)
        self.n_params = self.n_weights + self.n_body_params

        self.temp_dir = TemporaryDirectory()
        # self.world_file is the tmp copy of eval_terrain.xml with the robot injected
        self.world_file = join(self.temp_dir.name, "eval_terrain.xml")

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

        # Mirror FinalWorld.sensor_fn — set this if your training used a custom
        # sensor function so the eval run sees the same transformed observations.
        self.sensor_fn = None

    # ------------------------------------------------------------------
    # Controller management
    # ------------------------------------------------------------------

    @staticmethod
    def _default_controller():
        from evorob.world.robot.controllers.mlp_sol import NeuralNetworkController
        return NeuralNetworkController(input_size=27, output_size=8, hidden_size=8)

    def set_controller(self, controller: Controller) -> None:
        """Override the default MLP controller.

        Call this BEFORE update_robot_xml / load_from_checkpoint so that the
        genotype is split at the correct boundary.
        """
        self.controller = controller
        self.n_weights = controller.n_params
        self.n_params = self.n_weights + self.n_body_params
        print(f"Controller set: {type(controller).__name__}  ({controller.n_params} params)")

    # ------------------------------------------------------------------
    # Robot XML injection — same pattern as FinalWorld
    # ------------------------------------------------------------------

    def update_robot_xml(self, final_body_path: str) -> None:
        """Inject a robot XML into the eval terrain template and save to temp dir.

        Copies the robot XML next to the world file so MuJoCo can resolve the
        include relative to self.world_file.

        Args:
            final_body_path: Absolute path to the student's robot body XML
                             (e.g. the AntRobot.xml written by FinalWorld).
        """
        robot_filename = basename(final_body_path)
        shutil.copy2(final_body_path, join(self.temp_dir.name, robot_filename))

        world = xml.parse(_EVAL_TERRAIN_XML)
        robot_env = world.getroot()
        robot_env.append(xml.Element("include", attrib={"file": robot_filename}))
        world_xml = xml.tostring(robot_env, encoding="unicode")
        with open(self.world_file, "w") as f:
            f.write(world_xml)

    # ------------------------------------------------------------------
    # Genotype → controller
    # ------------------------------------------------------------------

    def geno2pheno(self, genotype: np.ndarray) -> None:
        """Pass the controller portion of the genotype to controller.geno2pheno().

        No scaling is applied — the controller's own geno2pheno is responsible
        for any necessary transformation of the raw genotype values.

        The body morphology is NOT regenerated here — call update_robot_xml first
        to provide the robot XML, then call geno2pheno to load the controller.
        """
        self.controller.geno2pheno(genotype[:self.n_weights])

    # ------------------------------------------------------------------
    # One-shot loader from a FinalWorld checkpoint
    # ------------------------------------------------------------------

    def load_from_checkpoint(self, checkpoint_dir: str) -> None:
        """Regenerate robot and load controller from a FinalWorld checkpoint.

        Finds x_best.npy, rebuilds AntRobot.xml from the body parameters,
        runs update_robot_xml, and loads the controller weights.

        Args:
            checkpoint_dir: Path to your results directory (e.g. results/final_project).
        """
        last_gen = get_last_checkpoint_dir(checkpoint_dir)

        def _load(fname):
            for d in ([last_gen] if last_gen else []) + [checkpoint_dir]:
                p = join(d, fname)
                if isfile(p):
                    return np.load(p, allow_pickle=True)
            return None

        genotype = _load("x_best.npy")
        if genotype is None:
            raise FileNotFoundError(f"x_best.npy not found in: {checkpoint_dir}")
        print(f"Loaded genotype: shape={genotype.shape}")

        # Rebuild robot body from genotype body parameters
        body_params = (genotype[self.n_weights:] + 1) / 4 + 0.1
        assert len(body_params) == self.n_body_params

        def knee(hip, dx, dy, length):
            return hip + np.array([dx, dy, 0.0]) * np.sqrt(0.5 * length ** 2)

        fl_hip = np.array([0.2, 0.2, 0]);   fr_hip = np.array([-0.2, 0.2, 0])
        bl_hip = np.array([-0.2, -0.2, 0]); br_hip = np.array([0.2, -0.2, 0])

        fl_leg, fl_ankle, fr_leg, fr_ankle, bl_leg, bl_ankle, br_leg, br_ankle = body_params

        fl_knee = knee(fl_hip, 1, 1, fl_leg);    fl_toe = knee(fl_knee, 1, 1, fl_ankle)
        fr_knee = knee(fr_hip, -1, 1, fr_leg);   fr_toe = knee(fr_knee, -1, 1, fr_ankle)
        bl_knee = knee(bl_hip, -1, -1, bl_leg);  bl_toe = knee(bl_knee, -1, -1, bl_ankle)
        br_knee = knee(br_hip, 1, -1, br_leg);   br_toe = knee(br_knee, 1, -1, br_ankle)

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

        robot = AntRobot(
            points, connectivity_mat,
            self.joint_limits, self.joint_axis,
            verbose=False,
        )
        robot.xml = robot.define_robot()
        robot.write_xml(self.temp_dir.name)  # → AntRobot.xml in temp dir

        # Combine terrain template + robot include → self.world_file
        world = xml.parse(_EVAL_TERRAIN_XML)
        root = world.getroot()
        root.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
        with open(self.world_file, "w") as f:
            f.write(xml.tostring(root, encoding="unicode"))

        # Load controller (passes controller slice directly to geno2pheno)
        self.geno2pheno(genotype)

    # ------------------------------------------------------------------
    # Gymnasium env factory
    # ------------------------------------------------------------------

    def create_env(self, render_mode: str = "rgb_array", **kwargs):
        """Return a ready-to-use EvalEnv-v0 gymnasium environment."""
        import gymnasium as gym
        import evorob.world  # ensures EvalEnv-v0 is registered
        return gym.make(
            "EvalEnv-v0",
            robot_path=self.world_file,
            render_mode=render_mode,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Required World abstract methods
    # ------------------------------------------------------------------

    def evaluate_individual(self, genotype: np.ndarray, n_repeats: int = 4,
                            n_steps: int = 500) -> float:
        """Evaluate a genotype on the eval terrain. Returns mean neutral reward."""
        self.geno2pheno(genotype)
        import gymnasium as gym
        import evorob.world

        rewards = []
        for _ in range(n_repeats):
            env = gym.make("EvalEnv-v0", robot_path=self.world_file,
                           max_episode_steps=n_steps)
            self.controller.reset_controller(batch_size=1)
            obs, _ = env.reset()
            total = 0.0
            done = False
            while not done:
                ctrl_obs = self.sensor_fn(obs) if self.sensor_fn is not None else obs
                action = self.controller.get_action(ctrl_obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, _, terminated, truncated, info = env.step(action)
                total += (
                    float(info.get("healthy_reward", 1.0))
                    + float(info.get("x_position", 0.0))
                    - float(info.get("ctrl_cost", 0.0))
                    - float(info.get("cfrc_cost", 0.0))
                )
                done = terminated or truncated
            rewards.append(total)
            env.close()
        return float(np.mean(rewards))

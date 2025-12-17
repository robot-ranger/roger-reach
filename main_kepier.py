from dataclasses import dataclass
import numpy as np
from pathlib import Path
from isaacsim import SimulationApp
from typing import Generator
from scipy.spatial.transform import Rotation as R
from ik_client import IKClient
import spatial_utils as su

simuluation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api import World
from isaacsim.core.cloner import GridCloner
from isaacsim.core.prims import SingleArticulation, XFormPrim
from isaacsim.core.utils.types import ArticulationAction

np.set_printoptions(suppress=True)

project_root = Path(__file__).parent
robot_library = project_root / "robots/library"


def matrix_from_xform(xform: XFormPrim) -> np.ndarray:
    xyz, quat = xform.get_world_poses()
    wxyz = quat[0]
    xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
    rmat = R.from_quat(xyzw).as_matrix()
    m = np.eye(4)
    m[:3, 3] = xyz
    m[:3, :3] = rmat
    return m


def matrix_from_prim_path(prim_path: str) -> np.ndarray:
    xform = XFormPrim(prim_path)
    return matrix_from_xform(xform)


def articulation_from_prim_path(prim_path: str) -> SingleArticulation:
    add_reference_to_stage(str(robot_usd_path), prim_path)
    return SingleArticulation(prim_path)

world = World()

world.scene.add_default_ground_plane() # type: ignore

# setup stage
scene_usd_path = project_root / "../kupier_scene.usd"
assert scene_usd_path.exists()
scene_prim_path = "/World"
add_reference_to_stage(str(scene_usd_path), scene_prim_path)

# # add machine
# machine_prim_path = f"{scene_prim_path}/machine"
# machine = add_reference_to_stage("C:/Users/ted/isaac-sim-standalone-5.1.0-windows-x86_64/workspace/mxi-isaacsim/assets/dm-1_solid_models_04_2025/DM-1_4_2025_2_no_ros.usd", machine_prim_path)
# machine_xform = XFormPrim(prim_paths_expr=machine_prim_path)
# machine_xform.set_world_poses(orientations=np.array([[np.deg2rad(90), 0.0, 0.0, np.deg2rad(90)]]), positions=np.array([[-2.25, 0.75, 0.0]]))
# machine_xform.set_local_scales(np.array([[1., 1., 1.]]))

# robot setup
# robot_usd_path = robot_library / "ABB/CRB15000_10kg_152_v1/CRB15000_10kg_152/CRB15000_10kg_152.usd"
robot_usd_path = robot_library / "ABB/CRB15000_12kg_127_v1/CRB15000_12kg_127/CRB15000_12kg_127.usd"
assert robot_usd_path.exists()
robot_mount_prim_path = f"{scene_prim_path}/robot_mount"
robot_prim_path = f"{robot_mount_prim_path}/robot"
add_reference_to_stage(str(robot_usd_path), robot_prim_path)

tool_usd_path = "C:/Users/ted/isaac-sim-standalone-5.1.0-windows-x86_64/workspace/SCHUNK-1490832 MTB DG-JGP-P 80-1.usd"
tool_prim_path = f"{robot_prim_path}/link_6/flange/tool0"
add_reference_to_stage(str(tool_usd_path), tool_prim_path)
tool_xform = XFormPrim(prim_paths_expr=tool_prim_path)
tool_xform.set_local_poses(orientations=np.array([[0.0, 0.0, 0.0, 0.0]]), translations=np.array([[0.0, 0.0, 0.015]]))
# tool_xform.set_local_scales(np.array([[1., 1., 1.]]))

robot = SingleArticulation(robot_prim_path)

sequence: list[str] = [
    f"{scene_prim_path}/targets/tray1_01/target",
    f"{scene_prim_path}/targets/tray1_02/target",
    f"{scene_prim_path}/targets/tray1_03/target",
    f"{scene_prim_path}/targets/tray1_04/target",
    f"{scene_prim_path}/targets/tray2_01/target",
    f"{scene_prim_path}/targets/tray2_02/target",
    f"{scene_prim_path}/targets/tray2_03/target",
    f"{scene_prim_path}/targets/tray2_04/target",
    f"{scene_prim_path}/machine/face_door/target",
    f"{scene_prim_path}/machine/chuck/target",
    f"{scene_prim_path}/machine/chuck/target",
    f"{scene_prim_path}/machine/face_door/target",
]

tool = [
    {"x": -0.124097, "y": 0.0, "z": 0.136269, "roll": 0.0, "pitch": -45.0, "yaw": 0.0},  # tool0
    {"x": 0.124097, "y": 0.0, "z": 0.136269, "roll": 0.0, "pitch": -45.0, "yaw": -180.} # tool1
]

@dataclass
class Action:
    q: np.ndarray | None  # radians
    dt: float  # seconds
    name: str = "action"

    def with_dt(self, dt: float):
        return Action(q=self.q, dt=dt)

    def get_q(self) -> np.ndarray:
        assert self.q is not None
        return self.q

    @property
    def art(self) -> ArticulationAction | None:
        if self.q is not None:
            return ArticulationAction(self.q)
        else:
            return None

ik_client = IKClient()

def flange_to_tool_T(tool_index: int) -> np.ndarray:
    # meters
    tcp = tool[tool_index]

    # Common robotics convention (ZYX / yaw-pitch-roll):
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    T_FT = (su.tx(tcp["x"]) @ su.ty(tcp["y"]) @ su.tz(tcp["z"])) \
        @ (su.rz_deg(tcp["yaw"]) @ su.ry_deg(tcp["pitch"]) @ su.rx_deg(tcp["roll"]))
    return T_FT

current_tool_index = 0
def action_generator2() -> Generator[Action, None, None]:
    global current_tool_index
    for target_prim_path in sequence:
        T_FT = flange_to_tool_T(current_tool_index)
        T_TF = np.linalg.inv(T_FT)
        world_to_robot = matrix_from_prim_path(robot_mount_prim_path)
        try:
            world_to_target = matrix_from_prim_path(target_prim_path)
        except Exception as e:
            print(f"Error getting matrix for {target_prim_path}: {e}")
            continue

        # robot(base)->target (you are currently treating this as the goal pose)
        T_R_goal = np.linalg.inv(world_to_robot) @ world_to_target

        # Treat target as TOOL goal, and convert to FLANGE goal for IK:
        T_RF_goal = T_R_goal @ T_TF

        

        # q_approach = ik_client.inverse_kinematics(T_RF_app, np.zeros(6))
        q = ik_client.inverse_kinematics(T_RF_goal, np.zeros(6))

        if target_prim_path.endswith("face_door/target"):
            yield Action(q, 1.)
            current_tool_index = 1
        else:
            yield from apply_approach(T_R_goal, T_TF, q)
            current_tool_index = 0

def apply_approach(T_R_goal: np.ndarray, T_TF: np.ndarray, q: np.ndarray) -> Generator[Action, None, None]:
    # Approach along TOOL -Z, then convert to flange for IK:
    T_R_tool_app = T_R_goal @ su.tz(-0.2)
    q_approach = ik_client.inverse_kinematics((T_R_tool_app @ T_TF), np.zeros(6))
    yield Action(q_approach, 1)
    yield Action(q, 0.5)
    yield Action(q_approach, 0.5)


world.reset()
robot.initialize()
i = 0
while True:
    for action in action_generator2():
        if action.art:
            robot.apply_action(action.art)
        for _ in range(int(action.dt * 60)):
            world.step(render=True)
            i += 1
    

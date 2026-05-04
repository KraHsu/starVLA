"""Plan-and-Verify benchmark — data config, embodiment tags, and mixtures.

Mirrors examples/LIBERO/train_files/data_registry/data_config.py so that the PaV
project keeps a self-contained registry (file-level isolation per repo convention).
LIBERO-side fields are duplicated verbatim; Bridge-v2 entries are placeholders that
become live once T-W2.1.2 (convert_bridge_to_lerobot.py) lands.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfigs
# ---------------------------------------------------------------------------
class Libero4in1DataConfig:
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x", "state.y", "state.z",
        "state.roll", "state.pitch", "state.yaw",
        "state.pad", "state.gripper",
    ]
    action_keys = [
        "action.x", "action.y", "action.z",
        "action.roll", "action.pitch", "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = list(range(-16, 0))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


class BridgeWidowxDataConfig:
    """Placeholder for Bridge-v2 (WidowX) — finalize fields once T-W2.1.2 lands.

    Bridge-v2 raw layout (OpenDataLab mirror) typically yields per-step:
      - 1 RGB view (image_0)
      - 7-D action [dx, dy, dz, droll, dpitch, dyaw, gripper]
      - 7-D state  [x, y, z, roll, pitch, yaw, gripper]

    The convert_bridge_to_lerobot.py script must emit a LeRobot v3 dir whose
    modality.json keys match the lists below. Until conversion runs, this class
    is import-safe but not registered in ROBOT_TYPE_CONFIG_MAP — see TODO below.
    """

    video_keys = ["video.image_0"]
    state_keys = [
        "state.x", "state.y", "state.z",
        "state.roll", "state.pitch", "state.yaw",
        "state.gripper",
    ]
    action_keys = [
        "action.x", "action.y", "action.z",
        "action.roll", "action.pitch", "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = list(range(-16, 0))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    # TODO[T-W2.1.2]: enable after convert_bridge_to_lerobot.py is verified
    # "bridge_widowx": BridgeWidowxDataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "libero_franka": EmbodimentTag.FRANKA,
    # TODO[T-W2.1.2]: pick the correct EmbodimentTag for WidowX once conversion lands
    # "bridge_widowx": EmbodimentTag.NEW_EMBODIMENT,
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    # PaV LIBERO 4-suite — identical to the LIBERO `libero_all` key, restated
    # here so PaV configs do not depend on the LIBERO benchmark dir.
    "pav_libero": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "pav_libero_long": [
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    # TODO[T-W2.1.2]: activate pav_full once Bridge-v2 lerobot dir is in place
    # "pav_full": [
    #     ("libero_object_no_noops_1.0.0_lerobot",  1.0, "libero_franka"),
    #     ("libero_goal_no_noops_1.0.0_lerobot",    1.0, "libero_franka"),
    #     ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    #     ("libero_10_no_noops_1.0.0_lerobot",      1.0, "libero_franka"),
    #     ("bridge_orig_1.0.0_lerobot",             1.0, "bridge_widowx"),
    # ],
}

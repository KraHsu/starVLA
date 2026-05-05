import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


def _make_dataset_for_subset(data_fraction: float, subset_seed: int) -> LeRobotSingleDataset:
    dataset = object.__new__(LeRobotSingleDataset)
    dataset.data_cfg = OmegaConf.create(
        {
            "data_fraction": data_fraction,
            "subset_seed": subset_seed,
            "seed": 42,
        }
    )
    dataset._dataset_name = "libero_goal_no_noops_1.0.0_lerobot"
    dataset.delete_pause_frame = False
    dataset._trajectory_ids = np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.int64)
    dataset._trajectory_lengths = np.array([3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
    dataset.trajectory_ids_to_metadata = {
        int(trajectory_id): {"dummy": True} for trajectory_id in dataset._trajectory_ids.tolist()
    }
    return dataset


def test_data_fraction_subset_is_deterministic_for_same_seed():
    dataset_a = _make_dataset_for_subset(data_fraction=0.25, subset_seed=2024)
    dataset_b = _make_dataset_for_subset(data_fraction=0.25, subset_seed=2024)

    dataset_a._apply_data_fraction_filter()
    dataset_b._apply_data_fraction_filter()

    np.testing.assert_array_equal(dataset_a.trajectory_ids, dataset_b.trajectory_ids)
    np.testing.assert_array_equal(dataset_a.trajectory_lengths, dataset_b.trajectory_lengths)
    assert set(dataset_a.trajectory_ids.tolist()) == set(dataset_a.trajectory_ids_to_metadata.keys())
    assert len(dataset_a.trajectory_ids) == 2


def test_data_fraction_subset_changes_with_subset_seed():
    dataset_a = _make_dataset_for_subset(data_fraction=0.25, subset_seed=2024)
    dataset_b = _make_dataset_for_subset(data_fraction=0.25, subset_seed=2025)

    dataset_a._apply_data_fraction_filter()
    dataset_b._apply_data_fraction_filter()

    assert dataset_a.trajectory_ids.tolist() != dataset_b.trajectory_ids.tolist()


def test_steps_cache_key_depends_on_data_fraction_and_subset_seed():
    dataset_a = _make_dataset_for_subset(data_fraction=0.25, subset_seed=2024)
    dataset_b = _make_dataset_for_subset(data_fraction=0.50, subset_seed=2024)
    dataset_c = _make_dataset_for_subset(data_fraction=0.25, subset_seed=2025)

    key_a = dataset_a._get_steps_config_key()
    key_b = dataset_b._get_steps_config_key()
    key_c = dataset_c._get_steps_config_key()

    assert key_a != key_b
    assert key_a != key_c

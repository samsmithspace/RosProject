import os
import sys
import yaml
import numpy as np
from easydict import EasyDict
import torch
from collections import Counter

# Add the project root to Python path to ensure imports work
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pose_estimation.single_cube_dataset import SingleCubeDataset


def inspect_dataset_annotations(config_path, data_root_path):
    """
    Loads the training dataset and analyzes the variation in orientation annotations.

    Args:
        config_path (str): Path to the config.yaml file.
        data_root_path (str): The local path to the root data directory.
    """
    print("--- Starting Data Inspection ---")

    # 1. Load configuration from YAML
    try:
        with open(config_path, 'r') as f:
            config = EasyDict(yaml.safe_load(f))
        print(f"✅ Configuration loaded from '{config_path}'")
    except FileNotFoundError:
        print(f"❌ ERROR: Config file not found at '{config_path}'")
        return

    # 2. Override data_root in config with the provided path
    config.system.data_root = data_root_path
    train_data_path = os.path.join(data_root_path, config.train.dataset_zip_file_name_training)
    print(f"🔍 Checking for training data in: {train_data_path}")

    if not os.path.exists(train_data_path):
        print(f"❌ ERROR: Training data directory not found.")
        print("Please ensure the path is correct and the directory exists.")
        return

    print("✅ Training data directory found.")

    # 3. Instantiate the dataset using the project's own loader
    try:
        dataset = SingleCubeDataset(
            config=config,
            data_root=config.system.data_root,
            split="train",
            zip_file_name=config.train.dataset_zip_file_name_training,
            sample_size=config.train.sample_size_train,  # Use full dataset
        )
        print(f"✅ Dataset object created. Found {len(dataset)} samples.")
    except Exception as e:
        print(f"❌ ERROR: Failed to create SingleCubeDataset object: {e}")
        return

    # 4. Iterate through the dataset to collect annotations
    orientations = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    print("\n--- Processing Samples ---")
    for i, (_, _, target_orient) in enumerate(data_loader):
        # The dataloader returns a batch, so we get the first item
        orientations.append(target_orient.squeeze().numpy())
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} samples...")

    if not orientations:
        print("❌ ERROR: No orientation data could be loaded from the dataset.")
        return

    print(f"✅ Finished processing. Total samples loaded: {len(orientations)}.")

    # 5. Analyze the collected data for variation
    orientations_np = np.array(orientations)

    # Find unique orientations
    unique_orientations = np.unique(orientations_np, axis=0)

    print("\n--- Annotation Analysis Results ---")
    print(f"Total annotations processed: {len(orientations_np)}")
    print(f"Number of unique orientations: {len(unique_orientations)}")

    if len(unique_orientations) == 1:
        print("🔴 All orientation annotations are identical!")
        print(f"The single unique value is: {unique_orientations[0]}")
    elif len(unique_orientations) < 10:
        print("🟡 Very low variation detected in orientation annotations.")
        print("Unique values found:")
        for u in unique_orientations:
            print(f"  {u}")
    else:
        print("🟢 Good variation detected in orientation annotations.")

    # Calculate and display statistics
    stats = {
        'Mean': np.mean(orientations_np, axis=0),
        'Std Dev': np.std(orientations_np, axis=0),
        'Min': np.min(orientations_np, axis=0),
        'Max': np.max(orientations_np, axis=0),
    }

    print("\n--- Quaternion Component Statistics (q_x, q_y, q_z, q_w) ---")
    print(f"{'Stat':<10} | {'q_x':<15} | {'q_y':<15} | {'q_z':<15} | {'q_w':<15}")
    print("-" * 75)
    for stat_name, values in stats.items():
        val_str = " | ".join([f"{v:<15.6f}" for v in values])
        print(f"{stat_name:<10} | {val_str}")
    print("-" * 75)


if __name__ == '__main__':
    # Path to your main config file
    config_file = 'config.yaml'

    # The data_root is the directory containing the training data folder.
    # Updated based on your screenshot. Using forward slashes for compatibility.
    data_root = 'D:/project backup/ROSProject/RosProject/pose_estimation'

    inspect_dataset_annotations(config_file, data_root)
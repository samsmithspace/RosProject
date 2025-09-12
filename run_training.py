#!/usr/bin/env python3
"""
Enhanced standalone training script with flexible data path options.
This script can be run directly to train the pose estimation model.
"""
import os
import sys
import argparse
import yaml
from easydict import EasyDict

# Add the project root to Python path to ensure imports work
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_config(config_file="/scratch/hpc/11/xiar3/RosProject/config.yaml", **overrides):
    """Load configuration from YAML file with optional overrides"""
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Apply overrides
    for key, value in overrides.items():
        if value is not None:
            if '.' in key:
                # Handle nested keys like 'system.data_root'
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value

    # Convert to EasyDict for easier access
    config = EasyDict(config)
    return config


def validate_data_paths(config):
    """Validate that data directories exist"""
    data_root = config.system.data_root
    train_data = os.path.join(data_root, config.train.dataset_zip_file_name_training)
    val_data = os.path.join(data_root, config.val.dataset_zip_file_name_validation)

    print(f"Checking data paths...")
    print(f"Data root: {data_root}")
    print(f"Training data: {train_data}")
    print(f"Validation data: {val_data}")

    if not os.path.exists(data_root):
        print(f"WARNING: Data root directory does not exist: {data_root}")
        print(f"Please create it or specify a different path with --data-root")
        return False

    if not os.path.exists(train_data):
        print(f"WARNING: Training data directory does not exist: {train_data}")
        print(f"Please create it or specify a different name with --train-data-name")
        return False

    if not os.path.exists(val_data):
        print(f"WARNING: Validation data directory does not exist: {val_data}")
        print(f"Please create it or specify a different name with --val-data-name")
        return False

    print("All data paths exist!")
    return True


def main():
    """Main function to run training directly"""
    parser = argparse.ArgumentParser(description='Train Pose Estimation Model')

    # Configuration
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file (default: config.yaml)')

    # Data paths
    parser.add_argument('--data-root', type=str, default=None,
                        help='Root directory containing training/validation folders')
    parser.add_argument('--train-data-name', type=str, default=None,
                        help='Name of training data folder (default: UR3_single_cube_training)')
    parser.add_argument('--val-data-name', type=str, default=None,
                        help='Name of validation data folder (default: UR3_single_cube_validation)')
    parser.add_argument('--train-data-path', type=str, default=None,
                        help='Full path to training data folder (overrides data-root + train-data-name)')
    parser.add_argument('--val-data-path', type=str, default=None,
                        help='Full path to validation data folder (overrides data-root + val-data-name)')

    # Output paths
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory to save logs and model checkpoints')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate for optimizer')

    # Other options
    parser.add_argument('--check-data-only', action='store_true',
                        help='Only check if data paths exist, do not train')
    parser.add_argument('--force', action='store_true',
                        help='Force training even if data paths do not exist')

    args = parser.parse_args()

    # Prepare configuration overrides
    overrides = {}

    if args.data_root:
        overrides['system.data_root'] = args.data_root
    if args.train_data_name:
        overrides['train.dataset_zip_file_name_training'] = args.train_data_name
    if args.val_data_name:
        overrides['val.dataset_zip_file_name_validation'] = args.val_data_name
    if args.log_dir:
        overrides['system.log_dir_system'] = args.log_dir
    if args.epochs:
        overrides['train.epochs'] = args.epochs
    if args.batch_size:
        overrides['train.batch_training_size'] = args.batch_size
    if args.learning_rate:
        overrides['adam_optimizer.lr'] = args.learning_rate

    # Load configuration
    config = load_config(args.config, **overrides)

    # Handle direct data paths (overrides data_root approach)
    if args.train_data_path:
        # Extract directory and folder name
        train_dir = os.path.dirname(args.train_data_path)
        train_name = os.path.basename(args.train_data_path)
        config.system.data_root = train_dir
        config.train.dataset_zip_file_name_training = train_name

    if args.val_data_path:
        val_dir = os.path.dirname(args.val_data_path)
        val_name = os.path.basename(args.val_data_path)
        if not args.train_data_path:  # Only update data_root if not already set
            config.system.data_root = val_dir
        config.val.dataset_zip_file_name_validation = val_name

    # Print configuration
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Configuration file: {args.config}")
    print(f"Data root: {config.system.data_root}")
    print(f"Training data: {config.train.dataset_zip_file_name_training}")
    print(f"Validation data: {config.val.dataset_zip_file_name_validation}")
    print(f"Log directory: {config.system.log_dir_system}")
    print(f"Training epochs: {config.train.epochs}")
    print(f"Batch size: {config.train.batch_training_size}")
    print(f"Learning rate: {config.adam_optimizer.lr}")
    print("=" * 60)

    # Validate data paths
    data_valid = validate_data_paths(config)

    if args.check_data_only:
        print("Data path check complete.")
        sys.exit(0 if data_valid else 1)

    if not data_valid and not args.force:
        print("\nERROR: Data validation failed. Use --force to proceed anyway or fix the paths.")
        sys.exit(1)

    # Import here to avoid circular imports
    try:
        from pose_estimation.pose_estimation_estimator import PoseEstimationEstimator

        # Create estimator
        print(f"\n{'=' * 80}")
        print("STARTING MODEL TRAINING")
        print(f"{'=' * 80}")
        estimator = PoseEstimationEstimator(config=config)
        print(f"Device: {estimator.device}")
        print(f"Model will be saved to: {os.path.abspath(config.system.log_dir_system)}")
        print(f"TensorBoard logs: {os.path.abspath(config.system.log_dir_system)}")
        print(f"{'=' * 80}")

        # Start training
        print("Starting training...")
        estimator.train()

        # Cleanup
        estimator.writer.done()
        print(f"\n{'=' * 80}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Models saved in: {os.path.abspath(config.system.log_dir_system)}")
        print(
            f"training progress with: tensorboard --logdir \"{os.path.abspath(config.system.log_dir_system)}\"")
        print(f"{'=' * 80}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
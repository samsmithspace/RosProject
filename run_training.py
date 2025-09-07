#!/usr/bin/env python3
"""
Standalone training script that avoids circular imports.
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


def load_config(config_file="config.yaml", data_root=None, log_dir=None):
    """Load configuration from YAML file with optional overrides"""
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Override paths if provided
    if data_root:
        config['system']['data_root'] = data_root
    if log_dir:
        config['system']['log_dir_system'] = log_dir

    # Convert to EasyDict for easier access
    config = EasyDict(config)
    return config


def main():
    """Main function to run training directly"""
    parser = argparse.ArgumentParser(description='Train Pose Estimation Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file (default: config.yaml)')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override data root path')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Override log directory path')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override training batch size')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config, args.data_root, args.log_dir)

    # Override config with command line arguments if provided
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.train.batch_training_size = args.batch_size

    print(f"Configuration loaded from: {args.config}")
    print(f"Data root: {config.system.data_root}")
    print(f"Log directory: {config.system.log_dir_system}")
    print(f"Training epochs: {config.train.epochs}")
    print(f"Batch size: {config.train.batch_training_size}")

    # Import here to avoid circular imports
    try:
        from pose_estimation.pose_estimation_estimator import PoseEstimationEstimator

        # Create estimator
        estimator = PoseEstimationEstimator(config=config)
        print(f"Device: {estimator.device}")

        # Start training
        print("Starting training...")
        estimator.train()

        # Cleanup
        estimator.writer.done()
        print("Training completed successfully!")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
import os
import sys
import yaml


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root_dir not in sys.path:
        sys.path.append(root_dir)

    config_path = os.path.join(root_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.yaml not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    from code.gma_core.train_gma import train_gma_model

    train_gma_model(config)


if __name__ == "__main__":
    main()

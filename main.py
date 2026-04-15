"""Root CLI entry point for training and sample generation."""

from __future__ import annotations

import argparse

from medical_diffusion_app.configs.config import DATASET_DIR


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for training and generation."""
    parser = argparse.ArgumentParser(description="Medical diffusion app command-line runner")
    parser.add_argument("--mode", required=True, choices=["train", "generate"], help="Workflow mode")
    parser.add_argument("--dataset_path", default=str(DATASET_DIR), help="Dataset root path for training")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--label", type=int, default=1, help="Disease label id for generation")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of images to generate")
    return parser.parse_args()


def main() -> int:
    """Runs the requested workflow."""
    args = parse_args()

    try:
        from medical_diffusion_app.utils.helper import ensure_output_dirs

        ensure_output_dirs()
        if args.mode == "train":
            from medical_diffusion_app.ml.training.train_ddpm import train

            result = train(dataset_path=args.dataset_path, epochs=args.epochs)
        else:
            from medical_diffusion_app.ml.generation.generate_samples import generate_images

            result = generate_images(label=args.label, num_samples=args.num_samples)
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "required dependency"
        print(
            "Missing dependency detected: "
            f"{missing_module}. Install app dependencies with `pip install -r medical_diffusion_app/requirements.txt` "
            "and ML dependencies with `pip install -r medical_diffusion_app/requirements-ml.txt`."
        )
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

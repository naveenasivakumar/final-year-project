import argparse
from backend.services.preprocessing_service import preprocess_dataset
from backend.services.diffusion_service import train_diffusion, generate_synthetic
from backend.services.classification_service import train_classification, evaluate_classification


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Rare Disease X-Ray AI pipeline")
    parser.add_argument("--mode", choices=["full_pipeline", "preprocess", "train_diffusion", "generate", "train_classification", "evaluate"], required=True)
    parser.add_argument("--dataset_path", default="data/raw")
    parser.add_argument("--processed_path", default="data/processed")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_training_samples", type=int, default=64)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "preprocess":
        result = preprocess_dataset(
            dataset_path=args.dataset_path,
            output_dir=args.processed_path,
            target_size=(224, 224),
            test_ratio=0.2,
        )
        print("Preprocessing result:", result)

    elif args.mode == "train_diffusion":
        result = train_diffusion(
            dataset_path=args.processed_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_training_samples=args.max_training_samples,
        )
        print("Diffusion training result:", result)

    elif args.mode == "generate":
        result = generate_synthetic(
            label=args.label,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
        )
        print("Generation result:", result)

    elif args.mode == "train_classification":
        result = train_classification(
            dataset_path=args.processed_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_training_samples=args.max_training_samples,
        )
        print("Classification training result:", result)

    elif args.mode == "evaluate":
        result = evaluate_classification(
            dataset_path=args.processed_path,
            batch_size=args.batch_size,
        )
        print("Evaluation result:", result)

    elif args.mode == "full_pipeline":
        print("Starting full pipeline...")
        preprocess_dataset(
            dataset_path=args.dataset_path,
            output_dir=args.processed_path,
            target_size=(224, 224),
            test_ratio=0.2,
        )
        train_diffusion(
            dataset_path=args.processed_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_training_samples=args.max_training_samples,
        )
        generate_synthetic(
            label=args.label,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
        )
        train_classification(
            dataset_path=args.processed_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_training_samples=args.max_training_samples,
        )
        metrics = evaluate_classification(
            dataset_path=args.processed_path,
            batch_size=args.batch_size,
        )
        print("Full pipeline metrics:", metrics)


if __name__ == "__main__":
    main()

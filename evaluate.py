import argparse
import os

import numpy as np
import torch

from Data import get_corruption_module, get_dataset_module
from Net import create_cnn_model
from train_utils import evaluate, load_checkpoint, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on clean and corrupted image datasets.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "tiny_imagenet"], default="cifar10")
    parser.add_argument("--data-root", "--dataset-root", dest="data_root", default=None)
    parser.add_argument("--model", choices=["resnet50", "resnet50_ti", "resnet110", "wide_resnet", "densenet121"], default="resnet50")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--wide-depth", type=int, default=26)
    parser.add_argument("--wide-width", type=int, default=10)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", "-b", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--output-name", default=None)

    parser.add_argument("--corruption-dataset", choices=["cifar10_c", "cifar100_c", "tiny_imagenet_c"], default=None)
    parser.add_argument("--corruption-root", default=None)
    parser.add_argument("--corruptions", nargs="+", default=["all"])
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    return parser.parse_args()


def resolve_data_root(args):
    if args.data_root:
        return args.data_root
    if args.dataset == "tiny_imagenet":
        return os.environ.get("FGR_TINY_ROOT", "./data/tiny-imagenet-200")
    return os.environ.get("FGR_CIFAR_ROOT", "./data")


def selected_corruptions(module, requested):
    if requested in {None, "all"} or requested == ["all"]:
        return module.CORRUPTIONS
    if isinstance(requested, str):
        return [requested]
    return list(requested)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_module = get_dataset_module(args.dataset)
    model_kwargs = {"depth": args.wide_depth, "width": args.wide_width}
    model = create_cnn_model(args.model, dataset_module.NUM_CLASSES, **model_kwargs).to(device)
    load_checkpoint(model, args.checkpoint)

    clean_loader = dataset_module.get_test_loader(
        root=resolve_data_root(args),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        download=args.download,
        input_size=args.input_size,
    )
    clean_metrics = evaluate(model, clean_loader, device, max_batches=args.max_batches)
    payload = {"args": vars(args), "clean": clean_metrics}
    print(
        f"clean acc={clean_metrics['acc']:.4f} ece={clean_metrics['ece']:.4f} "
        f"nll={clean_metrics['nll']:.4f}"
    )

    if args.corruption_dataset:
        if not args.corruption_root:
            raise ValueError("--corruption-root is required for corrupted evaluation")
        corruption_module = get_corruption_module(args.corruption_dataset)
        results = []
        for corruption in selected_corruptions(corruption_module, args.corruptions):
            for severity in args.severities:
                loader = corruption_module.get_test_loader(
                    root=args.corruption_root,
                    batch_size=args.batch_size,
                    corruption=corruption,
                    severity=severity,
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                    input_size=args.input_size,
                )
                metrics = evaluate(model, loader, device, max_batches=args.max_batches)
                row = {"corruption": corruption, "severity": severity, **metrics}
                results.append(row)
                print(
                    f"{corruption} severity={severity} acc={metrics['acc']:.4f} "
                    f"ece={metrics['ece']:.4f} nll={metrics['nll']:.4f}"
                )
        payload["corrupted"] = results
        if results:
            payload["corrupted_mean"] = {
                "acc": float(np.mean([row["acc"] for row in results])),
                "ece": float(np.mean([row["ece"] for row in results])),
                "nll": float(np.mean([row["nll"] for row in results])),
            }
            print(
                f"corrupted mean acc={payload['corrupted_mean']['acc']:.4f} "
                f"ece={payload['corrupted_mean']['ece']:.4f} nll={payload['corrupted_mean']['nll']:.4f}"
            )

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_name or f"eval_{args.dataset}.json"
    save_json(os.path.join(args.output_dir, output_name), payload)


if __name__ == "__main__":
    main()

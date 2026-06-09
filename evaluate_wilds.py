import argparse
import os

import torch

from Data.wilds_loader import WILDS_CONFIGS, get_wilds_loaders
from Net import create_wilds_model
from train_utils import evaluate, load_checkpoint, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a WILDS checkpoint.")
    parser.add_argument("--dataset", choices=["iwildcam", "camelyon17", "fmow"], default="iwildcam")
    parser.add_argument("--wilds-root", "--dataset-root", dest="wilds_root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def resolve_wilds_root(args):
    if args.wilds_root:
        return args.wilds_root
    return os.environ.get("FGR_WILDS_ROOT", "./data/wilds")


def main():
    args = parse_args()
    if args.dataset not in WILDS_CONFIGS:
        raise ValueError(f"Unsupported WILDS dataset: {args.dataset}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_wilds_model(
        args.dataset,
        num_classes=WILDS_CONFIGS[args.dataset]["num_classes"],
        pretrained=False,
    ).to(device)
    load_checkpoint(model, args.checkpoint)

    workers = WILDS_CONFIGS[args.dataset]["num_workers"] if args.num_workers is None else args.num_workers
    _, val_loader, test_loader = get_wilds_loaders(
        args.dataset,
        root_dir=resolve_wilds_root(args),
        batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        download=args.download,
        num_workers=workers,
    )
    split_loaders = {"val": val_loader, "test": test_loader}
    payload = {"args": vars(args), "metrics": {}}
    for split in args.splits:
        if split not in split_loaders:
            raise ValueError(f"Unsupported split: {split}")
        metrics = evaluate(model, split_loaders[split], device, max_batches=args.max_batches)
        payload["metrics"][split] = metrics
        print(f"{split} acc={metrics['acc']:.4f} ece={metrics['ece']:.4f} nll={metrics['nll']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_name or f"eval_wilds_{args.dataset}.json"
    save_json(os.path.join(args.output_dir, output_name), payload)


if __name__ == "__main__":
    main()

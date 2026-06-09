import argparse
import os

import torch
from torch.utils.data import DataLoader

from Data.corrupted_dataset import CorruptedDataset
from Data.wilds_loader import WILDS_CONFIGS, get_wilds_loaders
from Net import create_wilds_model, freeze_backbone, get_wilds_optimizer_config
from train_utils import (
    ExperimentLogger,
    build_optimizer,
    build_scheduler,
    ensure_dir,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    save_json,
    set_seed,
    train_one_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FGR or baseline losses on WILDS datasets.")
    parser.add_argument("--dataset", choices=["iwildcam", "camelyon17", "fmow"], default="iwildcam")
    parser.add_argument("--wilds-root", "--dataset-root", dest="wilds_root", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--loss", default="dual_focal_loss")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--lamda", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", "-b", type=int, default=None)
    parser.add_argument("--test-batch-size", "-tb", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--scheduler", choices=["multistep", "step", "none"], default="none")
    parser.add_argument("--milestones", type=int, nargs="+", default=[])
    parser.add_argument("--lr-decay", type=float, default=0.1)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--step-gamma", type=float, default=0.96)

    parser.add_argument("--fgr", action="store_true", help="Enable frequency-aware gradient rectification.")
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--filter-type", choices=["jpeg_compression", "fft_lowpass"], default="jpeg_compression")
    parser.add_argument("--filter-severities", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--calibration-loss", default="soft_ece")
    parser.add_argument("--calibration-weight", type=float, default=1.0)
    parser.add_argument("--correct-all", action="store_true")

    parser.add_argument("--load", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--swanlab-mode", default="disabled", choices=["disabled", "local", "cloud"])
    parser.add_argument("--swanlab-project", default="fgr-wilds")
    return parser.parse_args()


def fill_defaults(args):
    defaults = get_wilds_optimizer_config(args.dataset)
    args.epochs = defaults["epochs"] if args.epochs is None else args.epochs
    args.batch_size = defaults["batch_size"] if args.batch_size is None else args.batch_size
    args.test_batch_size = defaults["test_batch_size"] if args.test_batch_size is None else args.test_batch_size
    args.lr = defaults["lr"] if args.lr is None else args.lr
    args.weight_decay = defaults["weight_decay"] if args.weight_decay is None else args.weight_decay
    args.optimizer = defaults["optimizer"] if args.optimizer is None else args.optimizer
    args.num_workers = WILDS_CONFIGS[args.dataset]["num_workers"] if args.num_workers is None else args.num_workers
    if args.scheduler == "none" and "step_gamma" in defaults:
        args.scheduler = "step"
        args.step_gamma = defaults["step_gamma"]
    return args


def resolve_wilds_root(args):
    if args.wilds_root:
        return args.wilds_root
    return os.environ.get("FGR_WILDS_ROOT", "./data/wilds")


def run_name_from_args(args):
    if args.run_name:
        return args.run_name
    model_name = args.model or ("resnet50" if args.dataset == "iwildcam" else "densenet121")
    suffix = "fgr" if args.fgr else args.loss
    return f"{args.dataset}_{model_name}_{suffix}_seed{args.seed}"


def main():
    args = fill_defaults(parse_args())
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset not in WILDS_CONFIGS:
        raise ValueError(f"Unsupported WILDS dataset: {args.dataset}")

    model = create_wilds_model(
        args.dataset,
        num_classes=WILDS_CONFIGS[args.dataset]["num_classes"],
        pretrained=args.pretrained and not bool(args.checkpoint),
    ).to(device)
    if args.load or args.checkpoint:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required with --load")
        load_checkpoint(model, args.checkpoint)
    if args.freeze:
        freeze_backbone(model)

    train_loader, val_loader, test_loader = get_wilds_loaders(
        args.dataset,
        root_dir=resolve_wilds_root(args),
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        download=args.download,
        num_workers=args.num_workers,
    )
    optimizer = build_optimizer(
        model,
        args.optimizer,
        args.lr,
        args.weight_decay,
        momentum=args.momentum,
        nesterov=args.nesterov,
    )
    scheduler = build_scheduler(
        optimizer,
        {
            "scheduler": args.scheduler,
            "milestones": args.milestones,
            "lr_decay": args.lr_decay,
            "step_size": args.step_size,
            "step_gamma": args.step_gamma,
        },
    )

    ensure_dir(args.save_dir)
    ensure_dir(args.output_dir)

    corrupted_set = None
    corrupted_loader = None
    if args.fgr:
        corrupted_set = CorruptedDataset(
            train_loader.dataset,
            corruption_prob=args.rho,
            corruption_types=[args.filter_type],
            severities=args.filter_severities,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        workers = max(int(args.num_workers), 0)
        corrupted_loader = DataLoader(
            corrupted_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )

    run_name = run_name_from_args(args)
    logger = ExperimentLogger(
        args.swanlab_mode,
        project=args.swanlab_project,
        run_name=run_name,
        config=vars(args),
    )
    history = []
    best_val_ece = float("inf")
    best_path = os.path.join(args.save_dir, f"{run_name}_best.pt")
    last_path = os.path.join(args.save_dir, f"{run_name}_last.pt")

    for epoch in range(args.epochs):
        use_fgr = bool(args.fgr and epoch >= args.start_epoch)
        active_loader = corrupted_loader if use_fgr and corrupted_loader is not None else train_loader
        if use_fgr and corrupted_set is not None:
            corrupted_set.reset_corruption_selection()
        train_stats = train_one_epoch(
            model,
            active_loader,
            optimizer,
            device,
            loss_name=args.loss,
            gamma=args.gamma,
            lamda=args.lamda,
            use_fgr=use_fgr,
            calibration_loss=args.calibration_loss,
            calibration_weight=args.calibration_weight,
            correct_all=args.correct_all,
            max_steps=args.max_train_steps,
        )
        if scheduler is not None:
            scheduler.step()

        val_metrics = evaluate(model, val_loader, device, max_batches=args.max_eval_batches)
        test_metrics = evaluate(model, test_loader, device, max_batches=args.max_eval_batches)
        row = {
            "epoch": epoch + 1,
            "use_fgr": use_fgr,
            "train": train_stats,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(row)
        logger.log(
            {
                "train_loss": train_stats["loss"],
                "val_acc": val_metrics["acc"],
                "val_ece": val_metrics["ece"],
                "test_acc": test_metrics["acc"],
                "test_ece": test_metrics["ece"],
            },
            step=epoch + 1,
        )
        print(
            f"epoch={epoch + 1:03d} use_fgr={use_fgr} train_loss={train_stats['loss']:.6f} "
            f"val_acc={val_metrics['acc']:.4f} val_ece={val_metrics['ece']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} test_ece={test_metrics['ece']:.4f}"
        )
        if val_metrics["ece"] < best_val_ece:
            best_val_ece = val_metrics["ece"]
            save_checkpoint(best_path, model, optimizer, scheduler, epoch + 1, vars(args), val_metrics)

    save_checkpoint(last_path, model, optimizer, scheduler, args.epochs, vars(args), history[-1] if history else None)
    save_json(
        os.path.join(args.output_dir, f"{run_name}.json"),
        {"args": vars(args), "best_checkpoint": best_path, "last_checkpoint": last_path, "history": history},
    )
    logger.finish()


if __name__ == "__main__":
    main()

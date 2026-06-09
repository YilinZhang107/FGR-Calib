import argparse
import os

import torch
from torch.utils.data import DataLoader

from Data import CorruptedDataset, get_dataset_module
from Net import create_cnn_model, freeze_backbone
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
    parser = argparse.ArgumentParser(description="Train FGR or baseline losses on CIFAR/Tiny-ImageNet.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "tiny_imagenet"], default="cifar10")
    parser.add_argument("--data-root", "--dataset-root", dest="data_root", default=None)
    parser.add_argument("--model", choices=["resnet50", "resnet50_ti", "resnet110", "wide_resnet", "densenet121"], default="resnet50")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--wide-depth", type=int, default=26)
    parser.add_argument("--wide-width", type=int, default=10)

    parser.add_argument("--loss", default="dual_focal_loss")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--lamda", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch-size", "-b", type=int, default=128)
    parser.add_argument("--test-batch-size", "-tb", type=int, default=256)
    parser.add_argument("--valid-size", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--scheduler", choices=["multistep", "step", "none"], default="multistep")
    parser.add_argument("--milestones", type=int, nargs="+", default=[150, 250])
    parser.add_argument("--lr-decay", type=float, default=0.1)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--step-gamma", type=float, default=0.96)

    parser.add_argument("--fgr", action="store_true", help="Enable frequency-aware gradient rectification.")
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--start-epoch", type=int, default=200)
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
    parser.add_argument("--swanlab-project", default="fgr")
    return parser.parse_args()


def resolve_data_root(args):
    if args.data_root:
        return args.data_root
    if args.dataset == "tiny_imagenet":
        return os.environ.get("FGR_TINY_ROOT", "./data/tiny-imagenet-200")
    return os.environ.get("FGR_CIFAR_ROOT", "./data")


def build_loaders(args, device):
    module = get_dataset_module(args.dataset)
    pin_memory = device.type == "cuda"
    data_root = resolve_data_root(args)
    train_set, train_loader, val_loader = module.get_train_valid_loader(
        root=data_root,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        valid_size=args.valid_size,
        augment=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
        download=args.download,
        input_size=args.input_size,
    )
    test_loader = module.get_test_loader(
        root=data_root,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        download=args.download,
        input_size=args.input_size,
    )
    return module, train_set, train_loader, val_loader, test_loader


def run_name_from_args(args):
    if args.run_name:
        return args.run_name
    suffix = "fgr" if args.fgr else args.loss
    return f"{args.dataset}_{args.model}_{suffix}_seed{args.seed}"


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_module, train_set, train_loader, val_loader, test_loader = build_loaders(args, device)

    model_kwargs = {"depth": args.wide_depth, "width": args.wide_width}
    model = create_cnn_model(args.model, dataset_module.NUM_CLASSES, **model_kwargs).to(device)
    if args.load or args.checkpoint:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required with --load")
        load_checkpoint(model, args.checkpoint)
    if args.freeze:
        freeze_backbone(model)

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
            train_set,
            corruption_prob=args.rho,
            corruption_types=[args.filter_type],
            severities=args.filter_severities,
            mean=dataset_module.MEAN,
            std=dataset_module.STD,
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

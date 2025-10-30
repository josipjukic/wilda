from dataloader import *

import argparse

import transformers


transformers.logging.set_verbosity_error()


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=["rte"],
        nargs="+",
        help="Data corpus.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3-8b",
        help="Model",
    )

    parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
    # parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=5, help="upper epoch limit")
    parser.add_argument(
        "--batch-size", type=int, default=1, metavar="N", help="batch size"
    )
    parser.add_argument(
        "--l2", type=float, default=0, help="l2 regularization (weight decay)"
    )

    # Repeat experiments
    parser.add_argument(
        "--repeat", type=int, default=10, help="number of times to repeat training"
    )

    # Gpu based arguments
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Gpu to use for experiments (-1 means no GPU)",
    )
    parser.add_argument(
        "--aux-gpu",
        type=int,
        default=-1,
        help="Auxiliary GPU",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=128,
        help="Demonstration pool -- training data",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=200,
        help="Evaluation set size",
    )
    parser.add_argument(
        "--num-demos",
        type=int,
        default=16,
        help="Number of demonstrations",
    )
    # Number of adapters
    parser.add_argument(
        "--num-adapters",
        type=int,
        default=1,
    )
    # Storing & loading arguments
    parser.add_argument(
        "--save",
        type=str,
        default="chkp/",
        help="Folder to store final model",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="Folder to store final model (or model with best valid perf) in",
    )
    parser.add_argument(
        "--stratified",
        type=bool,
        default=False,
        help="Stratified warm start sample.",
    )

    parser.add_argument(
        "--scheduler",
        type=bool,
        default=False,
        help="Bool: use linear decay scheduler.",
    )

    parser.add_argument(
        "--peft",
        type=str,
        default="lora",
        help="PEFT: [lora, ptuning, prefix_tuning].",
    )

    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=[
            "fixed",
            "epoch-shuffle",
            "epoch-resample",
        ],
        help="Demonstration sampling mode [fixed, epoch-shuffle, instance-shuffle, epoch-resample, instance-resample].",
    )

    return parser.parse_args()

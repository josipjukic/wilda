from experiment import Experiment
from dataloader import *

import pickle
import logging
from datetime import datetime
from transformers import logging as trans_log

from util import set_seed_everywhere

import torch
from transformers import AutoTokenizer

from util import Config
from models import *

from args import *

from metrics import EvalSet, ResultSet
import numpy as np


if __name__ == "__main__":
    trans_log.set_verbosity_error()
    args = make_parser()
    seeds = list(range(1, args.repeat + 1))

    set_seed_everywhere(0)

    for dataset in args.data:
        dh = DATASETS[dataset]
        train, test, val = dh.get_dataset_splits()

        train_size = len(train)
        test_size = len(test)

        max_num_train = args.num_train + args.num_demos

        # Subsample train and test sets
        if max_num_train < train_size:
            train = train.select(
                np.random.choice(a=train_size, size=max_num_train, replace=False)
            )
        if args.num_test < test_size:
            test = test.select(
                np.random.choice(a=test_size, size=args.num_test, replace=False)
            )

        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMERS[args.model])
        meta = Config()
        meta.task_type = "lm"

        # Initialize logging
        fmt = "%Y-%m-%d-%H-%M"
        start_time = fname = datetime.now().strftime(fmt)
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(f"log/{dataset}-{args.model}-{start_time}.log"),
                logging.StreamHandler(),
            ],
        )

        meta_info = {
            "dataset": dataset,
            "model": args.model,
            "num_demos": args.num_demos,
            "num_adapters": args.num_adapters,
            "num_train": args.num_train,
            "num_test": args.num_test,
            "peft": args.peft,
            "epochs": args.epochs,
        }

        logging.info(meta_info)

        res_cc = {}
        # res_cc_nshot = {}
        for mode in args.modes:
            res_cc[mode] = EvalSet(meta=meta_info, name="copycat", demo_type=mode)
            # res_cc_nshot[mode] = EvalSet(
            #     meta=meta_info, name="copycat-n-shot", demo_type=mode
            # )
        eval_set_pbft = EvalSet(meta=meta_info, name="pbft", demo_type="none")
        results_zero = ResultSet(meta=meta_info, name="0-shot", demo_type="none")
        results_nshot = ResultSet(meta=meta_info, name="n-shot", demo_type="none")

        cuda = torch.cuda.is_available() and args.gpu != -1
        device = torch.device("cpu") if not cuda else torch.device("cuda")

        for i, seed in zip(range(1, args.repeat + 1), seeds):
            logging.info(f"Running experiment {i}/{args.repeat}")
            logging.info(f"=" * 100)

            set_seed_everywhere(seed)
            logging.info(f"Seed = {seed}")
            logging.info(f"Maximum train size: {len(train)}")

            # Setup the loss function
            criterion = nn.CrossEntropyLoss()
            experiment = Experiment(train, val, test, dh, device, args, meta)
            demo_indices = experiment.retrieve_sample()

            model = initialize_model(args, meta, device)
            model.to(device)

            zero, nshot = experiment.icl_eval(
                model=model, tokenizer=tokenizer, demonstration_indices=demo_indices
            )

            results_zero.add(zero, seed)
            results_nshot.add(nshot, seed)

            eval = experiment.pbft(
                model=model,
                tokenizer=tokenizer,
                criterion=criterion,
                demo_indices=demo_indices,
            )
            eval_set_pbft.add(eval, seed)

            for mode in args.modes:
                model = initialize_model(args, meta, device)
                model.to(device)
                evals = experiment.copycat(
                    model=model,
                    tokenizer=tokenizer,
                    criterion=criterion,
                    demonstration_indices=demo_indices,
                    mode=mode,
                )

                res_cc[mode].add(evals, seed)
                # res_cc_nshot[mode].add(evals_nshot, seed)

        fname = f"{dataset}-{args.model}-{args.peft}-r=-{start_time}.pkl"

        run_time = datetime.now() - datetime.strptime(start_time, fmt)
        meta_info["run_time"] = run_time.total_seconds()
        logging.info(run_time)

        with open(f"{args.save_dir}/{fname}", "wb") as f:
            pickle.dump(
                (
                    meta_info,
                    list(res_cc.values())
                    + [
                        eval_set_pbft,
                        results_zero,
                        results_nshot,
                    ],
                ),
                f,
            )

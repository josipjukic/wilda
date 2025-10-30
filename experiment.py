import numpy as np
import logging
from sklearn.metrics import f1_score
import torch
from time import time

from adapters import peft_factory

from metrics import hypersphere_distance


from dataloader import *
import pickle

from merging import *
from torch.utils.data import DataLoader
from iterator import (
    collate_queries,
    collate_demonstrations,
)

from functools import partial
from collections import Counter

import math

# import wandb

from metrics import Evaluation, Result

from random import shuffle


class Experiment:
    def __init__(self, train_set, val_set, test_set, data_handler, device, args, meta):
        self.args = args
        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set
        self.data_handler = data_handler
        self.device = device
        self.meta = meta

    # Pattern-based fine-tuning
    def pbft(
        self,
        tokenizer,
        criterion,
        demo_indices,
        model,
    ):

        peft_config = peft_factory(self.args.peft)
        adapter_name = "pbft"
        model.lm.add_adapter(peft_config, adapter_name=adapter_name)
        model.lm.set_adapter(adapter_name)

        # Retrieve a sample.
        train_indices = demo_indices

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.l2
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1,
        )

        # 2) Fine-tuning epochs
        eval = Evaluation(adapter_name)
        for epoch in range(self.args.epochs):
            logging.info(f"Training epoch: {epoch+1}/{self.args.epochs}")

            loss = self._train_model(
                model, tokenizer, optimizer, criterion, train_indices
            )
            logging.info(f"Loss {loss}")
            # eval.add_train_result(result)
            if self.args.scheduler:
                scheduler.step()
            logging.info("Epoch finished\n")

            result_pbft = self._evaluate_model(model, tokenizer, name="pbft")
            eval.add_test_result(result_pbft)

            # c) Evaluate model (test set)
            # result = self._evaluate_model(model, tokenizer)
            # eval.add_test_result(result)
            # logging.info("**********TEST**********")
            # logging.info(result)
            # logging.info("************************")

        # torch.save(model.state_dict(), "model.pth")

        return [eval]

    def icl_eval(
        self,
        model,
        tokenizer,
        demonstration_indices,
    ):

        demonstration_instances = self.train_set.select(demonstration_indices)
        demonstrations = [
            self.data_handler.make_demonstration(instance)
            for instance in demonstration_instances
        ]
        n_demos = len(demonstrations)
        demo_text = "".join(demonstrations)

        with torch.inference_mode():
            demo_ids = tokenizer(demo_text, return_tensors="pt")["input_ids"].to(
                self.device
            )
            outputs = model(demo_ids, use_cache=True)
            demo_key_values = outputs.past_key_values

        logging.info(f"{n_demos}-shot ICL...")
        result_n_shot = self._evaluate_model(
            tokenizer=tokenizer,
            model=model,
            name="n-shot",
            demo_key_values=demo_key_values,
        )

        logging.info("0-shot...")
        result_zero_shot = self._evaluate_model(model, tokenizer, "0-shot")

        return result_n_shot, result_zero_shot

    def copycat(
        self,
        tokenizer,
        criterion,
        demonstration_indices,
        model,
        mode,
    ):

        # Exclude the demonstrations from the copycat train set
        train_indices = set(range(len(self.train_set))) - set(demonstration_indices)

        demonstration_instances = self.train_set.select(demonstration_indices)
        demonstrations = [
            self.data_handler.make_demonstration(instance)
            for instance in demonstration_instances
        ]
        n_demos = len(demonstrations)
        demo_text = "".join(demonstrations)

        with torch.inference_mode():
            demo_ids = tokenizer(demo_text, return_tensors="pt")["input_ids"].to(
                self.device
            )
            outputs = model(demo_ids, use_cache=True)
            demo_key_values = outputs.past_key_values

        evals = []
        # evals_nshot = []

        # logging.info(f"{n_demos}-shot ICL...")
        # result_n_shot = self._evaluate_model(
        #     tokenizer=tokenizer,
        #     model=model,
        #     name="n-shot",
        #     demo_key_values=demo_key_values,
        # )

        # logging.info("0-shot...")
        # result_zero_shot = self._evaluate_model(model, tokenizer, "0-shot")

        peft_config = peft_factory(self.args.peft)

        start_idx = 0
        chunk_size = n_demos // self.args.num_adapters
        adapters = []

        logging.info(
            f"CopyCat fine-tuning {self.args.num_adapters} adapters with {chunk_size} demonstrations..."
        )

        for i in range(0, self.args.num_adapters):
            adapter_name = f"{self.args.peft}-{i+1}"
            eval = Evaluation(adapter_name)
            # eval_nshot = Evaluation(f"{adapter_name} n-shot")
            adapters.append(adapter_name)
            logging.info("=" * 50)
            logging.info(f"Copycat: {adapter_name}")
            model.lm.add_adapter(peft_config, adapter_name=adapter_name)

            end_idx = start_idx + chunk_size
            demos_i = demonstrations[start_idx:end_idx]

            logging.info(f"start={start_idx}, end={end_idx}; |D|={len(demos_i)}")
            start_idx += chunk_size

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.args.lr, weight_decay=self.args.l2
            )

            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1,
            )

            with torch.inference_mode():
                demo_ids = tokenizer("".join(demos_i), return_tensors="pt")[
                    "input_ids"
                ].to(self.device)
                outputs = model(demo_ids, use_cache=True)
                demo_key_values = outputs.past_key_values

            for epoch in range(self.args.epochs):
                logging.info(f"Training epoch: {epoch+1}/{self.args.epochs}")

                if mode == "epoch-shuffle":
                    with torch.inference_mode():
                        shuffle(demos_i)
                        demo_ids = tokenizer("".join(demos_i), return_tensors="pt")[
                            "input_ids"
                        ].to(self.device)
                        outputs = model(demo_ids, use_cache=True)
                        demo_key_values = outputs.past_key_values
                elif mode == "epoch-resample":
                    demo_indices = self.resample_demonstrations(
                        n=self.args.num_demos // self.args.num_adapters
                    )
                    demo_instances = self.train_set.select(demo_indices)
                    demos_i = [
                        self.data_handler.make_demonstration(instance)
                        for instance in demo_instances
                    ]
                    with torch.inference_mode():
                        demo_ids = tokenizer("".join(demos_i), return_tensors="pt")[
                            "input_ids"
                        ].to(self.device)
                        outputs = model(demo_ids, use_cache=True)
                        demo_key_values = outputs.past_key_values

                result_train, loss = self._copycat_train(
                    model,
                    tokenizer,
                    optimizer,
                    criterion,
                    train_indices,
                    demo_key_values,
                    demos_i,
                    adapter=adapter_name,
                    mode=mode,
                )
                eval.add_special_result(result_train)
                logging.info(f"Loss = {loss}")
                if self.args.scheduler:
                    scheduler.step()

                model.lm.set_adapter(adapter_name)
                result_copycat = self._evaluate_model(
                    model, tokenizer, name=adapter_name
                )
                eval.add_test_result(result_copycat)
                # result_copycat_n_shot = self._evaluate_model(
                #     model,
                #     tokenizer,
                #     name=adapter_name,
                #     demo_key_values=demo_key_values,
                # )
                # eval_nshot.add_test_result(result_copycat_n_shot)

                logging.info("Epoch finished\n")
            evals.append(eval)
            # evals_nshot.append(eval_nshot)

        if self.args.num_adapters > 1:
            logging.info("Merging")

            merged_adapter = "fusion"
            eval = Evaluation(merged_adapter)
            adapter_params = get_multiple_adapters_params(model, adapters)
            merged_params = merge_adapters(
                adapter_params, adapters, [1.0] * len(adapters), merged_adapter
            )

            peft_config = peft_factory(self.args.peft)

            model.lm.add_adapter(peft_config, adapter_name=merged_adapter)
            overwrite_adapter_params(model, merged_params)
            model.lm.set_adapter(merged_adapter)

            result_copycat = self._evaluate_model(model, tokenizer, name=merged_adapter)
            eval.add_test_result(result_copycat)
            evals.append(eval)
            # TODO: add n shot eval

        return evals

    def _copycat_train(
        self,
        model,
        tokenizer,
        optimizer,
        criterion,
        train_indices,
        demo_key_values,
        demos,
        adapter,
        mode,
    ):
        model.train()

        loss_agg = 0.0
        train = self.train_set.select(list(train_indices)[: self.args.num_train])

        dl = DataLoader(
            train,
            batch_size=self.args.batch_size,
            collate_fn=partial(
                collate_queries,
                dh=self.data_handler,
                tokenizer=tokenizer,
                device=self.device,
            ),
        )

        past_key_values = demo_key_values
        for batch in tqdm(dl):
            optimizer.zero_grad()

            if mode == "instance-shuffle":
                with torch.inference_mode():
                    shuffle(demos)
                    demo_ids = tokenizer("".join(demos), return_tensors="pt")[
                        "input_ids"
                    ].to(self.device)
                    outputs = model(demo_ids, use_cache=True)
                    past_key_values = outputs.past_key_values
            elif mode == "instance-resample":
                demo_indices = self.resample_demonstrations(
                    n=self.args.num_demos // self.args.num_adapters
                )
                demo_instances = self.train_set.select(demo_indices)
                demos = [
                    self.data_handler.make_demonstration(instance)
                    for instance in demo_instances
                ]
                with torch.inference_mode():
                    demo_ids = tokenizer("".join(demos), return_tensors="pt")[
                        "input_ids"
                    ].to(self.device)
                    outputs = model(demo_ids, use_cache=True)
                    past_key_values = outputs.past_key_values

            query_ids = batch.input_ids

            # Base model, inference mode: disable adapters
            model.eval()
            model.lm.disable_adapters()

            outputs_base = model(query_ids, past_key_values=past_key_values)
            logits_base = outputs_base.logits[:, -1, :]
            target = logits_base.argmax(dim=-1)

            # Back to train mode, enable adapters
            model.train()
            model.lm.enable_adapters()
            model.lm.set_adapter(adapter)

            outputs = model(query_ids)
            logits = outputs.logits[:, -1, :]

            loss = criterion(logits, target)

            loss_agg += float(loss)

            loss.backward()
            optimizer.step()

        avg_loss = loss_agg / len(train)

        model.eval()

        verbalizers = self.data_handler.class_labels
        verbalizer_ids = [tokenizer.encode(verb)[-1] for verb in verbalizers]

        result = None
        if mode == "fixed":
            teacher_logits = []
            student_logits = []
            labels = []
            torch.cuda.empty_cache()
            with torch.no_grad():
                for batch in tqdm(dl):
                    label = batch.target[0].cpu()
                    labels.append(label)

                    # TODO: streamline, move base predicitions to the training loop, as this
                    # only needs to be computed once
                    query_ids = batch.input_ids
                    # Base model, inference mode: disable adapters
                    model.lm.disable_adapters()
                    outputs_base = model(query_ids, past_key_values=past_key_values)
                    t_logits = outputs_base.logits.cpu()[:, -1, :].squeeze()[
                        verbalizer_ids
                    ]
                    teacher_logits.append(t_logits)

                    # Back to train mode, enable adapters
                    model.lm.enable_adapters()
                    model.lm.set_adapter(adapter)

                    outputs = model(query_ids)
                    s_logits = outputs.logits.cpu()[:, -1, :].squeeze()[verbalizer_ids]
                    student_logits.append(s_logits)

        result = Result(name="train")

        # teacher_tensor = torch.stack(teacher_logits)
        # student_tensor = torch.stack(student_logits)
        # y_true = torch.tensor(labels)

        # result.logits = (teacher_tensor, student_tensor, y_true)

        # logit_tensor = torch.cat(logit_list)
        # result.evaluate(y_true, logit_tensor)

        logging.info(f"Average copycat loss: {avg_loss}")

        return result, avg_loss

    def _train_model(self, model, tokenizer, optimizer, criterion, train_indices):
        model.train()

        loss_agg = 0.0
        logit_list = []
        ids = []
        vocab_size = len(tokenizer)

        train = self.train_set.select(train_indices)

        dl = DataLoader(
            train,
            batch_size=self.args.batch_size,
            collate_fn=partial(
                collate_demonstrations,
                dh=self.data_handler,
                tokenizer=tokenizer,
                device=self.device,
            ),
        )

        for batch in tqdm(dl):
            optimizer.zero_grad()

            input_ids = batch.input_ids
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            # logging.info("batch size", n)
            # logging.info(demonstration_ids.shape)
            # instance = self.train_set[tr_idx]
            # query = self.data_handler.make_prompt_instance(instance)
            # prompt = demonstrations + query
            # input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            # query_ids = tokenizer.encode(query, return_tensors="pt").to(self.device)

            logits = model(inputs).logits
            vocab_size = logits.shape[-1]
            # logging.info(logits.reshape(-1, vocab_size).shape, targets.reshape(-1).shape)
            # out = out.view(-1, len(tokenizer))
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss_agg += loss.item()
            loss.backward()

            optimizer.step()

        avg_loss = loss_agg / len(train_indices)
        # result = Result(
        #     loss=avg_loss, task_type=self.meta.task_type, result_type="train"
        # )

        # logit_tensor = torch.cat(logit_list)
        # result.evaluate(y_true, logit_tensor)

        logging.info(f"Average loss: {avg_loss}")

        return avg_loss

    def _evaluate_model(self, model, tokenizer, name, demo_key_values=None):
        model.eval()

        dl = DataLoader(
            self.test_set,
            batch_size=1,
            collate_fn=partial(
                collate_queries,
                **dict(
                    dh=self.data_handler,
                    tokenizer=tokenizer,
                    device=self.device,
                ),
            ),
        )

        prob_list = []
        pred_list = []
        target_list = []
        with torch.inference_mode():
            verbalizers = self.data_handler.class_labels
            verbalizer_ids = [tokenizer.encode(verb)[-1] for verb in verbalizers]

            correct_count_exact = 0
            for batch in tqdm(dl):
                input_ids = batch.input_ids
                label = batch.target[0]
                target_list.append(label.cpu())

                outputs = model(input_ids, past_key_values=demo_key_values)
                logits = outputs.logits[:, -1, :].squeeze()
                preds = logits[verbalizer_ids]
                prob_list.append(preds.cpu())
                pred_label = torch.argmax(preds)
                pred_list.append(pred_label.cpu())

                token_id = torch.argmax(logits)
                pred = tokenizer.decode(token_id)
                correct_count_exact += verbalizers[label] == pred

            logging.info(f"Exact match: {correct_count_exact}/{len(self.test_set)}")

        result = Result(name=name)

        y_pred = torch.tensor(pred_list)
        y_true = torch.tensor(target_list)
        y_logits = torch.stack(prob_list)
        result.evaluate(y_true, y_pred)
        result.logits = (y_true, y_logits)

        logging.info(result)

        return result

    def retrieve_sample(self):
        if self.args.num_demos > -1 and self.args.num_demos < len(self.train_set):
            indices = np.random.choice(
                len(self.train_set), self.args.num_demos, replace=False
            ).tolist()
        else:
            indices = np.arange(len(self.train_set)).tolist()

        return indices

    def resample_demonstrations(self, n):
        return np.random.choice(len(self.train_set), n, replace=True).tolist()

    def icl(
        self,
        tokenizer,
        model,
        demo_indices=None,
        demonstrations=None,
    ):

        eval = Evaluation()

        if demonstrations is None:
            demonstration_instances = [self.train_set[idx] for idx in demo_indices]

            demonstrations = []
            for instance in demonstration_instances:
                demo = self.data_handler.make_demonstration(instance)
                demonstrations.append(demo)

            demonstrations = "".join(demonstrations)

        # 2) ICL
        with torch.no_grad():
            model.eval()
            verbalizers = self.data_handler.class_labels
            verbalizer_ids = [tokenizer.encode(verb)[-1] for verb in verbalizers]
            logging.info(verbalizer_ids)

            correct_count = 0
            correct_count_exact = 0
            for i in tqdm(range(len(self.test_set))):
                instance = self.test_set[i]
                query = self.data_handler.make_prompt_instance(instance)
                prompt = demonstrations + query
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                    self.device
                )
                label = self.data_handler.get_target(instance)
                outputs = model(input_ids)
                logits = outputs.logits[:, -1, :].squeeze()

                token_id = torch.argmax(logits)
                pred = tokenizer.decode(token_id).strip()

                preds = logits[verbalizer_ids]

                correct = label == torch.argmax(preds)
                correct_count += correct

                correct_count_exact += verbalizers[label] == pred

                # eval.add_train_result(result)
            result = self._evaluate_model(model, tokenizer)
            eval.add_test_result(result)
            logging.info("**********TEST**********")
            logging.info(result)
            logging.info("************************")

            # torch.save(model.state_dict(), "model.pth")
            logging.info(f"Accuracy: {correct_count}/{len(self.test_set)}")
            logging.info(f"Accuracy: {correct_count_exact}/{len(self.test_set)}")

        return eval

    def single_icl(
        self, tokenizer, create_model_fn=None, model=None, demonstrations=None
    ):

        # 1) Model initialization
        if model is None:
            model = create_model_fn(self.args, self.meta, self.device)
            model.to(self.device)

        # Retrieve a sample.
        indices = self._retrieve_sample()
        logging.info(indices)

        eval = Evaluation()

        if demonstrations is None:
            demonstration_instances = [self.train_set[idx] for idx in indices]

            demonstrations = []
            for instance in demonstration_instances:
                demo = self.data_handler.make_demonstration(instance)
                demonstrations.append(demo)

            single_demos = demonstrations
            pairs = [
                "".join([demonstrations[i], demonstrations[i + 1]])
                for i in range(0, len(demonstrations) - 1, 2)
            ]
            demonstrations = "".join(demonstrations)

        target_list = []
        nearest_prob_list = []
        avg_dist_list = []
        ig_prob_list = []
        ig_probs_list = []

        dl = DataLoader(
            self.test_set,
            batch_size=self.args.batch_size,
            collate_fn=partial(
                collate_queries,
                dh=self.data_handler,
                tokenizer=tokenizer,
                device=self.device,
            ),
        )

        exact_match = 0
        majority_match = 0

        hidden_states = []

        # 2) ICL
        with torch.no_grad():
            model.eval()
            verbalizers = self.data_handler.class_labels
            verbalizer_ids = [tokenizer.encode(verb)[-1] for verb in verbalizers]

            for instance in tqdm(self.test_set):
                query = self.data_handler.make_prompt_instance(instance)
                prompt = demonstrations + query
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                    self.device
                )
                target = self.data_handler.get_target(instance)
                target_list.append(target)
                correct_label = verbalizers[target]
                output = model(input_ids, output_hidden_states=True)
                logits = output.logits[:, -1, :].squeeze()
                hidden = torch.stack(
                    [hs[0][-1].flatten().cpu() for hs in output.hidden_states]
                )  # (batch_size x sequence_length x hidden_size) => (0, -1, :)
                hidden_states.append(hidden)
                # X_i = torch.stack([h[-1][-1].flatten() for h in hidden_states[i]])
                # hidden_states.append(X_i)

                # Exact match
                token_id = torch.argmax(logits)
                token_pred = tokenizer.decode(token_id)
                exact_match += correct_label == token_pred

                # Class probs
                probs = logits[verbalizer_ids].softmax(dim=-1)
                nearest_prob_list.append(probs.cpu())

                # TODO: ig probs for all demonstrations
                ig_probs_all = hypersphere_distance(probs.reshape(-1, 1).cpu())
                ig_probs_list.append(ig_probs_all)

                # Single demonstrations
                demo_probs = []
                k = 1
                for i in range(0, len(single_demos), k):
                    demos = "".join([demo[j] for j in range(i, i + k)])
                    prompt = demos + query
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                        self.device
                    )
                    outputs = model(input_ids)
                    logits = outputs.logits[:, -1, :].squeeze()[verbalizer_ids]
                    demo_probs.append(logits.softmax(dim=-1))

                verbalizer_probs = torch.stack(demo_probs)
                verbalizer_preds = torch.argmax(verbalizer_probs, dim=-1)
                verbalizer_probs_mean = verbalizer_probs.mean(dim=0)
                avg_dist_list.append(verbalizer_probs_mean.cpu())

                ig_probs = hypersphere_distance(verbalizer_probs.cpu())
                ig_prob_list.append(ig_probs.mean(dim=0))

                counter = Counter(verbalizer_preds)
                majority_pred = counter.most_common()[0][0]
                majority_match += target == majority_pred

                # # logging.info(
                # #     "verb probs argmax", torch.argmax(verbalizer_probs.mean(0)).item()
                # # )
                # correct_prob += label == torch.argmax(verbalizer_probs.mean(0)).item()

                # eval.add_train_result(result)
                # result = self._evaluate_model(model, tokenizer)
                # eval.add_test_result(result)
                # logging.info("**********TEST**********")
                # logging.info(result)
                # logging.info("************************")

            # torch.save(model.state_dict(), "model.pth")
            logging.info(
                f"Exact: {exact_match}/{len(self.test_set)} | {exact_match / len(target_list):.4f}"
            )
            logging.info(
                f"Majority: {majority_match}/{len(self.test_set)} | {majority_match / len(self.test_set):.4f}"
            )

            probs_all = torch.stack(nearest_prob_list).to(torch.float32)
            probs_chunked = torch.stack(avg_dist_list).to(torch.float32)
            ig_probs = torch.stack(ig_prob_list).to(torch.float32)
            target = torch.tensor(target_list, dtype=torch.float32)
            ig_probs_all = torch.stack(ig_probs_list).to(torch.float32).squeeze()

            result = Result("test-all")
            result.evaluate(target, probs_all)
            eval.add_test_result(result)
            logging.info(result)

            result = Result("test-chunked")
            result.evaluate(target, probs_chunked)
            eval.add_test_result(result)
            logging.info(result)

            result = Result("ig-all")
            result.evaluate(target, ig_probs_all)
            eval.add_test_result(result)
            logging.info(result)

            result = Result("ig-chunked")
            result.evaluate(target, ig_probs)
            eval.add_test_result(result)
            logging.info(result)

            with open("hidden.pkl", "wb") as f:
                pickle.dump((hidden_states, target.cpu()), f)

        return eval

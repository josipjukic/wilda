import torch
from torch.utils.data import DataLoader
from functools import partial

from util import Config


def collate_queries(batch, dh, tokenizer, device, **kwargs):
    target = torch.tensor([dh.get_target(instance) for instance in batch]).to(device)
    # instance_ids = torch.tensor([instance["idx"] for instance in batch]).to(device)

    tokenizer.pad_token = tokenizer.eos_token
    text = [dh.make_prompt_instance(instance) for instance in batch]
    input_ids = tokenizer(text, return_tensors="pt", padding=True)["input_ids"].to(
        device
    )

    return Config(
        {
            "input_ids": input_ids,
            # "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
            "target": target,
            # "instance_ids": instance_ids,
            "original": batch,
        }
    )


def collate_queries_with_demonstrations(
    batch, dh, tokenizer, device, demo_text, **kwargs
):
    target = torch.tensor([dh.get_target(instance) for instance in batch]).to(device)
    # instance_ids = torch.tensor([instance["idx"] for instance in batch]).to(device)

    tokenizer.pad_token = tokenizer.eos_token
    text = [demo_text + dh.make_prompt_instance(instance) for instance in batch]
    input_ids = tokenizer(text, return_tensors="pt", padding=True)["input_ids"].to(
        device
    )

    return Config(
        {
            "input_ids": input_ids,
            # "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
            "target": target,
            # "instance_ids": instance_ids,
            "original": batch,
        }
    )


def collate_demonstrations(batch, dh, tokenizer, device, **kwargs):
    target = torch.tensor([dh.get_target(instance) for instance in batch]).to(device)
    # instance_ids = torch.tensor([instance["idx"] for instance in batch]).to(device)

    tokenizer.pad_token = tokenizer.eos_token
    text = [dh.make_demonstration(instance) for instance in batch]
    input_ids = tokenizer(text, return_tensors="pt", padding=True)["input_ids"].to(
        device
    )

    return Config(
        {
            "input_ids": input_ids,
            # "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
            "target": target,
            # "instance_ids": instance_ids,
            "original": batch,
        }
    )


def create_data_loader(
    dataset,
    batch_size,
    collate_fn,
    collate_fn_kwargs,
):
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, **collate_fn_kwargs),
    )

    return dl

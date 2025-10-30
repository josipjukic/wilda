from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
)


class Transformer(nn.Module):
    def __init__(self, name, args, meta, device, peft=None):
        super().__init__()

        self.name = name
        self.args = args
        self.device = device
        self.peft = peft

        model_cls = MODEL_CLS[meta.task_type]
        name = TRANSFORMERS[name]

        self.lm = model_cls.from_pretrained(
            name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )

    def forward(
        self,
        input_ids,
        **kwargs,
    ):
        outputs = self.lm(
            input_ids=input_ids,
            **kwargs,
        )
        return outputs


def initialize_model(args, meta, device):
    model = Transformer(
        name=args.model,
        args=args,
        meta=meta,
        device=device,
        peft=args.peft,
    )

    return model


TRANSFORMERS = {
    "bert": "bert-base-uncased",
    "electra": "google/electra-base-discriminator",
    "gptj": "EleutherAI/gpt-j-6b",
    "llama-2-7b-it": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama-3-8b-it": "meta-llama/Meta-Llama-3-8B-Instruct",
    "phi-3-mini-it-4k": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-mini-it-128k": "microsoft/Phi-3-mini-128k-instruct",
}


MODEL_CLS = {
    "clf": AutoModelForSequenceClassification,
    "reg": AutoModelForSequenceClassification,
    "seq": AutoModelForTokenClassification,
    "lm": AutoModelForCausalLM,
}

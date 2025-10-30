import random

from tqdm import tqdm
from dataclasses import dataclass

from datasets import load_dataset
from functools import partial

# Bookkeeping
supported_datasets = [
    "commonsense_qa",
    "allenai/ai2_arc",  # ARC-Challenge, ARC-Easy
    "lukaemon/mmlu",  # Task list: https://huggingface.co/datasets/lukaemon/mmlu
    "lucasmccabe/logiqa",
    "nyu-mll/glue",  # Task list: https://huggingface.co/datasets/nyu-mll/glue
]

ANSWER_CANDIDATE_SEP = ", "


class DataHandler:
    def __init__(self):
        super()

    def get_dataset_splits(self):
        pass

    def make_prompt_instance(self, instance):
        pass

    def make_demonstration(self, instance):
        pass

    def get_target(self, instance):
        return instance["label"]

    def label_index(self, label):
        return self.class_labels.index(label)


class LogiQA(DataHandler):
    def __init__(self):
        self.class_labels = ["A", "B", "C", "D"]
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset("lucasmccabe/logiqa")
        return dataset["train"], dataset["test"], dataset["validation"]

    def make_prompt_instance(self, instance):
        template = (
            "Q: {context} {query}\n" + "Answer choices: {answer_choices}\n" + "A:"
        )
        # Format answer choices
        letter_choices = self.class_labels
        text_choices = instance["options"]
        answer_choices = [f"{A} ({L})" for L, A in zip(letter_choices, text_choices)]
        answer_choices = ANSWER_CANDIDATE_SEP.join(answer_choices)

        return template.format(
            context=instance["context"],
            query=instance["query"],
            answer_choices=answer_choices,
        )

    def make_demonstration(self, instance):
        template = (
            "Q: {context} {query}\n"
            + "Answer choices: {answer_choices}\n"
            + "A: {answer}\n"
        )
        # Format answer choices
        letter_choices = self.class_labels
        text_choices = instance["options"]
        answer_choices = [f"{A} ({L})" for L, A in zip(letter_choices, text_choices)]
        answer_choices = ANSWER_CANDIDATE_SEP.join(answer_choices)
        correct_answer_index = instance["correct_option"]
        answer = f"{instance['options'][correct_answer_index]} ({letter_choices[correct_answer_index]})"

        return template.format(
            context=instance["context"],
            query=instance["query"],
            answer_choices=answer_choices,
            answer=answer,
        )

    def get_target(self, instance):
        correct_answer_index = instance["correct_option"]
        return instance["options"][correct_answer_index]


class RTE(DataHandler):
    def __init__(self):
        self.class_labels = ["True", "False"]
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset("nyu-mll/glue", "rte")
        return dataset["train"], dataset["validation"], dataset["test"]

    def make_prompt_instance(self, instance):
        template = (
            "{premise}\n"
            + "Answer with one word. Question: {hypothesis} True or False?\n"
            + "Answer: "
        )

        return template.format(
            premise=instance["sentence1"],
            hypothesis=instance["sentence2"],
        )

    def make_demonstration(self, instance):
        template = (
            "{premise}\n"
            + "Question: {hypothesis} True or False?\n"
            + "Answer: {label}\n"
        )

        return template.format(
            premise=instance["sentence1"],
            hypothesis=instance["sentence2"],
            label=self.class_labels[instance["label"]],
        )


class QNLI(DataHandler):
    def __init__(self):
        self.class_labels = ["Yes", "No"]
        self.label_key = "label"
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset("nyu-mll/glue", "qnli")
        return dataset["train"], dataset["validation"], dataset["test"]

    def make_prompt_instance(self, instance):
        template = (
            "Question: {question}\n"
            + "Sentence: {sentence}\n"
            + "Answer with one word. Does the sentence answer the question, Yes or No?\n"
            + "Answer: "
        )

        return template.format(
            question=instance["question"],
            sentence=instance["sentence"],
        )

    def make_demonstration(self, instance):
        template = (
            "Question: {question}\n"
            + "Sentence: {sentence}\n"
            + "Answer with one word. Does the sentence answer the question, Yes or No?\n"
            + "Answer: {label}\n"
        )

        return template.format(
            question=instance["question"],
            sentence=instance["sentence"],
            label=self.class_labels[instance["label"]],
        )


class QQP(DataHandler):
    def __init__(self):
        self.class_labels = ["No", "Yes"]
        self.label_key = "label"
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset("nyu-mll/glue", "qqp")
        return dataset["train"], dataset["validation"], dataset["test"]

    def make_prompt_instance(self, instance):
        template = (
            "Question 1: {question1}\n"
            + "Question 2: {question2}\n"
            + "Answer with one word. Question: Do both questions ask the same thing, Yes or No? "
            + "Answer: "
        )
        return template.format(
            question1=instance["question1"],
            question2=instance["question2"],
        )

    def make_demonstration(self, instance):
        template = (
            "Question 1: {question1}\n"
            + "Question 2: {question2}\n"
            + "Answer with one word. Question: Do both questions ask the same thing, yes or no?"
            + "Answer: {label}\n"
        )

        return template.format(
            question1=instance["question1"],
            question2=instance["question2"],
            label=self.class_labels[instance["label"]],
        )


class SST(DataHandler):
    def __init__(self):
        self.class_labels = ["Negative", "Positive"]
        self.label_key = "label"
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset("nyu-mll/glue", "sst2")
        return dataset["train"], dataset["validation"], dataset["test"]

    def make_prompt_instance(self, instance):
        template = (
            "Sentence: {sentence}\n"
            + "Answer with one word. Question: Is the sentiment Positive or Negative?"
            + "Answer: "
        )
        return template.format(
            sentence=instance["sentence"],
        )

    def make_demonstration(self, instance):
        template = (
            "Sentence: {sentence}\n"
            + "Answer with one word. Question: Is the sentiment Positive or Negative?"
            + "Answer: {label}\n"
        )
        return template.format(
            sentence=instance["sentence"],
            label=self.class_labels[instance["label"]],
        )


class COLA(DataHandler):
    def __init__(self):
        self.class_labels = ["No", "Yes"]
        self.label_key = "label"
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset("nyu-mll/glue", "cola")
        return dataset["train"], dataset["validation"], dataset["test"]

    def make_prompt_instance(self, instance):
        template = (
            "Sentence: {sentence}\n"
            + "Answer with one word. Question: Is this sentence linguistically acceptable, Yes or No? "
            + "Answer: "
        )
        return template.format(
            sentence=instance["sentence"],
        )

    def make_demonstration(self, instance):
        template = (
            "Sentence: {sentence}\n"
            + "Answer with one word. Question: Is this sentence linguistically acceptable, Yes or No? "
            + "Answer: {label}\n"
        )
        return template.format(
            sentence=instance["sentence"],
            label=self.class_labels[instance["label"]],
        )


class MRPC(DataHandler):
    def __init__(self):
        self.class_labels = ["No", "Yes"]
        self.label_key = "label"
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset("nyu-mll/glue", "mrpc")
        return dataset["train"], dataset["validation"], dataset["test"]

    def make_prompt_instance(self, instance):
        template = (
            "Sentence 1: {sentence1}\n"
            + "Sentence 2: {sentence2}\n"
            + "Answer with one word. Question: Do both sentences say the same thing, Yes or No? "
            + "Answer: "
        )

        return template.format(
            sentence1=instance["sentence1"],
            sentence2=instance["sentence2"],
        )

    def make_demonstration(self, instance):
        template = (
            "Sentence 1: {sentence1}\n"
            + "Sentence 2: {sentence2}\n"
            + "Answer with one word. Question: Do both sentences say the same thing, Yes or No? "
            + "Answer: {label}\n"
        )

        return template.format(
            sentence1=instance["sentence1"],
            sentence2=instance["sentence2"],
            label=self.class_labels[instance["label"]],
        )


class MMLU(DataHandler):
    def __init__(self, subkey):
        self.subkey = subkey
        self.class_labels = ["A", "B", "C", "D"]
        self.label_key = "answer"
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset("cais/mmlu", self.subkey)
        return dataset["validation"], dataset["test"], dataset["dev"]

    def make_prompt_instance(self, instance):
        template = "Q: {question}\n" + "Answer choices: {answer_choices}\n" + "A:"
        # Format answer choices
        answer_choices = [
            f"{L}: {A}" for L, A in zip(self.class_labels, instance["choices"])
        ]
        answer_choices = ANSWER_CANDIDATE_SEP.join(answer_choices)

        return template.format(
            question=instance["question"],
            answer_choices=answer_choices,
        )

    def make_demonstration(self, instance):
        template = (
            "Question: {question}\n"
            + "Answer choices: {answer_choices}\n"
            + "Answer: {answer}\n"
        )
        # Format answer choices
        answer_choices = [
            f"{L}: {A}" for L, A in zip(self.class_labels, instance["choices"])
        ]
        answer_choices = ANSWER_CANDIDATE_SEP.join(answer_choices)
        # Format correct answer
        correct_answer_idx = instance["answer"]
        answer = f"{self.class_labels[correct_answer_idx]}: {instance['choices'][correct_answer_idx]}"

        return template.format(
            question=instance["question"],
            answer_choices=answer_choices,
            answer=answer,
        )

    def get_target(self, instance):
        return instance["answer"]


DATASETS = {
    "qqp": QQP(),
    "rte": RTE(),
    "sst": SST(),
    "mrpc": MRPC(),
    "cola": COLA(),
    "qnli": QNLI(),
    "logiqa": LogiQA(),
    "mmlu-clinic": MMLU("clinical_knowledge"),
    "mmlu-math": MMLU("elementary_mathematics"),
}

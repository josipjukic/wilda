import torch
import torch.nn.functional as F
from util import logits_to_probs
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    brier_score_loss,
)
from scipy.stats import spearmanr, pearsonr
import pandas as pd


class EvalSet:

    def __init__(self, meta, demo_type, name):
        self.meta = meta
        self.demo_type = demo_type
        self.name = name
        self.evals = {}

    def add(self, eval, seed):
        self.evals[seed] = eval

    def to_pandas(self):
        rows = []

        for k, v in self.evals.items():
            for model in v:
                for epoch, test in enumerate(model.test, 1):
                    result_dict = test.to_dict()
                    print(result_dict)
                    row = (
                        {
                            "model": self.meta["model"],
                            "dataset": self.meta["dataset"],
                        }
                        | {
                            "mode": self.name,
                            "demo_type": self.demo_type,
                            "seed": k,
                            "epoch": epoch,
                        }
                        | result_dict
                    )
                    rows.append(row)

        df = pd.DataFrame(rows)
        df.set_index(
            ["model", "dataset", "mode", "demo_type", "seed", "adapter", "epoch"],
            inplace=True,
        )
        return df


class ResultSet:

    def __init__(self, meta, demo_type, name):
        self.meta = meta
        self.demo_type = demo_type
        self.name = name
        self.results = {}

    def add(self, result, seed):
        self.results[seed] = result

    def to_pandas(self):
        rows = []
        for k, v in self.results.items():
            result_dict = v.to_dict()
            row = (
                {
                    "model": self.meta["model"],
                    "dataset": self.meta["dataset"],
                }
                | {
                    "mode": self.name,
                    "demo_type": self.demo_type,
                    "seed": k,
                    "epoch": 1,
                }
                | result_dict
            )
            rows.append(row)

        df = pd.DataFrame(rows)
        df.set_index(
            ["model", "dataset", "mode", "demo_type", "seed", "adapter", "epoch"],
            inplace=True,
        )
        return df


class Evaluation:
    def __init__(self, name):
        self.name = name
        self.special = []
        self.test = []

    def add_special_result(self, result):
        self.special.append(result)

    def add_test_result(self, result):
        self.test.append(result)


class Result:
    def __init__(self, name):
        self.name = name

    def evaluate(self, y_true, y_pred):
        self.y_true = y_true
        # self.probs = probs
        # y_pred = torch.argmax(self.probs, dim=-1)
        self.y_pred = y_pred
        self.accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        self.f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
        self.f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        self.f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
        self.matthews_corrcoef = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
        # self.brier_score = brier_multi(y_true=y_true, y_prob=probs[:, 1])

    def to_dict(self):
        return {
            f"adapter": self.name,
            f"accuracy": self.accuracy,
            f"f1_micro": self.f1_micro,
            f"f1_macro": self.f1_macro,
            f"f1_weighted": self.f1_weighted,
            f"matthews_corrcoef": self.matthews_corrcoef,
        }

    def __repr__(self):
        return "\n".join(
            [
                f"accuracy: {self.accuracy:.4f}",
                # f"F1 micro: {self.f1_micro:.3f}",
                # f"F1 macro: {self.f1_macro:.3f}",
                # f"F1 weighted: {self.f1_weighted:.3f}",
                # f"Matthews corrcoef: {self.matthews_corrcoef:.3f}",
            ]
        )

    def __str__(self):
        return self.__repr__()


class HyperSphere:
    def __init__(self, n_dim):
        self.n_dim = n_dim

    def simplex_to_sphere(self, X):
        return X.sqrt()

    def geodesic_distance(self, p1, p2):
        dot_product = torch.dot(p1, p2)
        return torch.arccos(dot_product)

    def matrix_geodesic_distance(self, X):
        Y = torch.eye(X.shape[-1])
        dot_product = torch.mm(X, Y)
        return torch.arccos(dot_product)


def hsic(G):
    dev = G.device
    m = G.shape[0]

    H = torch.eye(m, device=dev) - (1.0 / m) * torch.ones(G.shape, device=dev)
    prod = H * G * H
    norm = torch.norm(prod, p="fro")
    G_ = prod / norm
    return G_


def angular_cka(X, Y):
    Gx = torch.mm(X, X.T)
    Gy = torch.mm(Y, Y.T)

    Gx_ = hsic(Gx)
    Gy_ = hsic(Gy)

    frobenius_product = torch.trace(torch.mm(Gx_.T, Gy_))
    arc_dist = torch.arccos(frobenius_product)

    return arc_dist


def hypersphere_distance(probs):
    D, N = probs.shape
    probs = F.softmax(probs, dim=-1)
    hs = HyperSphere(n_dim=N)
    sphere_points = hs.simplex_to_sphere(probs)
    dists = hs.matrix_geodesic_distance(sphere_points)
    return dists


def completion_probabilities(model, tokenizer, prefix, targets):
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids  # [1, T]
    prefix_length = prefix_ids.size(-1)

    n_sequences = len(targets)
    n_prefix_ids = prefix_ids.repeat(n_sequences, 1)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    # Convert targets to tensors, concat to inputs
    target_ids = tokenizer(targets, padding=True, return_tensors="pt").input_ids

    # Count lengths of individual target sequences for scaling
    non_pad = target_ids != pad_token_id
    lengths = torch.count_nonzero(non_pad, dim=-1)

    # Stack inputs
    input = torch.hstack(
        [n_prefix_ids, target_ids[:, :-1]]  # Exclude last target token from input
    )

    # Fwd pass
    outputs = model.forward(input, return_dict=True)  # logits = B, T, V
    relevant_logits = outputs["logits"][:, prefix_length - 1 :]
    # Any benefits from logsoftmax if we want the actual probability in the end?
    token_probs = torch.softmax(relevant_logits, dim=-1)

    # Set pad probabilities to one for .prod()
    token_probs[:, :, pad_token_id] = 1.0
    target_probs = token_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze()

    seq_probs = torch.prod(target_probs, dim=1)
    seq_probs /= lengths

    return seq_probs


if __name__ == "__main__":
    probs = torch.randn(10, 4)
    dists = hypersphere_distance(probs)
    print(dists)

# **Disentangling Latent Shifts of In-Context Learning with Weak Supervision** 
Official repository for **NeurIPS 2025** paper

**Josip Jukić, Jan Šnajder**  
TakeLab

[📄 Full paper](https://openreview.net/forum?id=tAq9Gxdhr0)


---

## Overview

This repository contains the **official implementation of WILDA**, a parameter-efficient method for stabilizing and accelerating in-context learning (ICL) by **disentangling demonstration-induced latent shifts from the query representation** using weak supervision.

WILDA treats ICL as a source of *noisy supervision* and distills its effects into compact adapters that:
- eliminate the need to reprocess demonstrations at inference,
- improve stability with respect to prompt ordering and selection,
- enable compositional reuse of contextual knowledge,
- and outperform the original ICL teacher through weak-to-strong generalization.

---

## Paper Summary

### Motivation

In-context learning is flexible but unstable:
- sensitive to demonstration order,
- inefficient for long contexts,
- and fragile to noisy prompts.

From a mechanistic perspective, demonstrations induce **latent activation shifts** inside the model that interfere with the query representation. Existing work approximates these shifts by operating on attention heads or hidden states — but does not capture the *full effect* of context on model behavior.

---

### Method: WILDA

WILDA introduces a **teacher–student framework** to encode ICL behavior into adapters:

1. **Teacher (ICL model)**  
   Runs standard in-context learning with demonstrations.

2. **Pseudo-labeling**  
   The ICL teacher generates distributions over outputs on unlabeled queries.

3. **Student (adapter-augmented model)**  
   Learns to reproduce teacher outputs **using only the query**, updating a small adapter module (LoRA).

This causes demonstration effects to be:
- stored parametrically,
- reusable across prompts,
- and composable across tasks.

> WILDA bypasses architectural interventions and directly **learns the latent ICL shift from outputs**.

---

### Weak-to-Strong Generalization

Despite training only on noisy pseudo-labels, the student often exceeds the teacher.  
Two emergent behaviors explain this:

- **Pseudo-label correction**  
  The model systematically fixes inconsistent or unstable teacher outputs.

- **Coverage expansion**  
  Learning generalizes beyond regions supported by demonstrations.

These behaviors match recent theory on weak-to-strong learning and are empirically confirmed via:
- Lipschitz analysis,
- correction rates over training,
- distance-based behavior on OOD points.

---

### Results

WILDA was evaluated on:

- **GLUE** tasks (RTE, SST, QNLI, MNLI, COLA, MRPC, QQP)
- **MMLU** subsets (MATH, MISC)
- Models: LLaMA-3 (8B), Phi-3, LLaMA-2

Findings:
- Higher accuracy than standard ICL
- Dramatically reduced variance across prompt ordering  
- Strong OOD generalization  
- Stable behavior with large prompts  
- Efficient inference (no demonstrations at test time)

WILDA also supports **adapter arithmetic**, enabling the merging of multiple adapters trained on disjoint contexts — achieving scalable knowledge fusion.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{
  jukic2025disentangling,
  title={Disentangling Latent Shifts of In-Context Learning with Weak Supervision},
  author={Josip Juki{\'c} and Jan {\v{S}}najder},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=tAq9Gxdhr0}
}

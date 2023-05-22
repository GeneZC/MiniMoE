## MiniMoE

This repository contains code for ACL 2023 paper titled [Lifting the Curse of Capacity Gap in Distilling Language Models]().

**************************** **Updates** ****************************

<!-- Thanks for your interest in our repo! -->

* Code is under preparation. Stay tuned.

## Quick Links

  - [Overview](#overview)
  - [Getting Started](#getting-started)
    - [Requirements](#requirements)
    - [Distillation](#distillation)
    - [Finetuning](#finetuning)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Overview

Pretrained language models (LMs) have shown compelling performance on various downstream tasks, but unfortunately they require a tremendous amount of inference compute. Knowledge distillation finds a path to compress LMs to small ones with a teacher-student paradigm. However, when the capacity gap between the teacher and the student is large, a curse of capacity gap appears, invoking a deficiency in distilling LMs. While a few studies have been carried out to fill the gap, the curse is not yet well tackled. In this paper, we aim at lifting the curse of capacity gap via enlarging the capacity of the student without notably increasing the inference compute. Largely motivated by sparse activation regime of mixture of experts (MoE), we propose a mixture of minimal experts (MiniMoE), which imposes extra parameters to the student but introduces almost no additional inference compute. Experimental results on GLUE and CoNLL demonstrate the curse of capacity gap is lifted by the magic of MiniMoE to a large extent. MiniMoE also achieves the state-of-the-art performance at small FLOPs compared with a range of competitive baselines. With a compression rate as much as $\sim$50$\times$, \textsc{MiniMoE} preserves $\sim$95\% GLUE score of the teacher.

<img src="./assets/minimoe_motivation.png" alt="minimoe" align=center/>

## Getting Started

### Requirements

- PyTorch
- Numpy
- Transformers

### Distillation

**Wikipedia Data**

### Finetuning

**GLUE & CoNLL Data**

<!--
### GLUE Data

Get GLUE data through the [link](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py) and put it to the corresponding directory. For example, MRPC dataset should be placed into `datasets/mrpc`.

### CoNLL Data

### Training & Evaluation


The training and evaluation are achieved in several scripts. We provide example scripts as follows.

**Finetuning**

We provide an example of finetuning `bert-base-uncased` on RTE in `scripts/run_finetuning_rte.sh`. We explain some important arguments in following:
* `--model_type`: Variant to use, should be `ft` in the case.
* `--model_path`: Pretrained language models to start with, should be `bert-base-uncased` in the case and can be others as you like.
* `--task_name`: Task to use, should be chosen from `rte`, `mrpc`, `stsb`, `sst2`, `qnli`, `qqp`, `mnli`, and `mnlimm`.
* `--data_type`: Input format to use, default to `combined`.

**Pruning**

We provide and example of pruning a finetuned checkpoint on RTE in `scripts/run_pruning_rte.sh`. The arguments should be self-contained.

**Distillation**

We provide an example of distilling a finetuned teacher to a layer-dropped or parameter-pruned student on RTE in `scripts/run_distillation_rte.sh`. We explain some important arguments in following:
* `--model_type`: Variant to use, should be `kd` in the case.
* `--teacher_model_path`: Teacher models to use, should be the path to the finetuned teacher checkpoint.
* `--student_model_path`: Student models to initialize, should be the path to the pruned/finetuned teacher checkpoint depending on the way you would like to initialize the student.
* `--student_sparsity`: Student sparsity, should be set if you would like to use parameter-pruned student, e.g., 70. Otherwise, this argument should be left blank.
* `--student_layer`: Student layer, should be set if you would like to use layer-dropped student, e.g., 4.

**Teacher Sparsification**

We provide an example of sparsfying the teacher based on the student on RTE in `scripts/run_sparsification_rte.sh`. We explain some important arguments in following:
* `--model_type`: Variant to use, should be `kd` in the case.
* `--teacher_model_path`: Teacher models to use, should be the path to the finetuned teacher checkpoint.
* `--student_model_path`: Student models to use, should be the path to the distilled student checkpoint.
* `--student_sparsity`: Student sparsity, should be set if you would like to use parameter-pruned student, e.g., 70. Otherwise, this argument should be left blank.
* `--student_layer`: Student layer, should be set if you would like to use layer-dropped student, e.g., 4.
* `--lam`: the knowledgeableness tradeoff term to keep a balance between expressiveness and student-friendliness.

**Rewinding**

We provide an example of rewinding the student on RTE in `scripts/run_rewinding_rte.sh`. We explain some important arguments in following:
* `--model_type`: Variant to use, should be `kd` in the case.
* `--teacher_model_path`: Teacher models to use, should be the path to the sparsified teacher checkpoint.
* `--student_model_path`: Student models to initialize, should be the path to the pruned/finetuned teacher checkpoint depending on the way you would like to initialize the student.
* `--student_sparsity`: Student sparsity, should be set if you would like to use parameter-pruned student, e.g., 70. Otherwise, this argument should be left blank.
* `--student_layer`: Student layer, should be set if you would like to use layer-dropped student, e.g., 4.
* `--lam`: the knowledgeableness tradeoff term to keep a balance between expressiveness and student-friendliness. Here, it is just used for folder names.
! -->

## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Chen (`czhang@bit.edu.cn`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use the code in your work:

```bibtex
@inproceedings{zhang2022minimoe,
   title={Lifting the Curse of Capacity Gap in Distilling Language Models},
   author={Zhang, Chen and Yang, Yang and Liu, Jiahao and Wang, Jingang and Xian, Yunsen and Wang, Benyou and Song, Dawei},
   booktitle={ACL},
   year={2023}
}
```


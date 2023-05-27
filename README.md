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

Pretrained language models (LMs) have shown compelling performance on various downstream tasks, but unfortunately they require a tremendous amount of inference compute. Knowledge distillation finds a path to compress LMs to small ones with a teacher-student paradigm. However, when the capacity gap between the teacher and the student is large, a curse of capacity gap appears, invoking a deficiency in distilling LMs. While a few studies have been carried out to fill the gap, the curse is not yet well tackled. In this paper, we aim at lifting the curse of capacity gap via enlarging the capacity of the student without notably increasing the inference compute. Largely motivated by sparse activation regime of mixture of experts (MoE), we propose a mixture of minimal experts (MiniMoE), which imposes extra parameters to the student but introduces almost no additional inference compute. Experimental results on GLUE and CoNLL demonstrate the curse of capacity gap is lifted by the magic of MiniMoE to a large extent. MiniMoE also achieves the state-of-the-art performance at small FLOPs compared with a range of competitive baselines. With a compression rate as much as 50 times, MiniMoE preserves 95% GLUE score of the teacher.

<img src="./assets/minimoe_motivation.png" alt="minimoe" align=center/>

## Getting Started

### Requirements

- PyTorch
- Numpy
- Transformers

### Distillation

**Wikipedia Data**

Download Wikipedia dump through the [link](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2). Process the dump with [wikiextractor](https://github.com/attardi/wikiextractor), which will convert the dump into this [format](https://github.com/attardi/wikiextractor/wiki/File-Format).

Format the processed Wikipedia to the format that aligns with the style of BERT pretraining, i.e., one sentence per line in one document, and one empty line between two documents. We provide example scripts of formatting the Wikipedia in `scripts/bert_format_wiki.sh`. We explain some importance arguments in the following:
* `--input_dir`: directory to input files.
* `--input_regex`: regex to enumerate input files in above `input_dir`.
* `--output_dir`: directory to save formatted files.
* `--num_processors`: num of precessors to conduct multiprocessing.

Build the formatted Wikipedia with TFRecord so that we could later read the data in a stream-like manner. This could help to save much cpu memory when we are faced with huge volume of data. We provide example scripts of building the Wikipedia in `scripts/bert_build_wiki.sh`. We explain some importance arguments in the following:
* `--input_dir`: same as above.
* `--input_regex`: same as above.
* `--output_dir`: same as above.
* `--tokenizer_name_or_path`: name or path to the tokenizer, used to tokenize the sentences.
* `--do_lower_case`: whether the sentences should be lower cased or not.
* `--max_seq_length`: the maximum sequenth length of the input, used to truncate the input if necessary.
* `--num_processors`: same as above.

**Distillation**

Distil with MiniLM.

Distil with MiniLM w/ TA.

Distil with MiniMoE.

### Finetuning

Please refer to our previous work [StarK](https://github.com/GeneZC/StarK/blob/main/run_finetuning.py) and [MiniDisc](https://github.com/GeneZC/MiniDisc/blob/main/run_distillation_ner.py) for finetuning on GLUE and CoNLL, respectively.

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


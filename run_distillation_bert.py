# -*- coding: utf-8 -*-

import os
import time
import glob
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

import transformers
from transformers import AdamW, Adafactor, get_scheduler

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from filelock import FileLock

from data import get_pipeline_class, TFRecordReader, TFRecordDataset, TFRecordDistributedDataset
from models import get_model_class
from utils import set_seed, add_kwargs_to_config, keep_recent_ckpt, Logger, AverageMeter

from torch.utils.tensorboard import SummaryWriter

logger = Logger()


def acc(preds, labels):
    acc = accuracy_score(y_true=labels, y_pred=preds)
    return {"acc": acc}


def gather(tensor):
    output_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # output = concat[:num_examples] # Truncate dummy elements added by DistributedSampler.
    output = concat
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Compressing a transformers model via disitllation.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Type of pretrained model, for indexing model class.",   
    )
    parser.add_argument( # We'd better download the model for ease of use.
        "--teacher_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",    
    )
    parser.add_argument(
        "--student_model_name_or_path",
        type=str,
        required=True,
        help="Path to configurated model.",   
    )
    parser.add_argument(
        "--record_path_or_regex",
        type=str,
        required=True,
        help="Where to load the records.",
    )
    parser.add_argument( # NIL for distillation.
        "--data_type",
        type=str,
        required=True,
        help="Type of formatted data, for indexing data pipeline.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs", 
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the training loader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the evaluation loader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--log_interval", type=int, default=1000, help="Interval of logging and possible saving.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--num_grad_accum_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_proportion", type=float, default=0.01, help="Proportion of the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Maximum norm of gradients."
    )
    parser.add_argument("--seed", type=int, default=776, help="A seed for reproducible training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 or not.")
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU or not.")
    parser.add_argument("--model_suffix", type=str, default="none", help="Suffix for outputs.")
    parser.add_argument(
        "--num_relation_heads",
        type=int,
        default=-1,
        help="Num of attention heads so that attention scores can be aligned.",    
    )

    parser.add_argument("--iteration", type=int, default=-1, help="Iteration.")
    parser.add_argument("--layer", type=int, default=4, help="Layer.")
    parser.add_argument("--hidden", type=int, default=384, help="Hidden.")
    parser.add_argument("--moe", type=str, default="1,4", help="MoE.")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    args.output_dir = os.path.join(args.output_dir, f"{args.model_type}_{args.model_suffix}_{args.iteration}")
    os.makedirs(args.output_dir, exist_ok=True)

    is_dist = (args.local_rank != -1)
    if is_dist:
        # Initialize DDP.
        dist.init_process_group(backend='nccl')
        # Pin GPU to be used to process local rank (one GPU per process).
        torch.cuda.set_device(args.local_rank)
    is_main = (args.local_rank == -1 or dist.get_rank() == 0) 
    is_fp16 = args.use_fp16
    device = torch.device("cpu") if args.use_cpu else torch.device("cuda")

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.add_stream_handler()
    logger.add_file_handler(args.output_dir)
    if is_main:
        transformers.utils.logging.set_verbosity_warning()
        logger.set_verbosity_info()
        summary = SummaryWriter(args.output_dir)
    else:
        transformers.utils.logging.set_verbosity_error()
        logger.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    logger.info(f"***** Use FP16 {is_fp16}! *****")
    logger.info("***** Configuration ready! *****")

    # Load record reader.
    data_reader = TFRecordReader(args.record_path_or_regex, 
        description={"indices": "int", "segments": "int"})
    
    logger.info("***** Data ready! *****")

    # Get classes which shall be used.
    tokenizer_class, config_class, model_class = get_model_class(args.model_type)
    pipeline_class = get_pipeline_class(args.data_type)

    # Load pretrained tokenizer with necessary resizing.
    tokenizer = tokenizer_class.from_pretrained(args.teacher_model_name_or_path, use_fast=not args.use_slow_tokenizer)
    
    # Data pipeline.
    data_pipeline = pipeline_class(tokenizer, args.max_length)

    if is_dist:
        train_dataset = TFRecordDistributedDataset(data_reader, shuffle=True)
    else:
        train_dataset = TFRecordDataset(data_reader, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, collate_fn=data_pipeline.collate)
    
    # if is_dist:
    #     dev_dataset = TFRecordDistributedDataset(dev_examples, shuffle=False)
    # else:
    #     dev_dataset = TFRecordDataset(dev_examples, shuffle=False)
    # dev_loader = DataLoader(dev_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=data_pipeline.collate)
        
    # Initialize, then rewrite or add kwargs of original config for distillaiton alignment.
    t_config = config_class.from_pretrained(args.teacher_model_name_or_path)
    assert args.num_relation_heads != -1, "Relation head number is not set properly."
    add_kwargs_to_config(t_config, num_relation_heads=args.num_relation_heads)
    if args.iteration == 1:
        if "1b" in args.model_suffix:
            from models import MegatronBertMiniLM
            t_model = MegatronBertMiniLM.from_pretrained(
                args.teacher_model_name_or_path,
                config=t_config,
            )
        elif "ro" in args.model_suffix:
            from models import RobertaMiniLM
            t_model = RobertaMiniLM.from_pretrained(
                args.teacher_model_name_or_path,
                config=t_config,
            )
        else:
            from models import BertMiniLM
            t_model = BertMiniLM.from_pretrained(
                args.teacher_model_name_or_path,
                config=t_config,
            )
    else:
        t_model = model_class.from_pretrained(
            args.teacher_model_name_or_path,
            config=t_config,
        )
    t_model = t_model.to(device)
    if is_dist:
        t_model = DistributedDataParallel(t_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        
    s_config = config_class.from_pretrained(args.student_model_name_or_path)
    assert args.num_relation_heads != -1, "Relation head number is not set properly."
    add_kwargs_to_config(s_config, num_relation_heads=args.num_relation_heads)
    add_kwargs_to_config(s_config, num_hidden_layers=args.layer)
    add_kwargs_to_config(s_config, hidden_size=args.hidden)
    add_kwargs_to_config(s_config, attention_head_size=int(args.hidden/12))
    if "moe" in args.model_type:
        add_kwargs_to_config(s_config, intermediate_size=int(args.moe.split(",")[1])*4*args.hidden)
        add_kwargs_to_config(s_config, num_attention_heads=int(args.moe.split(",")[0])*12) 
    else:
        add_kwargs_to_config(s_config, intermediate_size=4*args.hidden)
        add_kwargs_to_config(s_config, num_attention_heads=12) # Fixed.
    s_model = model_class(
        config=s_config,
    )
    if "moe" in args.model_type:
        base_model = getattr(s_model, s_model.base_model_prefix, s_model)
        base_model.moefy(args.moe)
        s_config.moe = args.moe
    s_model = s_model.to(device)
    if is_dist:
        s_model = DistributedDataParallel(s_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "ln.weight", "layer_norm.weight"]
    grouped_parameters = [
        {
            "params": [p for n, p in s_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in s_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.num_grad_accum_steps)
    num_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )

    logger.info("***** Model & Opitimizer ready! *****")

    # Train!
    total_batch_size = args.per_device_train_batch_size * args.num_grad_accum_steps
    if is_dist:
        total_batch_size = total_batch_size * dist.get_world_size()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. accumulation, parallel & distributed) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.num_grad_accum_steps}")
    logger.info(f"  Total optimization steps = {num_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(num_train_steps), disable=not is_main)
    num_completed_steps = 0
    train_losses = AverageMeter(args.num_grad_accum_steps)

    if is_fp16:
        scaler = amp.GradScaler()

    t_model.eval()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_loader):
            s_model.train()
            batch = [v.to(device) for k, v in batch._asdict().items()]
            if is_fp16:
                with amp.autocast():
                    with torch.no_grad():
                        t_output = t_model(batch)
                    s_output = s_model(batch)
                    loss = model_class.loss_fn(t_output, s_output) 
            else:
                with torch.no_grad():
                    t_output = t_model(batch)
                s_output = s_model(batch)
                loss = model_class.loss_fn(t_output, s_output)
            train_losses.update(loss.item())
            loss = loss / args.num_grad_accum_steps
            if is_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % args.num_grad_accum_steps == 0 or step == len(train_loader) - 1:
                if is_fp16:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(s_model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer) # Will check whether the gradients are unscaled or not before stepping.
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(s_model.parameters(), args.max_grad_norm)
                    optimizer.step()    
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                num_completed_steps += 1
                if is_main:
                    summary.add_scalar("loss/train", train_losses.avg, num_completed_steps)
            
                if num_completed_steps % args.log_interval == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info(f"  Num completed epochs = {epoch}")
                    logger.info(f"  Num completed steps = {num_completed_steps}")
                    # model.eval()
                    # with torch.no_grad():
                    #     preds, labels = [], []
                    #     for batch in dev_loader:
                    #         batch = [v.to(device) for k, v in batch._asdict().items()]
                    #         output = model(batch)
                    #         pred, label = output.prediction, output.label
                    #         if is_dist:
                    #             preds.extend(gather(pred).cpu().numpy().tolist())
                    #             labels.extend(gather(label).cpu().numpy().tolist())
                    #         else:
                    #             preds.extend(pred.cpu().numpy().tolist())
                    #             labels.extend(label.cpu().numpy().tolist())

                    dev_metric = {}
                    dev_metric.update(**{"loss": train_losses.avg})
                    logger.info(f"  Train loss = {train_losses.avg}")
                    # logger.info(f"  Dev metric = {dev_metric}")

                    logger.info("***** Saving the current *****")
                    time_stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
                    save_dir = os.path.join(args.output_dir, \
                        f"ckpt-{num_completed_steps}-{time_stamp}")
                    os.makedirs(save_dir, exist_ok=True)
                    if is_main:
                        getattr(s_model, "module", s_model).save_pretrained(save_dir)
                    if is_main:
                        tokenizer.save_pretrained(save_dir)
                        s_config.save_pretrained(save_dir)
                        keep_recent_ckpt(args.output_dir, 1)

    logger.info("***** Finalizing training *****")
    logger.info("***** Saving the last *****")
    # time_stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    last_dir = os.path.join(args.output_dir, \
        "ckpt-last")
    os.makedirs(last_dir, exist_ok=True)
    if is_main:
        getattr(s_model, "module", s_model).save_pretrained(last_dir)
    if is_main:
        tokenizer.save_pretrained(last_dir)
        s_config.save_pretrained(last_dir)
    

if __name__ == "__main__":
    """
    1. Single-Node multi-process distributed training

    ::

        >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
                YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
                arguments of your training script)

    2. Multi-Node multi-process distributed training: (e.g. two nodes)


    Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

    ::

        >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
                --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
                --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
                and all other arguments of your training script)

    Node 2:

    ::

        >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
                --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
                --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
                and all other arguments of your training script)
    """
    main()
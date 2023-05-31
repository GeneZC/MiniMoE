
python -m torch.distributed.launch --nproc_per_node=8 --master_port 37776 run_distillation_bert.py \
    --model_type moebert_minilm \
    --teacher_model_name_or_path bert-base-uncased \
    --student_model_name_or_path bert-base-uncased \
    --record_path_or_regex "path/to/builded/wikipedia_bert_128/wiki_*.format.tfrecord" \
    --data_type bert_nil \
    --output_dir path/to/outputs \
    --max_length 128 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --learning_rate 3e-4 \
    --weight_decay 1e-2 \
    --log_interval 1000 \
    --num_train_epochs 5 \
    --warmup_proportion 0.01 \
    --max_grad_norm 5.0 \
    --seed 776 \
    --use_fp16 \
    --num_relation_heads 32 \
    --iteration 1 \
    --layer 3 \
    --hidden 384 \
    --moe 1,4 \
    --model_suffix 3L_384H_1,4E
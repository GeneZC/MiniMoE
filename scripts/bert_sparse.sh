
python run_sparsification_bert.py \
    --model_type sparsebert_mlm \
    --teacher_model_name_or_path bert-base-uncased \
    --record_path_or_regex "path/to/builded/wikipedia_bert_128/wiki_00.format.tfrecord" \
    --data_type bert_mlm \
    --output_dir path/to/outputs \
    --max_length 128 \
    --per_device_eval_batch_size 128
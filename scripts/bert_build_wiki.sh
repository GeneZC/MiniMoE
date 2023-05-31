
python run_building_data_bert.py \
    --input_dir path/to/formatted/wikipedia \
    --input_regex "wiki_*.format" \
    --output_dir path/to/builded/wikipedia_bert_128 \
    --tokenizer_name_or_path bert-base-uncased \
    --do_lower_case \
    --max_seq_length 128 \
    --num_processors 16
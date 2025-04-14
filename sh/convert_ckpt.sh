CKPT_DIR=""     # path to the dir named actor
OUTPUT_DIR=""   # path to the output dir
python verl/scripts/model_merger.py \
    --backend fsdp \
    --tie-word-embedding \
    --hf_model_path Qwen/Qwen2.5-0.5B \
    --local_dir $CKPT_DIR \
    --target_dir $OUTPUT_DIR
echo "Converted $step"
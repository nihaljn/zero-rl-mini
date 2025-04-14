CKPT_PATH=""
EXP_NAME=""
MODE="greedy"
SAMPLING_TEMPERATURE=1.0
SAMPLING_N=5

if [ "$MODE" = "greedy" ]; then
    echo "=== Greedy evaluation ==="
    python3 evaluate.py \
        --max_samples -1 \
        --temperature 0.0 \
        --n_samples 1 \
        --batch_size 50 \
        --response_extractor qwen \
        --seed 100 \
        --max_new_tokens 1500 \
        --use_compile \
        --load_in_half \
        --ckpt_path $CKPT_PATH \
        --exp_name $EXP_NAME
else
    echo "=== Sampling evaluation: temperature = $SAMPLING_TEMPERATURE, n = $SAMPLING_N ==="
    python3 evaluate.py \
        --max_samples -1 \
        --temperature $SAMPLING_TEMPERATURE \
        --n_samples $SAMPLING_N \
        --batch_size 50 \
        --response_extractor qwen \
        --seed 100 \
        --max_new_tokens 1500 \
        --use_compile \
        --load_in_half \
        --ckpt_path $CKPT_PATH \
        --exp_name $EXP_NAME
fi

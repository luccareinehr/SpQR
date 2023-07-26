export MODEL_PATH="/Users/lucrei01/SpQR/models/open_llama_3b_v2"
export CUSTOM_DATA_PATH="/Users/lucrei01/SpQR/data/red_pajama_n=1024.pth"

echo "Launching main.py..."
python main.py $MODEL_PATH custom \
    --custom_data_path=$CUSTOM_DATA_PATH \
    --wbits 4 \ 
    --groupsize 16 \
    --perchannel \
    --qq_scale_bits 3 \
    --qq_zero_bits 3 \
    --qq_groupsize 16 \
    --outlier_threshold=0.2 \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128 \
    --wandb
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

DATA_DIR=/mnt/data/zkliu/llava_evaluation
MODEL_PATH=/mnt/data/zkliu/hf_models/llava-v1.5-7b
# MODEL_PATH=/mnt/data/zkliu/llava_for_okvqa/llava_pretrain_no_multitask

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $DATA_DIR/combined_data_v0/combined_data_v0.jsonl \
        --image-folder /mnt/data/zkliu \
        --answers-file $DATA_DIR/combined_data_v0/combined_data_v0.jsonl/answers/${CHUNKS}_${IDX}.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \ &
done

wait

output_file=$DATA_DIR/combined_data_v0/answers/merge.jsonl

> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat  $DATA_DIR/combined_data_v0/combined_data_v0.jsonl/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
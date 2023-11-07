DATA_DIR=/mnt/data/zkliu/llava_evaluation
# MODEL_PATH=/mnt/data/zkliu/hf_models/llava-v1.5-7b
# MODEL_PATH=/mnt/data/zkliu/llava_for_okvqa/llava_pretrain_no_multitask
MODEL_PATH=/mnt/data/zkliu/llava_for_okvqa/llava_t1_v1

python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/vizwiz/vizwiz_val_full.json \
    --image-folder $DATA_DIR/vizwiz/val \
    --answers-file $DATA_DIR/vizwiz/answers/llava-t1_v1-7b_f.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# CUDA_VISIBLE_DEVICES=5 bash scripts/v1_5/eval/new_scripts/vizwiz_full.sh
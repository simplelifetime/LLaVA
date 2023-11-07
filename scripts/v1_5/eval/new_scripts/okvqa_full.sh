DATA_DIR=/mnt/data/zkliu/llava_evaluation
# MODEL_PATH=/mnt/data/zkliu/hf_models/llava-v1.5-7b
# MODEL_PATH=/mnt/data/zkliu/llava_for_okvqa/llava_pretrain_no_multitask
MODEL_PATH=/mnt/data/zkliu/llava_for_okvqa/llava_t1_v1

python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/okvqa/okvqa_test_full.json \
    --image-folder /mnt/data/zkliu/datasets/coco/images \
    --answers-file $DATA_DIR/okvqa/answers/llava-t1_v1-7b_f.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# python -m llava.eval.eval_textvqa \
#     --annotation-file $DATA_DIR/textvqa/TextVQA_0.5.1_val.json \
#     --result-file $DATA_DIR/textvqa/answers/llava-v1.5-7b.jsonl

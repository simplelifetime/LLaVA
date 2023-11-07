result_dir=/mnt/data/zkliu/llava_evaluation
MODEL_PATH=/mnt/data/zkliu/llava_for_okvqa/llava_t3_v1
MODEL_NAME=llava_t3_v1
DATA_DIR=/mnt/data/zkliu/llava_evaluation

bash finetune_v1.sh
# OKVQA
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/okvqa/okvqa_test.json \
    --image-folder /mnt/data/zkliu/datasets/coco/images \
    --answers-file $DATA_DIR/okvqa/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 &

# python -m llava.eval.eval_okvqa \
#     --annotation-file $DATA_DIR/okvqa/okvqa_val_eval.json \
#     --result-file $DATA_DIR/okvqa/answers/${MODEL_NAME}.jsonl \
#     --lemmatize

# TextVQA
CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $DATA_DIR/textvqa/train_images \
    --answers-file $DATA_DIR/textvqa/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 &

# python -m llava.eval.eval_textvqa \
#     --annotation-file $DATA_DIR/textvqa/TextVQA_0.5.1_val.json \
#     --result-file $DATA_DIR/textvqa/answers/${MODEL_NAME}.jsonl


# VizWiz
CUDA_VISIBLE_DEVICES=2 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/vizwiz/llava_test.jsonl \
    --image-folder $DATA_DIR/vizwiz/test \
    --answers-file $DATA_DIR/vizwiz/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 &

# python scripts/convert_vizwiz_for_submission.py \
#     --annotation-file $DATA_DIR/vizwiz/llava_test.jsonl \
#     --result-file $DATA_DIR/vizwiz/answers/${MODEL_NAME}.jsonl \
#     --result-upload-file $DATA_DIR/vizwiz/answers_upload/${MODEL_NAME}.json

# AOKVQA
CUDA_VISIBLE_DEVICES=3 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/aokvqa/aokvqa_test.json \
    --image-folder /mnt/data/zkliu/datasets/coco/images \
    --answers-file $DATA_DIR/aokvqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 


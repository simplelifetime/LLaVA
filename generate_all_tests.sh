# MODEL_PATH=/mnt/data/zkliu/llava_for_okvqa/llava_t5_v1
MODEL_NAME=llava_t5_v7
MODEL_PATH=/mnt/data/zkliu/llava_for_okvqa/$MODEL_NAME
DATA_DIR=/mnt/data/zkliu/llava_evaluation

# bash finetune_v1.sh
# OKVQA
CUDA_VISIBLE_DEVICES=4 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/okvqa/okvqa_test.json \
    --image-folder /mnt/data/zkliu/datasets/coco/images \
    --answers-file $DATA_DIR/okvqa/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 &

# TextVQA
CUDA_VISIBLE_DEVICES=5 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $DATA_DIR/textvqa/train_images \
    --answers-file $DATA_DIR/textvqa/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 &

# VizWiz test
# CUDA_VISIBLE_DEVICES=7 python -m llava.eval.model_vqa_loader \
#     --model-path $MODEL_PATH \
#     --question-file $DATA_DIR/vizwiz/llava_test.jsonl \
#     --image-folder $DATA_DIR/vizwiz/test \
#     --answers-file $DATA_DIR/vizwiz/answers/${MODEL_NAME}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1 &


# Vizwiz val
CUDA_VISIBLE_DEVICES=6 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/vizwiz/vizwiz_val.json \
    --image-folder $DATA_DIR/vizwiz/val \
    --answers-file $DATA_DIR/vizwiz/val_answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 &

# AOKVQA test
# CUDA_VISIBLE_DEVICES=3 python -m llava.eval.model_vqa_loader \
#     --model-path $MODEL_PATH \
#     --question-file $DATA_DIR/aokvqa/aokvqa_test.json \
#     --image-folder /mnt/data/zkliu/datasets/coco/images \
#     --answers-file $DATA_DIR/aokvqa/answers/$MODEL_NAME.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1 
\

# AOKVQA val
CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/aokvqa/aokvqa_val.json \
    --image-folder /mnt/data/zkliu/datasets/coco/images \
    --answers-file $DATA_DIR/aokvqa/val_answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 &

# GQA val
CUDA_VISIBLE_DEVICES=7 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/gqa/llava_gqa_testdev_balanced.jsonl \
    --image-folder /mnt/data/zkliu/llava_datasets/gqa/images \
    --answers-file $DATA_DIR/gqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 &

# MME val
CUDA_VISIBLE_DEVICES=2 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/MME/llava_mme.jsonl \
    --image-folder $DATA_DIR/MME/MME_Benchmark_release_version \
    --answers-file $DATA_DIR/MME/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
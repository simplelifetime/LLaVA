result_dir=/mnt/data/zkliu/llava_evaluation
MODEL_PATH=/mnt/data/zkliu/llava_for_okvqa/llava_t3_v1
MODEL_NAME=llava_t3_v1_f
DATA_DIR=/mnt/data/zkliu/llava_evaluation

# bash finetune_v0.sh

# aokvqa
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/aokvqa/aokvqa_val_full.json \
    --image-folder /mnt/data/zkliu/datasets/coco/images \
    --answers-file $DATA_DIR/aokvqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 &

# textvqa
CUDA_VISIBLE_DEVICES=2 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/textvqa/textvqa_full.json \
    --image-folder $DATA_DIR/textvqa/train_images \
    --answers-file $DATA_DIR/textvqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 &

# vizwiz 
CUDA_VISIBLE_DEVICES=3 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/vizwiz/vizwiz_val_full.json \
    --image-folder $DATA_DIR/vizwiz/val \
    --answers-file $DATA_DIR/vizwiz/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 &

CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/okvqa/okvqa_test_full.json \
    --image-folder /mnt/data/zkliu/datasets/coco/images \
    --answers-file $DATA_DIR/okvqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_chatgpt.py --result_file $result_dir/aokvqa/answers/$MODEL_NAME.jsonl --annotation_file /home/zkliu/vlm_examin/aokvqa_data/aokvqa_v1p0_val.json --output_file $result_dir/aokvqa/results/$MODEL_NAME.jsonl --openaikey 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 --anskey direct_answers &

python llava/eval/eval_chatgpt.py --result_file $result_dir/okvqa/answers/$MODEL_NAME.jsonl --annotation_file /home/zkliu/vlm_examin/okvqa_data/okvqa_val_eval.json --output_file $result_dir/okvqa/results/$MODEL_NAME.jsonl --openaikey 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 --anskey answer &

python llava/eval/eval_chatgpt.py --result_file $result_dir/textvqa/answers/$MODEL_NAME.jsonl --annotation_file $result_dir/textvqa/TextVQA_0.5.1_val.json --output_file $result_dir/textvqa/results/$MODEL_NAME.jsonl --openaikey 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 --anskey answers &

python llava/eval/eval_chatgpt.py --result_file $result_dir/vizwiz/answers/$MODEL_NAME.jsonl --annotation_file $result_dir/vizwiz/val.json --output_file $result_dir/vizwiz/results/$MODEL_NAME.jsonl --openaikey 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 --anskey answers &

# python llava/eval/eval_chatgpt.py --result_file $result_dir/aokvqa/answers/$MODEL_NAME.jsonl --annotation_file /home/zkliu/vlm_examin/aokvqa_data/aokvqa_v1p0_val.json --output_file $result_dir/aokvqa/results/$MODEL_NAME.jsonl --openaikey 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 --anskey direct_answers --eval_only

# python llava/eval/eval_chatgpt.py --result_file $result_dir/okvqa/answers/$MODEL_NAME.jsonl --annotation_file /home/zkliu/vlm_examin/okvqa_data/okvqa_val_eval.json --output_file $result_dir/okvqa/results/$MODEL_NAME.jsonl --openaikey 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 --anskey answer --eval_only
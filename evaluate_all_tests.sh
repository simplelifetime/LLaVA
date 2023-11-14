# MODEL_PATH=/mnt/data/zkliu/hf_models/llava-v1.5-7b
MODEL_NAME=llava_t5_v7
DATA_DIR=/mnt/data/zkliu/llava_evaluation

# echo "evaluating okvqa lemmatized"
# python -m llava.eval.eval_okvqa \
#     --annotation-file $DATA_DIR/okvqa/okvqa_val_eval.json \
#     --result-file $DATA_DIR/okvqa/answers/${MODEL_NAME}.jsonl \
#     --lemmatize


# echo "evaluating okvqa"
# python -m llava.eval.eval_okvqa \
#     --annotation-file $DATA_DIR/okvqa/okvqa_val_eval.json \
#     --result-file $DATA_DIR/okvqa/answers/${MODEL_NAME}.jsonl 


# echo "evaluating textvqa"
# python -m llava.eval.eval_textvqa \
#     --annotation-file $DATA_DIR/textvqa/TextVQA_0.5.1_val.json \
#     --result-file $DATA_DIR/textvqa/answers/${MODEL_NAME}.jsonl


# echo "evaluating vizwiz"
# python -m llava.eval.eval_okvqa \
#     --annotation-file $DATA_DIR/vizwiz/val.json \
#     --result-file $DATA_DIR/vizwiz/val_answers/${MODEL_NAME}.jsonl \
#     --anskey answers


# echo "evaluating aokvqa"
# python -m llava.eval.eval_okvqa \
#     --annotation-file /home/zkliu/vlm_examin/aokvqa_data/aokvqa_v1p0_val.json \
#     --result-file $DATA_DIR/aokvqa/val_answers/${MODEL_NAME}.jsonl \
#     --anskey direct_answers


echo "evaluating gqa"
python scripts/convert_gqa_for_eval.py --src $DATA_DIR/gqa/answers/$MODEL_NAME.jsonl --dst $DATA_DIR/gqa/data/testdev_balanced_predictions.json

cd $DATA_DIR/gqa/data
python eval.py --tier testdev_balanced


# echo "evaluating MME"
# cd ${DATA_DIR}/MME

# python convert_answer_to_mme.py --experiment ${MODEL_NAME}

# cd eval_tool

# python calculation.py --results_dir answers/${MODEL_NAME}

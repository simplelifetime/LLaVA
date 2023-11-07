result_dir=/mnt/data/zkliu/llava_evaluation

python llava/eval/eval_chatgpt.py --result_file $result_dir/aokvqa/answers/llava-v1.0-7b_v_f.jsonl --annotation_file /home/zkliu/vlm_examin/aokvqa_data/aokvqa_v1p0_val.json --output_file $result_dir/aokvqa/results/llava-v1.0-7b_v_f.jsonl --openaikey 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 --anskey direct_answers

# python llava/eval/eval_chatgpt.py --result_file $result_dir/aokvqa/answers/llava-v1.5-7b_v_f.jsonl --annotation_file /home/zkliu/vlm_examin/aokvqa_data/aokvqa_v1p0_val.json --output_file $result_dir/aokvqa/results/llava-v1.5-7b_v_f.jsonl --openaikey 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 --anskey direct_answers --eval_only

# python llava/eval/eval_chatgpt.py --result_file $result_dir/textvqa/answers/llava-v1.5-7b_f.jsonl --annotation_file $result_dir/textvqa/TextVQA_0.5.1_val.json --output_file $result_dir/textvqa/results/llava-v1.5-7b_f.jsonl --openaikey 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 --anskey answers --eval_only

# python llava/eval/eval_chatgpt.py --result_file $result_dir/textvqa/answers/llava-v1.0-7b_f.jsonl --annotation_file $result_dir/textvqa/TextVQA_0.5.1_val.json --output_file $result_dir/textvqa/results/llava-v1.0-7b_f.jsonl --openaikey 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 --anskey answers --eval_only

# python llava/eval/eval_chatgpt.py --result_file $result_dir/vizwiz/answers/llava-v1.0-7b_f.jsonl --annotation_file $result_dir/vizwiz/val.json --output_file $result_dir/vizwiz/results/llava-v1.0-7b_f.jsonl --openaikey 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 --anskey answers --eval_only

# python llava/eval/eval_chatgpt.py --result_file $result_dir/vizwiz/answers/llava-v1.5-7b_f.jsonl --annotation_file $result_dir/vizwiz/val.json --output_file $result_dir/vizwiz/results/llava-v1.5-7b_f.jsonl --openaikey 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 --anskey answers --eval_only

# bash scripts/v1_5/eval/new_scripts/eval_chatgpt.sh
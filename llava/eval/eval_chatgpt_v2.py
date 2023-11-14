import os
import openai
import time
import json
from tqdm import tqdm
import re
from collections import Counter
import random
import argparse
import IPython
# 账户的api调用密钥
# openai.organization = 'org-J0hV1SZHisb0USJYJNynlHRn' # 需要在网页端登录账号后在个人设置获取，具体操作可以参考前面使用方法部分。
# openai.api_key = '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'
# openai.api_key = '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'
openai.api_key = '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'
openai.api_base = 'https://0aa60511.cloudflare-test-96v.pages.dev/v1'
import math
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_API_RETRY = 1000
REQ_TIME_GAP = 2

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_eval(prompts, max_tokens = 128, model = 'text-davinci-003'):
    logging.basicConfig(level=logging.INFO)
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo-0613",
                messages = prompts,
                max_tokens = max_tokens,
                request_timeout = 10
            )
            content = response["choices"][0]["message"]
            logger.info(content)
            return content
        except Exception as e:
            logger.error(e)
            time.sleep(min(5*(i+1), 100))
    logger.error(f"Failed after {MAX_API_RETRY} retries.")
    return "error"

prompt =   '''You are an examiner who can judge whether a student's answer matches the correct answers. Next, I will provide you with one answer or several correct answers in the form of a list and a student's answer. Please judge whether the student's answer matches one of the 10 correct answers. If it matches, please output the correct answer directly (must be an element in the list, if it matches multiple correct answers, please output the most frequent occurrence in the list); if not, please output <NAN> directly. Do NOT output anything else!\n
correct answers: {gold}\n
student answer: {predict}'''

                 
def generate_prompt(sample):
    gold = sample['gold']
    predict = sample['predict']
    question = sample['question']
    input = dict(question=question, predict=predict.split('</s>')[0], gold=gold)
    input_text = prompt.format_map(input)
    return input_text

def json_load(f):
    res = []
    with open(f, 'r') as file:
    # Iterate over each line in the file
        for line in file:
            # Parse the line as JSON
            json_data = json.loads(line)
            res.append(json_data)   
    return res

def extract_ans(string):
    pattern = r'<(\w+)>|\((\w+)\)|\[(\w+)\]'
    match = re.search(pattern, string)
    if match:
        result = match.group(1) or match.group(2) or match.group(3)
        return result
    else:
        return None
    
def main(args):
    res = json_load(args.result_file)
    annotations = json.load(open(args.annotation_file, 'r'))
    
    if args.eval_only:
        outputs = json.load(open(args.output_file, 'r'))
        
    if isinstance(annotations, dict):
        annotations = annotations['data']
    response_list = []
    sum = 0
    
    res_chunk = get_chunk(res, args.num_chunks, args.chunk_idx)
    annotations_chunk = get_chunk(annotations, args.num_chunks, args.chunk_idx)
    
    for idx, r in tqdm(enumerate(res_chunk)):
        gold_answer = annotations_chunk[idx][args.anskey]
        if isinstance(gold_answer[0], dict):
            gold_answer = [g['answer'] for g in gold_answer]
        
        if args.eval_only:
            gpt_result = outputs[idx]['gpt_text']
            if extract_ans(gpt_result):
                gpt_result = extract_ans(gpt_result)
            gold_answer_dict = Counter(gold_answer)
            if gpt_result in gold_answer_dict.keys():
                v = gold_answer_dict[gpt_result]
                sum += min(v / 3.0, 1.0)
                
            continue
        
        pred_answer = r['text'].lower()
        input_prompt = prompt.format_map(dict(gold=gold_answer, predict=pred_answer))
        input_dict = [{"role":"user", "content":input_prompt}]
        output = get_eval(input_dict)['content'].strip()
        r['gpt_text'] = output
        response_list.append(r)
        
    if args.eval_only:
        print(f'GPT Accuracy: {sum / len(res)}')
    else:
        json.dump(response_list, open(args.output_file + f"{args.chunk_idx}.json", 'w'), indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, default="facebook/opt-350m")
    parser.add_argument("--annotation_file", type=str, default="facebook/opt-350m")
    parser.add_argument("--output_file", type=str, default="facebook/opt-350m")
    parser.add_argument("--anskey", type=str, default="answer")
    parser.add_argument("--eval_only", action='store_true')
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()
    main(args)
    
    
# python eval_chatgpt.py --result_file aokvqa/llava-v1.0-7b_v_f.jsonl --annotation_file aokvqa/aokvqa_v1p0_val.json --output_file aokvqa/results/llava-v1.0-7b_v_f.jsonl
# python eval_chatgpt.py --result_file aokvqa/llava-v1.5-7b_v_f.jsonl --annotation_file aokvqa/aokvqa_v1p0_val.json --output_file aokvqa/results/llava-v1.5-7b_v_f.jsonl
import os
import argparse
import json
import re

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    parser.add_argument('--lemmatize', action='store_true')
    return parser.parse_args()

def read_vocab(file):
    answer_vocab = []
    with open(file, 'r') as f:
        for line in f:
            answer_vocab.append(line.strip())
    return answer_vocab

def _lemmatize(answers):
    import spacy

    _lemmatizer = spacy.load("en_core_web_sm")

    def apply(answer):
        doc =_lemmatizer(answer)

        words = []
        for token in doc:
            if token.pos_ in ["NOUN", "VERB"]:
                words.append(token.lemma_)
            else:
                words.append(token.text)
        answer = " ".join(words)

        return answer

    return [apply(answer) for answer in answers]

def eval_single(annotation_file, result_file, args):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    annotations = json.load(open(annotation_file))
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for idx, result in enumerate(results):
        annotation = annotations[idx]
        pred_list.append({
            "pred_answer": result['text'].lower(),
            "gt_answers": annotation['answer'],
        })

    evaluator = TextVQAAccuracyEvaluator()
    if args.lemmatize:
        pred_answers = [p['pred_answer'] for p in pred_list]
        pred_answers = _lemmatize(pred_answers)
        for idx, p in enumerate(pred_list):
            p['pred_answer'] = pred_answers[idx]
        
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file, args)

    if args.result_dir is not None:
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith('.jsonl'):
                print(f'Skipping {result_file}')
                continue
            eval_single(args.annotation_file, os.path.join(args.result_dir, result_file))

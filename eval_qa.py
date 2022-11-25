import argparse
import os
import json
import numpy as np

import termcolor
import torch

from nltk import word_tokenize

from eval_utils import *
from utils import set_seed, get_injected_tokens, get_utter_len, tokenize, str2bool, get_loader
from utils import is_primary, setup_multi_gpu, gather, remove_eos_token, ids2text
from tqdm import tqdm


def pprint(i, passage, query, answer, answer_gen, injections=None):
    print(termcolor.colored(str(i), 'blue'))
    print(termcolor.colored(passage, 'yellow'))
    print(termcolor.colored(query, 'red'))
    print(termcolor.colored(answer, 'green'))
    if injections is not None:
        print(termcolor.colored(injections, 'cyan'))
    print(answer_gen)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    add_params(ap)
    ap.add_argument('-d', '--domain', required=False, type=str, default='1')
    ap.add_argument('-td', '--target_domain', type=str, default=None)
    args = ap.parse_args()
    args.exp = 'qa'

    setup_multi_gpu(args)
    set_seed(args.seed)

    model, tokenizer, config = load_model(args)
    if config is not None:
        args.domain = config.domain

    output_dir = f'qa/log_dir/results/{args.domain}'
    os.makedirs(output_dir, exist_ok=True)

    is_stg = False
    if config is not None:
        is_stg = config.is_stg

    def _get_result_fpath(device_id=None):
        return get_result_fpath(args, output_dir, device_id=device_id)

    output_file = _get_result_fpath(args.device_id)

    print(f"output_file:{output_file} , use_eos: {args.use_eos}")
    loader = get_loader(tokenizer, args, mode='test').loader

    iter = loader
    if is_primary():
        iter = tqdm(iter)

    passages, queries, gt_answers, gen_answers = [], [], [], []
    acts = []

    for i, batch in enumerate(iter):
        with torch.no_grad():
            _, _, (_, texts) = batch
            sample, injection_acts = generate_sample(model, tokenizer, batch, args, is_stg=is_stg)

            if is_stg:
                acts.append(injection_acts)

            passages.append(texts['passage'][0])
            queries.append(texts['query'][0])
            gt_answers.append(texts['answer'][0])
            gen_answers.append(sample)

    scores = []
    results = []

    for i in range(len(passages)):
        passage = passages[i]
        query = queries[i]
        gt_answer = remove_eos_token(gt_answers[i], tokenizer.eos_token)
        gen_answer = ids2text(gen_answers[i], tokenizer, rm_remain=not args.use_eos)

        gt_answer = ' '.join(word_tokenize(gt_answer))

        result = [passage, query, gt_answer, gen_answer]

        injections = None
        if is_stg and acts[i] is not None:
            _acts = acts[i]
            injections = get_injected_tokens(tokenizer, _acts, config.vocab_size)
            injections = '[' + injections.strip() + ']'
            result.append(injections.strip())

        # if is_primary() and i % 100 == 0:
        #     pprint(i, passage, query, gt_answer, gen_answer, injections=injections)

        results.append(result)
        score = get_scores(gt_answer, gen_answer, 'qa')
        score_tensor = torch.DoubleTensor([score]).cuda()
        scores.append(score_tensor)

    json.dump(results, open(output_file, 'w'), indent=2)

    scores = torch.cat(scores)
    _scores = np.mean(scores.cpu().numpy(), axis=0)

    # print('############################')
    # print(f'device_id:{args.device_id}')
    # print('%s: %s' % ('BLEU', _scores[0]))
    # print('%s: %s' % ('RL', _scores[1]))
    # print('############################')

    torch.distributed.barrier()
    tot_list = gather(scores)
    if is_primary():
        scores = torch.cat(tot_list)
        print("gatherd:", scores.shape)
        mean_scores = np.mean(scores.cpu().numpy(), axis=0)
        result_path = _get_result_fpath(None)

        score_dict = get_info(args)
        score_dict['bleu'] = mean_scores[0]
        score_dict['rl'] = mean_scores[1]

        if config is not None:
            score_dict['fixed_inj_prob'] = config.fixed_inj_prob

        print(result_path)
        print(score_dict)

        overall_result = []
        for _id in range(args.world_size):
            fpath = _get_result_fpath(_id)
            with open(fpath, 'r') as fin:
                result_data = json.load(fin)
            overall_result.extend(result_data)
            os.remove(fpath)

        json.dump(overall_result, open(result_path, 'w'), indent=2)









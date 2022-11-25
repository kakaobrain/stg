import argparse
import os
import json
import numpy as np

import termcolor
import torch

from nltk import word_tokenize
from utils import set_seed, get_injected_tokens, get_utter_len, tokenize, get_loader
from utils import is_primary, setup_multi_gpu, gather, remove_eos_token, ids2text
from tqdm import tqdm
from eval_utils import *


def pprint(article, gt, summ, injections=None):
    print(termcolor.colored(article, 'yellow'))
    print(termcolor.colored(gt, 'green'))
    if injections is not None:
        print(termcolor.colored(injections, 'red'))
    print(termcolor.colored(summ, 'cyan'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    add_params(ap)
    ap.add_argument('-d', '--domain', type=str, default='CNN', choices=['CNN', 'CNN01', 'CNN05', 'CNN2', 'CNN003', 'CNN001'])
    ap.add_argument('--n_test', type=int, default=None)
    ap.add_argument('--gen_seed', type=int, default=None)
    args = ap.parse_args()
    args.exp = 'summ'
    args.preseqlen = 0

    setup_multi_gpu(args)
    set_seed(args.seed)

    output_dir = f'summ/log_dir/results'
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer, config = load_model(args)

    is_stg = False
    if config is not None:
        is_stg = config.is_stg

    prefix = ''
    if args.n_test is not None:
        prefix = str(args.n_test)
    if args.gen_seed:
        if prefix:
            prefix += '_'
        prefix += f'genSEED-{args.gen_seed}'
        set_seed(args.gen_seed)

    def _get_result_fpath(device_id=None):
        return get_result_fpath(args, output_dir, device_id=device_id, prefix=prefix)

    output_file = _get_result_fpath(args.device_id)

    print(f"output_file:{output_file} , use_eos: {args.use_eos}")
    loader = get_loader(tokenizer, args, mode='test').loader


    iter = loader
    if is_primary():
        iter = tqdm(iter)

    passages, gt_answers, gen_answers = [], [], []
    acts = []

    for i, batch in enumerate(iter):
        if args.n_test is not None and i == args.n_test:
            break
        with torch.no_grad():
            passage, _, (gt_str, _) = batch
            sample, injection_acts = generate_sample(model, tokenizer, batch, args, is_stg=is_stg)

            passages.append(passage.squeeze(0))
            gt_answers.append(gt_str[0])
            gen_answers.append(sample)
            if is_stg or args.inj_scheme is not None:
                acts.append(injection_acts)

    scores = []
    results = []
    for i in range(len(passages)):
        passage = passages[i]
        gt_answer = remove_eos_token(gt_answers[i], tokenizer.eos_token)
        gen_answer = ids2text(gen_answers[i], tokenizer)

        gt_answer = ' '.join(word_tokenize(gt_answer))
        article = tokenizer.decode(passage.squeeze(0)).strip().lower()

        result = [article, gt_answer, gen_answer]
        injections = None
        if (is_stg or args.inj_scheme is not None) and acts[i] is not None:
            _acts = acts[i]
            injections = get_injected_tokens(tokenizer, _acts, config.vocab_size)
            injections = '[' + injections.strip() + ']'
            result.append(injections.strip())

        # if is_primary() and i % 100 == 0:
        #     pprint(article, gt_answer, gen_answer, injections=injections)

        results.append(result)
        score = get_scores(gt_answer, gen_answer, 'summ')
        score_tensor = torch.DoubleTensor([score]).cuda()
        scores.append(score_tensor)

    json.dump(results, open(output_file, 'w'), indent=2)

    scores = torch.cat(scores)
    _scores = np.mean(scores.cpu().numpy(), axis=0)

    # print('############################')
    # print(f'device_id:{args.device_id}')
    # print('%s: %s' % ('R1', _scores[0]))
    # print('%s: %s' % ('R2', _scores[1]))
    # print('%s: %s' % ('RL', _scores[2]))
    # print('############################')

    torch.distributed.barrier()
    tot_list = gather(scores)
    if is_primary():
        scores = torch.cat(tot_list)
        print("gatherd:", scores.shape)
        mean_scores = np.mean(scores.cpu().numpy(), axis=0)
        result_path = _get_result_fpath(None)

        score_dict = get_info(args)
        score_dict['r1'] = mean_scores[0]
        score_dict['r2'] = mean_scores[1]
        score_dict['rl'] = mean_scores[2]

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



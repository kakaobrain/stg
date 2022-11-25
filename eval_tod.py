import os
import torch
import argparse
import json
import termcolor
from utils import set_seed, get_injected_tokens, setup_multi_gpu, get_loader, is_primary, gather
from eval_qa import set_config, add_params
import modules
from tqdm import tqdm

from eval_utils import *


def do_eval(model, tokenizer, args, is_stg):
    output_file = _get_result_fpath(args.device_id)
    output_inj_file = None
    is_ensemble = is_stg or args.inj_scheme is not None
    if is_ensemble:
        output_inj_file = _get_result_fpath(args.device_id, postfix='inj')

    print(f"output_file:{output_file}")
    print(f"output_inj_file:{output_inj_file}")

    loader = get_loader(tokenizer, args, mode='test')
    iter = tqdm(range(len(loader.loader)))

    outputs = []
    output_injections = []
    ids = []


    for k in iter:
        with torch.no_grad():
            batch = loader.get_batch()
            _, _, aux_data = batch
            (_, _, _, _, data_id) = aux_data

            samples, injection_acts = generate_sample(model, tokenizer, batch, args, is_stg=is_stg, do_sample=True)
            utters, injections = [], []

            for i, utter in enumerate(samples):
                if is_ensemble:
                    injected_str = get_injected_tokens(tokenizer, injection_acts[i], config.vocab_size)
                    injections.append(injected_str)
                    # print(termcolor.colored('[' + injected_str.strip() + ']', 'red'))
                next_utterance = tokenizer.decode(utter)
                cl_idx = next_utterance.find(tokenizer.eos_token)
                next_utterance = next_utterance[:cl_idx].strip().lower()
                utters.append(next_utterance)

                # print(termcolor.colored(next_utterance, 'green'))
            outputs.append(utters)
            output_injections.append(injections)

            ids.append(data_id.cuda())

    json.dump(outputs, open(output_file, 'w'), indent=2)
    if is_ensemble:
        json.dump(output_injections, open(output_inj_file, 'w'), indent=2)

    ids = torch.cat(ids)
    tot_ids = gather(ids)

    if is_primary():
        ids = torch.cat(tot_ids)
        print("gatherd:", ids.shape)
        result_path = _get_result_fpath(None)
        result_inj_path = None
        if is_ensemble:
            result_inj_path = _get_result_fpath(None, postfix='inj')

        print(result_path, result_inj_path)

        overall_result = []
        overall_inj_result = []
        for _id in range(args.world_size):
            fpath = _get_result_fpath(_id)
            with open(fpath, 'r') as fin:
                result_data = json.load(fin)
            overall_result.extend(result_data)
            os.remove(fpath)
            if is_ensemble:
                fpath = _get_result_fpath(_id, postfix='inj')
                with open(fpath, 'r') as fin:
                    result_data = json.load(fin)
                overall_inj_result.extend(result_data)
                os.remove(fpath)

        ids = ids.detach().cpu().numpy()

        result_data = {}
        for i, data_id in enumerate(ids):
            result_data[int(data_id)] = overall_result[i]

        if is_ensemble:
            result_inj_data = {}
            for i, data_id in enumerate(ids):
                result_inj_data[int(data_id)] = overall_inj_result[i]

        json.dump(result_data, open(result_path, 'w'), indent=2)
        if is_ensemble:
            json.dump(result_inj_data, open(result_inj_path, 'w'), indent=2)

        bleu, err = eval_tod(args.domain, result_path, verbose=1)
        score_dict = {
            'seed': args.seed,
            'scheme': args.scheme,
            'PLM_temperature': args.temp,
            'Calib_temperature': args.temp2,
            'topk': args.top_k,
            'n_sample': args.n_sample,
            'BLEU': "{0:.2f}".format(bleu * 100),
            'ERR': "{0:.2f}".format(err)
        }
        if args.inj_scheme is not None:
            score_dict['inj_scheme'] = args.inj_scheme

        print(result_path)
        print(score_dict)



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    add_params(ap)
    ap.add_argument('-d', '--domain', default=None, type=str, required=False)
    ap.add_argument('-td', '--target_domain', type=str, default=None)
    ap.add_argument('--inj_test', type=str2bool, default=False)
    ap.add_argument('--seed_test', type=str2bool, default=False)
    args = ap.parse_args()
    args.exp = 'tod'
    args.use_train_score = False

    setup_multi_gpu(args)


    model, tokenizer, config = load_model(args)
    if config is not None:
        args.domain = config.domain

    output_dir = f'tod/log_dir/results/{args.domain}'
    os.makedirs(output_dir, exist_ok=True)

    is_stg = False
    if config is not None:
        is_stg = config.is_stg

    def _get_result_fpath(device_id=None, postfix=None):
        prefix = None
        if args.mode == 'rl':
            prefix = f'genSEED-{args.seed}'
        return get_result_fpath(args, output_dir, device_id=device_id, postfix=postfix, prefix=prefix)

    def do_test(args):
        if args.inj_test:
            inj_schemes = ['max', 'mix']
            for scheme in inj_schemes:
                config.inj_scheme = scheme
                args.inj_scheme = scheme
                do_eval(model, tokenizer, args, is_stg)
        else:
            do_eval(model, tokenizer, args, is_stg)

    if args.seed_test:
        seeds = [9, 99, 999, 9999, 99999]
        for seed in seeds:
            set_seed(seed)
            args.seed = seed
            config.seed = seed
            do_test(args)
    else:
        set_seed(args.seed)
        do_test(args)

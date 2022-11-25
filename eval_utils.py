import os
import torch
import numpy as np

import transformers
from utils import str2bool, sequence_mask, load_dm, build_ddp_model, get_plm_path, clip_and_log, ids2text, get_utter_len, tokenize
from qa.src.Evaluation.bleu.bleu import Bleu
from qa.src.Evaluation.rouge.rouge import Rouge
from rouge_score import rouge_scorer
from tod.fswoz.evaluator import evaluate
from tod.fswoz.utils.loader.DataReader import DataReader
from tod.fswoz.utils.loader.GentScorer import GentScorer


from config import Config_PG

transformers.logging.set_verbosity(transformers.logging.ERROR)

bleu_scorer = Bleu()
qa_rouge_scorer = Rouge()
summ_rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def add_params(parser):
    parser.add_argument('--exp', default='qa', choices=['qa', 'summ'])
    parser.add_argument('-m', '--mode', required=False, type=str, default='ft', choices=['ft', 'rl', 'pt'])
    parser.add_argument('-cp', '--cp_path', required=False, type=str, default=None)
    parser.add_argument('-n', '--n_sample', type=int, default=3)
    parser.add_argument('-cid', '--cuda_id', default=0, type=int)

    parser.add_argument('--world_size', type=int)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--backend', type=str, default='nccl')

    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--scheme", type=str, default='sample')
    parser.add_argument("--score", type=str, default='sample', choices=['sample', 'oracle', 'beam', 'greedy'])
    parser.add_argument("--use_eos", type=str2bool, default=True)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=int, default=0.9)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--temp2", type=float, default=1.0)
    parser.add_argument('--inj_scheme', type=str, default=None, choices=[None, 'random', 'max', 'mix'])
    parser.add_argument('--fixed_inj_prob', type=float, default=None)


def set_config(config, args):
    config.top_k = args.top_k
    config.top_p = args.top_p
    config.temperature4plm = args.temp
    config.temperature4calib = args.temp2
    config.scheme = args.scheme
    config.inj_scheme = args.inj_scheme
    config.fixed_inj_prob = args.fixed_inj_prob
    if 'greedy' in args.score:
        config.scheme = args.score


def get_info(args):
    info = {
        'n_sample': args.n_sample,
        'scheme': args.scheme,
        'PLM_temperature': args.temp,
        'Calib_temperature': args.temp2,
        'topk': args.top_k,
        'scoring': args.score
    }
    return info


def calc_rouge_score(gt_text, gen_text):
    gts, res = {'0': [gt_text]}, {'0': [gen_text]}
    score, _ = qa_rouge_scorer.compute_score(gts, res)
    return score


def select_seq_with_oracle(gens, gt_text, tokenizer):
    best_score = -np.inf
    best_idx = None
    result = None
    for i, gen in enumerate(gens):
        gen_text = ids2text(gen, tokenizer)
        score = calc_rouge_score(gt_text, gen_text)
        if score > best_score:
            best_score = score
            best_idx = i
            result = gen_text
    return result, best_idx


def eval_tod(domain, result_path, verbose=0):
    return evaluate(domain, result_path, DataReader, GentScorer, verbose=verbose)


def calc_score(lm_score, masks, seq_len=None):
    scores = clip_and_log(lm_score) * masks.float()
    scores = scores.sum(-1)
    if seq_len is not None:
        scores /= seq_len

    return scores


def sample_from_plm(model, input_context, utter_len):
    input_context = input_context.cuda()
    scores, gens, seq_len = model.plm_gen(input_context, utter_length=utter_len)
    maxlen = gens.shape[-1]
    masks = sequence_mask(seq_len + 1, maxlen=maxlen, device=input_context.device)
    scores = calc_score(scores, masks, seq_len)
    return gens, scores, masks


def beam_search(model, input_context, num_beams):
    beam_output, scores, injection_masks = model.beam_gen(
        input_context.cuda(),
        max_length=1024,
        num_beams=num_beams,
        early_stopping=True,
        do_sample=False,
    )
    idx = input_context.shape[-1]
    beam_output = beam_output[:, idx:]
    injection_masks = injection_masks[:beam_output.shape[-1]]

    acts = torch.ones_like(beam_output) * model.tokenizer.vocab_size
    acts = (torch.logical_not(injection_masks) * acts) + (injection_masks * beam_output)

    return beam_output[0], acts.long()[0]


def sample_from_calib_lm(sampler, batch, utter_len, is_stg, args):
    data = sampler.sample(batch=batch, utter_length=utter_len, n_rounds=args.n_sample,
                              is_valid=True)

    gens = data['sequences']

    lm_probs = data['obs'].squeeze(-1)
    maxlen = lm_probs.shape[1]
    seq_len = data['seq_lengths']
    masks = sequence_mask(seq_len + 1, maxlen=maxlen, device=lm_probs.device)

    if is_stg:
        lm_score = lm_probs[:, :, 1]
        inj_probs = lm_probs[:, :, 0]
        scores = lm_score * inj_probs
    else:
        scores = lm_probs

    sti_acts = None
    if is_stg or args.inj_scheme is not None:
        sti_acts = data['acts']

    scores = calc_score(scores, masks, seq_len)

    return gens, scores, masks, sti_acts


def get_result_fpath(args, output_dir, device_id=None, prefix='', postfix=''):
    fname = ''
    if prefix:
        fname = prefix + '_'

    if args.mode in ['ft', 'pt']:
        fname += args.mode

        domain = args.domain
        try:
            if args.target_domain is not None:
                domain = args.target_domain
        except:
            pass

        if args.score == 'beam':
            fname += f'-{domain}-{args.seed}_temp-{args.temp}'
        else:
            fname += f'-{domain}-{args.seed}_tk-{args.top_k}_temp-{args.temp}'

        if args.use_eos:
            fname += '_EOS'
    else:
        directory, filename = os.path.split(args.cp_path)
        fname += f'{os.path.splitext(filename)[0]}'
        if args.score == 'beam':
            fname += f'_temp-{args.temp}_temp2-{args.temp2}'
        else:
            fname += f'_tk-{args.top_k}_temp-{args.temp}_temp2-{args.temp2}_{args.scheme}'

        if args.inj_scheme is not None:
            fname += f'_inj-{args.inj_scheme}'

        if args.fixed_inj_prob is not None:
            fname += f'_FIP-{args.fixed_inj_prob}'

    if postfix:
        fname += '_' + postfix

    fname += f'_scoring-{args.score}'
    if args.score in ['sample', 'beam', 'oracle']:
        fname += f'-{args.n_sample}'
    fpath = os.path.join(output_dir, fname)

    if device_id is not None:
        fpath += f'_{device_id}'

    fpath += '.json'

    return fpath


def load_model(args, use_sampler=True):
    import modules
    if args.mode == 'ft':
        config = Config_PG()
        plm_path = get_plm_path(args)
        config.mode = args.mode
        config.exp = args.exp
        config.domain = args.domain
        config.ft_lm_path = plm_path
        config.adapter_type = None
        model = modules.PLM_wrapper(config)
    else:
        model, config = load_dm(args.cp_path, args=args)
        args.use_eos = config.use_eos

    model = build_ddp_model(model, args)
    set_config(config, args)
    tokenizer = model.module.tokenizer
    if use_sampler:
        model = modules.DialogueSampler(model, config, mode='test')

    return model, tokenizer, config


def max_utter_len(exp):
    if exp in ['qa']:
        max_len = 95
    elif exp == 'summ':
        max_len = 103
    elif exp == 'tod':
        max_len = 80

    return max_len


def generate_sample(model, tokenizer, batch, args, is_stg=False, do_sample=False):
    input_context, utter_len, aux_data = batch
    if args.exp == 'qa':
        (_, texts) = aux_data
        gt = texts['answer'][0]
    elif args.exp == 'summ':
        (gt_str, _) = aux_data
        gt = gt_str[0]
    elif args.exp == 'tod':
        (_, _, ref_sent, _, _) = aux_data
        gt = ref_sent
    else:
        pass

    utter_len = max_utter_len(args.exp)
    utter_len = get_utter_len(input_context.shape[1], utter_len, 0)

    best_one = None
    injection_acts = None
    sti_acts = None
    if args.score == 'beam':
        best_one, injection_acts = beam_search(model.dm.module, input_context, num_beams=args.n_sample)
    else:
        if args.score == 'sample' or args.score == 'oracle':
            input_context = input_context.repeat(args.n_sample, 1)

        if args.mode == 'ft':
            gens, scores, masks = sample_from_plm(model.dm.module, input_context, utter_len)
        else:
            batch = input_context, utter_len, aux_data
            gens, scores, masks, sti_acts = sample_from_calib_lm(model, batch, utter_len, is_stg, args)

        if do_sample:
            return gens, sti_acts
        else:
            if args.score == 'oracle':
                assert len(gens) == args.n_sample
                gt_text = tokenize(gt)
                _, best_idx = select_seq_with_oracle(gens, gt_text, tokenizer)
            elif args.score == 'sample':
                assert len(gens) == args.n_sample
                best_idx = torch.argmax(scores)
            elif 'greedy' in args.score:
                best_idx = 0

            best_one = gens[best_idx]

            if is_stg or args.inj_scheme is not None:
                injection_acts = sti_acts[best_idx]

    return best_one, injection_acts


def get_scores(gt_answer, gen_answer, exp):
    if exp == 'qa':
        gts, res = {'0': [gt_answer]}, {'0': [gen_answer]}
        bleu_scores, _ = bleu_scorer.compute_score(gts, res)
        rouge_score, _ = qa_rouge_scorer.compute_score(gts, res)
        return [np.mean(bleu_scores), rouge_score]
    elif exp == 'summ':
        score = summ_rouge_scorer.score(gt_answer, gen_answer)
        return [score['rouge1'].fmeasure, score['rouge2'].fmeasure, score['rougeL'].fmeasure]
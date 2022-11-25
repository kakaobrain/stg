import torch
import argparse
from eval_utils import *
import utils

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    add_params(ap)
    ap.add_argument('-d', '--domain', required=False, type=str, default='05')
    ap.add_argument('-td', '--target_domain', type=str, default=None)
    args = ap.parse_args()
    args.n_sample = 10
    args.exp = 'qa'
    args.cp_path = None
    utils.set_seed(9)

    # args.cp_path = 'qa/log_dir/05/BEST-0.383_qa-05_SEED-9_pg-lstm_gpt2-medium_dim-256_bs-16_nr-1_lr-0.0001_nlayer-2_algo-ppo_LRS-linear_UCE_EOS_arch-old_scheme-cat_temp-1.0_interm_NAP.checkpoint'
    # args.cp_path = 'qa/log_dir/05/BEST-0.362_MLE_FTI_qa-05_SEED-9_pg-lstm_gpt2-medium_dim-256_bs-16_nr-1_lr-5e-05_nlayer-2_LRS-linear_UCE_EOS_arch-old_NAP.checkpoint'
    # args.cp_path = 'qa/log_dir/05/BEST-0.346_FTI_qa-05_SEED-9_pg-lstm_gpt2-medium_dim-256_bs-16_nr-1_lr-5e-05_nlayer-2_algo-ac_PRE-1_LRS-linear_UCE_EOS_arch-old_NAP.checkpoin'

    # this example does not contain the 0.5% subset data
    passage_text = 'three types of conflicts are : 1. intrapersonal conflicts , 2. interpersonal conflicts and 3. unconscious conflicts . the word conflict has been derived from a latin word ‘ conflicts ’ which means ‘ strike two things at the same time ’ . conflict is an opposition or a tug-of-war between contradictory impulses . according to colman ‘ a conflict is the anticipated frustration entailed in the choice of either alternative ’ .'
    query_text = 'conflict definition psychology'

    if args.cp_path is None:
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
        args.domain = config.domain

    model.eval().cuda()

    tokenizer = model.tokenizer
    eos_token_id = [tokenizer.eos_token_id]
    sep_token_id = tokenizer.encode(' ?')
    input_ids = tokenizer.encode(passage_text) + eos_token_id + sep_token_id
    input_ids += tokenizer.encode(query_text) + eos_token_id + sep_token_id
    print(tokenizer.decode(input_ids, skip_special_tokens=False))
    input_ids = torch.LongTensor(input_ids).repeat(args.n_sample, 1).cuda()

    utter_len = max_utter_len(args.exp)
    inj_items = []
    if args.cp_path is None:
        scores, sequences, seq_length = model.plm_gen(input_ids, utter_length=utter_len)
        masks = utils.sequence_mask(seq_length + 1, maxlen=scores.shape[-1], device=scores.device)
        scores = calc_score(scores, masks, seq_length)
    else:
        sequences, prev_state, obs, injection_masks, p_plm_tokens, seq_length, ext_values, extra_stats_dict = \
            model(input_ids, utter_length=utter_len)


        lm_probs = obs.squeeze(-1)
        maxlen = lm_probs.shape[1]
        masks = utils.sequence_mask(seq_length + 1, maxlen=maxlen, device=lm_probs.device)

        if config.is_stg:
            lm_score = lm_probs[:, :, 1]
            inj_probs = lm_probs[:, :, 0]
            scores = lm_score * inj_probs
        else:
            scores = lm_probs

        scores = calc_score(scores, masks, seq_length)

        init_len = input_ids.shape[-1]
        sequences = sequences[:, init_len:]

        if config.is_stg:
            acts = torch.ones_like(sequences) * tokenizer.vocab_size
            acts = (torch.logical_not(injection_masks) * acts) + (injection_masks * sequences)
            for act in acts:
                injections = utils.get_injected_tokens(tokenizer, act, config.vocab_size)
                injections = '[' + injections.strip() + ']'
                inj_items.append(injections.strip())

    output_ids = sequences.detach().cpu().numpy()
    for i, _output_ids in enumerate(output_ids):
        gen_answer = tokenizer.decode(_output_ids, skip_special_tokens=True)
        print(gen_answer)
        print('t\score:{}'.format(scores[i]))
        if config.is_stg:
            print(inj_items[i])
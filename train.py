import os
import torch
from torch.nn import functional as F
import numpy as np
import modules
import utils
from eval_utils import max_utter_len
from config import Config_PG
from termcolor import colored
from time import time
import argparse
from tqdm import tqdm
from transformers import AdamW
import random
import shutil
from torch.utils.tensorboard import SummaryWriter


def store_model(dm, cp_path):
    torch.save(dm.module.policy.state_dict(), cp_path)


def may_store_model(dm, avg_reward, best_reward, cp_path):
    if avg_reward is None:
        return best_reward, cp_path

    config = dm.module.config

    print("cur_score:{0:.3f}, best_score:{1:.3f}".format(avg_reward, best_reward))
    if best_reward < avg_reward:
        print("SCORED!!")
        cp_name = "{}.checkpoint".format(config.get_experiment_name())

        prev_best_score = best_reward
        prev_best_cp_name = "BEST-{0:.3f}_".format(prev_best_score) + cp_name
        prev_fpath = os.path.join(config.model_path, prev_best_cp_name)

        if os.path.exists(prev_fpath):
            os.remove(prev_fpath)

        cp_name = "BEST-{0:.3f}_".format(avg_reward) + cp_name
        cp_path = os.path.join(config.model_path, cp_name)

        best_reward = avg_reward

        store_model(dm, cp_path)


    # if best_reward < avg_reward:
    #     print("SCORED!!")
    #     prev_best_score = best_reward
    #     prev_best_cp_name = "best-{0:.3f}.checkpoint".format(prev_best_score)
    #     prev_fpath = os.path.join(root_dir, prev_best_cp_name)
    #     if ppl is not None:
    #         cp_name = '{0:}_ppl-{1:.2f}.checkpoint'.format(t, ppl)
    #     else:
    #         cp_name = '{0:}.checkpoint'.format(t)
    #
    #     if os.path.exists(prev_fpath):
    #         os.remove(prev_fpath)
    #
    #     cp_name = "best-{0:.3f}_".format(avg_reward) + cp_name
    #     cp_path = os.path.join(root_dir, cp_name)
    #
    #     best_reward = avg_reward
    #     store_model(dm, cp_path)
    # else:
    #     if ppl is not None:
    #         cp_name = '{0:}_ppl-{1:.2f}.checkpoint'.format(t, ppl)
    #     else:
    #         cp_name = '{0:}.checkpoint'.format(t)
    #     cp_path = os.path.join(root_dir, cp_name)
    #     store_model(dm, cp_path)

    return best_reward, cp_path


def get_perplexity(sampler, config, is_train_score=False, percentage=1):
    ppl = []
    dm = sampler.dm
    with torch.no_grad():
        if is_train_score:
            loader = sampler.train_valid_loader
        else:
            loader = sampler.valid_loader

        n_valid_sample = len(loader.loader)
        if percentage < 1:
            n_valid_sample = int(n_valid_sample * percentage)

        is_primary = utils.is_primary()

        iter = range(n_valid_sample)
        if is_primary:
            iter = tqdm(iter)

        loader.sampler.set_epoch(0)
        for _ in iter:
            batch = sampler.get_batch(1, loader)

            inputs, target, _, _ = parse_batch(batch, config.exp)

            inputs = inputs.cuda()
            target = target.cuda()

            if config.is_stg:
                _ppl = calc_sti_mle_loss(dm, inputs, target)
            else:
                _ppl = calc_mle_loss(dm, inputs, target, config)
            ppl.append(_ppl)
    return ppl


def validation(sampler, config, is_train_score=False, percentage=1):
    scores = []
    with torch.no_grad():
        if is_train_score:
            loader = sampler.train_valid_loader
        else:
            loader = sampler.valid_loader

        n_valid_sample = len(loader.loader)
        if percentage < 1:
            n_valid_sample = int(n_valid_sample * percentage)

        is_primary = utils.is_primary()

        iter = range(n_valid_sample)
        if is_primary:
            iter = tqdm(iter)

        loader.sampler.set_epoch(0)
        for _ in iter:
            batch = sampler.get_batch(1, loader)

            inputs, _, _, _ = parse_batch(batch, config.exp)
            utter_len = max_utter_len(config.exp)
            utter_len = utils.get_utter_len(inputs.shape[1], utter_len, 0)

            forward_data = sampler.sample(batch=batch, utter_length=utter_len, is_valid=True)
            scores.extend(forward_data['ext_rewards'])
            del forward_data

    return scores


def validate(sampler, config, writer=None, t=None, use_multi_gpu=True, percentage=1, check_ppl=None):
    if check_ppl is None:
        check_ppl = config.do_valid_ppl

    dm = sampler.dm
    dm.eval()
    is_primary = utils.is_primary() 
    avg_ppl = None
    if check_ppl:
        ppls = get_perplexity(sampler, config, percentage=percentage)
        avg_ppl = get_avg_score(ppls, use_multi_gpu)
        if writer and is_primary:
            writer.add_scalar('info/valid_ppl', avg_ppl, t)

    avg_score = 0
    if config.do_valid_gen:
        scores = validation(sampler, config, percentage=percentage)
        avg_score = get_avg_score(scores, use_multi_gpu)
        if writer and is_primary:
            writer.add_scalar('info/valid_score', avg_score, t)

    dm.train()
    return avg_score, avg_ppl


def validate_and_store_model(sampler, config, best_reward, cp_path,
                             writer=None, t=None):
    avg_score, avg_ppl = validate(sampler, config, writer=writer, t=t, check_ppl=False)
    if utils.is_primary():
        best_reward, cp_path = may_store_model(sampler.dm, avg_score, best_reward, cp_path)
    return best_reward, cp_path


def check_train_score(sampler, config, writer=None, t=None, use_multi_gpu=True, percentage=1):
    dm = sampler.dm
    dm.eval()

    is_primary = utils.is_primary()
    if config.do_valid_ppl:
        ppls = get_perplexity(sampler, config, is_train_score=True, percentage=percentage)
        avg_ppl = get_avg_score(ppls, use_multi_gpu)
        if writer and is_primary:
            writer.add_scalar('info/train_ppl', avg_ppl, t)

    if config.do_valid_gen:
        scores = validation(sampler, config, is_train_score=True, percentage=percentage)
        avg_reward = get_avg_score(scores, use_multi_gpu)
        if writer and is_primary:
            writer.add_scalar('info/train_score', avg_reward, t)
    dm.train()


def pretrain(config, sampler, optimizer):
    epochs = config.pretrain_epochs
    num_batch = config.pretrain_num_batch
    is_primary = utils.is_primary()

    if is_primary:
        print(f"Pretrain epochs: {epochs}, batch_size: {num_batch}")
        print(config.get_experiment_name())

    n_data = len(sampler.loader.loader)

    steps_per_epoch = int(n_data / num_batch)
    if n_data % num_batch > 0:
        steps_per_epoch += 1

    training_steps = int(steps_per_epoch * epochs)
    optimizer.zero_grad()

    for t in range(1, training_steps + 1):
        loss = torch.zeros(1)
        loss = loss.cuda()

        for i in range(num_batch):
            batch = sampler.get_batch(1)
            inputs, target, gt, utter_len = parse_batch(batch, config.exp)

            inputs = inputs.cuda()
            target = target.cuda()

            ml_loss = calc_mle_loss(dm, inputs, target, config)
            loss = ml_loss.mean()/num_batch
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if is_primary:
            t_loss = loss.detach().cpu().numpy()
            msg = "t-step:{0:}, loss: {1:.4f}, ".format(t, t_loss)
            print(msg)


def get_cum_reward(rewards, seq_len, max_len, gamma, is_delayed=True):
    cum_rewards_list = []
    seq_lens = seq_len.int().cpu().numpy().tolist()
    for i, reward in enumerate(rewards):
        T = seq_lens[i]
        cum_rewards = np.zeros(max_len)
        for k in range(T):
            if is_delayed:
                cum_rewards[k] = np.power(gamma, (T - k)) * reward
            else:
                for t in range(k, T):
                    cum_rewards[k] += np.power(gamma, (t - k)) * reward[t]
        cum_rewards_list.append(torch.FloatTensor(cum_rewards).cuda())
    cum_rewards_list = torch.stack(cum_rewards_list, dim=0)
    return cum_rewards_list


def calc_sti_mle_loss(dm, inputs, target, nlm=1):
    idx = inputs.shape[-1] - 1

    _inputs = torch.cat([inputs, target], dim=-1)
    outputs = dm.module.plm(_inputs, output_hidden_states=True)
    hidden_states = torch.cat(outputs.hidden_states[-nlm:], dim=-1).detach()

    calib_logits, inj_logits, _ = dm.module.policy.forward_logits(hidden_states, dm.module.init_hidden())
    lm_logits = outputs.logits

    lm_dist = F.softmax(lm_logits, dim=-1)[:, idx:-1]
    calib_dist = F.softmax(calib_logits[0], dim=-1)[:, idx:-1]
    inj_probs = F.softmax(inj_logits, dim=-1)[:, idx:-1]

    dist = lm_dist * inj_probs[:, :, 0, None] + calib_dist * inj_probs[:, :, 1, None]
    one_hot = F.one_hot(target, num_classes=dist.shape[-1])
    loss = -utils.clip_and_log((one_hot * dist).sum(-1)).sum(-1)

    return loss


def calc_mle_loss(dm, inputs, target, config: Config_PG):
    _inputs = torch.cat([inputs, target], dim=-1)
    outputs = dm.module.plm(_inputs, output_hidden_states=True)
    hidden_states = torch.cat(outputs.hidden_states[-config.nlm:], dim=-1).detach()

    if config.is_stg:
        logits, _ = dm(hidden_states, is_mle=True)
    else:
        logits, _, _ = dm(hidden_states, is_mle=True)

    if type(logits) is not list:
        logits = [logits]

    assert len(logits) == config.n_agent

    loss = 0
    for _logits in logits:
        idx = inputs.shape[-1] - 1
        shift_logits = _logits[..., idx:-1, :].contiguous()
        shift_labels = target.contiguous()

        dist = F.softmax(shift_logits, dim=-1)
        one_hot = F.one_hot(shift_labels, num_classes=dist.shape[-1])
        loss += -utils.clip_and_log((one_hot * dist).sum(-1)).sum(-1)

    return loss


def calc_mle_loss_w_plm(forward_data, target):
    injection_mask = torch.cat(forward_data['injection_masks'], dim=1)
    logits = torch.cat(forward_data['obs'], dim=1).squeeze(-1)

    maxlen = min(target.shape[1], logits.shape[1])

    injection_mask = injection_mask[:, :maxlen]
    logits = logits[:, :maxlen]
    target = target[:, :maxlen]

    dist = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(target, num_classes=dist.shape[-1])
    loss = -utils.clip_and_log((one_hot * dist).sum(-1))
    loss = (loss * injection_mask.float()).sum(-1)

    return loss


def init_int_module(int_module):
    torch.distributed.barrier()
    torch.distributed.broadcast(int_module.rew_rms_mean, 0)
    torch.distributed.broadcast(int_module.rew_rms_var, 0)
    torch.distributed.broadcast(int_module.rew_rms_count, 0)


def calc_rl_loss(forward_data, max_utter_len, verbose=False, int_module=None):
    injection_mask = forward_data['injection_masks'].detach().float()
    plm_tok_log_probs = forward_data['log_p_plm_tokens']
    p_plm = torch.exp(plm_tok_log_probs.detach())
    kl_div = 0
    device = plm_tok_log_probs.device

    maxlen = forward_data['obs'].shape[1]
    if config.is_ftg:
        probs = forward_data['obs']
        if config.algorithm == 'ppo':
            ratios = probs / p_plm
            kl_div = probs * utils.clip_and_log(ratios)
            kl_div = -kl_div.sum(-1)
    else:
        all_probs = forward_data['obs']
        p_inj = all_probs[:, :, 0]  # sample_batch x num_seq
        p_calib = all_probs[:, :, 1]

        if verbose or config.algorithm == 'ppo':
            p_not_inj = all_probs[:, :, 3]
            ratios = (p_inj * p_calib) / (p_not_inj * p_plm)
            kl_div = (p_inj * p_calib) * utils.clip_and_log(ratios)
            kl_div = -kl_div.sum(-1)
        sti_tok_probs = p_calib

    seq_len = forward_data['seq_lengths'] + 1
    sequence_mask = utils.sequence_mask(seq_len, maxlen=maxlen, device=device)

    ti_len = torch.sum(injection_mask, dim=-1).float().detach().cpu().numpy()
    score = np.array(forward_data['ext_rewards']).reshape(-1)
    ext_rewards = score

    info = {
        'reward': ext_rewards.mean(),
        'ti_len': ti_len.mean(),
    }

    if config.is_stg:
        with torch.no_grad():
            _inj_probs = forward_data['inj_probs'].detach()
            info['max(inj_prob)'] = _inj_probs.max(dim=-1).values.detach().cpu().numpy()[0]
            info['mean(inj_prob)'] = _inj_probs.mean(dim=-1).detach().cpu().numpy()[0]
            info['min(inj_prob)'] = _inj_probs.min(dim=-1).values.detach().cpu().numpy()[0]
    seq_len = sequence_mask.sum(-1)
    ext_rewards = get_cum_reward(ext_rewards, seq_len, maxlen, config.gamma)   # (n_batch, n_seq)
    ext_values = forward_data['ext_values']

    critic_loss = F.mse_loss(ext_values, ext_rewards.float(), reduction='none')
    critic_loss = (critic_loss * sequence_mask).sum(-1) / seq_len
    advantage = config.ext_scale * (ext_rewards - ext_values.detach().float())   # (n_batch, n_seq)

    if config.algorithm.startswith('ppo'):
        surr1 = ratios * advantage
        surr2 = torch.clamp(ratios, 1 - config.eps_clip, 1 + config.eps_clip) * advantage
        rl_loss = torch.min(surr1, surr2) * sequence_mask
    else:
        if config.is_stg:
            probs = sti_tok_probs * p_inj

        rl_loss = advantage * utils.clip_and_log(probs) * sequence_mask

        if config.inj_scheme is not None:
            rl_loss *= injection_mask

    rl_loss = -rl_loss.sum(-1) / seq_len

    loss_dict = {
        'rl': rl_loss,
        'critic': critic_loss
    }


    return loss_dict, kl_div, info


def get_writer(config, use_train_score=False):
    root_dir = f'./camera_ready_exp/{config.exp}-{config.domain}_SEED-{config.seed}'
    # root_dir = f'./runs_new_test/{config.exp}-{config.domain}_SEED-{config.seed}'
    if use_train_score:
        root_dir += '_UTS'
    writer_dir = os.path.join(root_dir, config.get_experiment_name())
    if os.path.exists(writer_dir):
        shutil.rmtree(writer_dir)
    os.makedirs(writer_dir, exist_ok=True)
    writer = SummaryWriter(writer_dir)
    return writer


class AverageMeter:
    def __init__(self):
        self.clear()

    def add_loss(self, loss_info):
        _loss_info = {key: val.clone().detach().cpu().numpy().mean().tolist() for key, val in loss_info.items()}
        self.losses.append(_loss_info)

    def add_info(self, info):
        self.infos.append(info)

    def clear(self):
        self.infos = []
        self.losses = []

    def _get_avg_data(self, data):
        n_data = len(data)
        if n_data == 0:
            return None

        avg_info = {key: 0 for key in data[0].keys()}

        for info in data:
            for key, val in info.items():
                avg_info[key] += val

        for key, val in avg_info.items():
            avg_info[key] /= n_data

        return avg_info

    def _sample_one_data(self, data):
        n_data = len(data)-1
        idx = random.randint(0, n_data)
        return {key: val for key, val in data[idx].items()}

    def get_avg_loss(self):
        return self._get_avg_data(self.losses)

    def get_avg_loss_msg(self):
        avg_losses = self._get_avg_data(self.losses)
        msg = ""
        for key, val in avg_losses.items():
            msg += "{0}: {1:.3f}, ".format(key, val)

        return msg

    def get_info_msg(self):
        msg = ""
        # avg_info = self._get_avg_data(self.infos)
        avg_info = self._sample_one_data(self.infos)
        for key, val in avg_info.items():
            msg += "{0}: {1:.2f}, ".format(key, val)
        return msg, avg_info


def parse_batch(batch, exp):
    if exp == 'summ':
        inputs, utter_len, (gt, target) = batch
    elif exp == 'qa':
        inputs, utter_len, (target, texts) = batch
        gt = texts['answer']
    else:
        inputs, target, _ = batch
        utter_len = target.shape[-1]
        gt = None

    return inputs, target, gt, utter_len


def train_rl(config, sampled_data, max_utter_len, verbose=False, int_module=None):
    loss_dict, kl_div, info = calc_rl_loss(sampled_data,
                                           max_utter_len,
                                           verbose=verbose,
                                           int_module=int_module)

    return loss_dict, info


def train_mle(config, dm, inputs, target):
    if config.is_ftg:
        loss = calc_mle_loss(dm, inputs, target, config)
    else:
        loss = calc_sti_mle_loss(dm, inputs, target)

    # if config.inj_scheme is None:
    #     _loss = calc_mle_loss(dm, inputs, target, config)
    # else:
    #     utter_len = utils.get_utter_len(inputs.shape[1], utter_len, config.gen_margin)
    #     sampled_data = sampler.sample(batch=batch, n_rounds=config.num_rounds, utter_length=utter_len)
    #     _loss = calc_mle_loss_w_plm(sampled_data, target)

    loss_dict = {'mle': loss}
    return loss_dict, None


def train(config, sampler, optimizer, verbose=0, obj='rl'):
    best_reward = 0
    prev_time = time()
    cp_path = None

    n_data = len(sampler.loader.loader)
    steps_per_epoch = int(n_data / config.num_batch)
    if n_data % config.num_batch > 0:
        steps_per_epoch += 1
    training_steps = steps_per_epoch * config.num_epochs

    dm = sampler.dm
    int_module = None

    is_primary = utils.is_primary()

    writer = None
    avg_meter = None
    if is_primary:
        writer = get_writer(config, args.use_train_score)
        avg_meter = AverageMeter()

    scheduler = None
    if config.scheduler == 'linear':
        def lr_lambda(env_steps):
            return (
                    1
                    - min(env_steps, training_steps)
                    / float(training_steps)
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    config.valid_freq = config.valid_freq // args.world_size

    if is_primary:
        print("EXP:", config.get_experiment_name())
        print(f"n_data: {n_data}, steps_per_epoch: {steps_per_epoch}")
        print(f"num_batch: {config.num_batch}, num_epoch: {config.num_epochs}")
        print(f"valid_freq: {config.valid_freq}, num_steps: {training_steps}")

    config.step = 0
    for t in range(training_steps):
        if t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            sampler.loader.sampler.set_epoch(epoch)
            if epoch > 0:
                best_reward, cp_path = validate_and_store_model(sampler, config, best_reward, cp_path)

        # if t % config.valid_freq == 0:
        #     if args.use_train_score:
        #         check_train_score(sampler, config, writer=writer, t=t, percentage=1)
        #     validate(sampler, config, writer=writer, t=t, use_multi_gpu=use_multi_gpu, percentage=1)

        if int_module is not None:
            init_int_module(int_module)

        optimizer.zero_grad()
        for i in range(config.num_batch):
            batch = sampler.get_batch(config.num_rounds)
            inputs, target, gt, utter_len = parse_batch(batch, config.exp)

            if obj == 'rl':
                # utter_len = utils.get_utter_len(inputs.shape[1], utter_len, config.gen_margin)
                utter_len = max_utter_len(config.exp)
                utter_len = utils.get_utter_len(inputs.shape[1], utter_len, 0)
                sampled_data = sampler.sample(batch=batch, n_rounds=config.num_rounds, utter_length=utter_len)

                loss_dict, info = train_rl(config, sampled_data, utter_len, verbose=verbose, int_module=int_module)
            else:
                inputs = inputs.cuda()
                target = target.cuda()

                loss_dict, info = train_mle(config, dm, inputs, target)

            if is_primary and info is not None:
                avg_meter.add_info(info)

            loss = 0
            for key, val in loss_dict.items():
                loss += val

            loss /= config.num_batch

            # gradient accumulation
            loss.mean().backward()

            if is_primary:
                avg_meter.add_loss(loss_dict)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            if is_primary and writer:
                writer.add_scalar("info/lr", scheduler.get_last_lr()[0], t)

        config.step += 1

        if is_primary:
            avg_loss_data = avg_meter.get_avg_loss()
            for key, val in avg_loss_data.items():
                writer.add_scalar(f"loss/{key}", val, t)

            if t % config.print_freq == 0:
                cur_time = time()

                info_msg = None

                if obj == 'rl':
                    info_msg, avg_info = avg_meter.get_info_msg()
                    for key, val in avg_info.items():
                        writer.add_scalar(f"info/{key}", val, t)

                msg = f"step: {t}, "
                msg += avg_meter.get_avg_loss_msg()

                if info_msg is not None:
                    msg += info_msg

                msg += "elapsed: {0:.2f}".format(cur_time - prev_time)
                print(colored(msg, 'yellow'))
                prev_time = cur_time

                if obj == 'rl':
                    dm.module.print_dialogue(sampled_data, print_injection_info=config.is_stg)
                    if gt is not None:
                        print("GT:", gt[0])

                avg_meter.clear()

    best_reward, cp_path = validate_and_store_model(sampler, config, best_reward, cp_path)


    if is_primary:
        writer.close()

    return cp_path


def get_avg_score(scores, use_multi_gpu=False):
    if use_multi_gpu:
        _scores = []
        for score in scores:
            _scores.append(torch.FloatTensor([score]).cuda())
        _scores = torch.cat(_scores)

        tot_list = utils.gather(_scores)

        if utils.is_primary():
            scores = torch.cat(tot_list)
            return np.mean(scores.cpu().numpy())

    else:
        return np.mean(scores)

    return None


def add_params(parser):
    parser.add_argument('-m', '--mode', default='stg', choices=['stg', 'ftg'], type=str)
    parser.add_argument('-e', '--exp', default='tod', choices=['tod', 'summ', 'qa'], type=str)
    parser.add_argument('-d', '--domain', default=None, type=str, required=True)
    parser.add_argument('-td', '--target_domain', default=None, type=str, required=False)
    parser.add_argument('-cid', '--cuda_id', default=0, type=int)
    parser.add_argument('--obj', default='rl', choices=['mle', 'rl'], type=str)
    parser.add_argument('--use_train_score', type=utils.str2bool, default=False)

    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--algo', type=str, default='ac', choices=['ppo', 'ac'])
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--dim', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=None)
    parser.add_argument('--nr', type=int, default=1)
    parser.add_argument('--sa', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--temp2", type=float, default=1.0)
    parser.add_argument('--scheme', type=str, default=None, choices=['sample', 'cat', 'max', 'greedy'])
    parser.add_argument('--inj_scheme', type=str, default=None, choices=[None, 'random', 'max', 'mix'])
    parser.add_argument('--use_eos', type=utils.str2bool, default=True)
    parser.add_argument('--pretrain_epochs', type=int, default=None)
    parser.add_argument('--num_batch', type=int, default=None)
    parser.add_argument('--scheduler', type=str, default=None, choices=[None, 'linear'])

    parser.add_argument('--world_size', type=int)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--backend', type=str, default='nccl')


def set_config(config, args):
    config.mode = args.mode
    config.domain = args.domain
    config.target_domain = args.target_domain
    config.algorithm = args.algo
    config.seed = args.seed
    config.use_eos = args.use_eos
    config.num_rounds = args.nr

    config.temperature4plm = args.temp
    config.temperature4calib = args.temp2
    config.inj_scheme = args.inj_scheme
    config.scheduler = args.scheduler

    config.set_exp(args.exp,
                   dim=args.dim,
                   gamma=args.gamma,
                   score_alpha=args.sa,
                   scheme=args.scheme,
                   lr=args.lr,
                   wd=args.wd,
                   pretrain_epochs=args.pretrain_epochs,
                   num_layers=args.n_layers,
                   num_batch=args.num_batch)

    config.init()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    add_params(ap)
    args = ap.parse_args()
    utils.setup_multi_gpu(args)
    utils.set_seed(args.seed)

    config = Config_PG()
    if args.obj == 'mle':
        args.mode = f'mle_{args.mode}'
    set_config(config, args)

    device = None
    use_multi_gpu = True
    if args.world_size is None:
        device = 'cuda:{}'.format(args.cuda_id)
        use_multi_gpu = False

    assert use_multi_gpu

    if config.is_stg:
        dm = modules.DM_STG(config)
    else:
        dm = modules.DM_FTG(config)

    dm = utils.build_ddp_model(dm, args, is_train=True)
    sampler = modules.DialogueSampler(dm, config, args=args)

    optimizer = AdamW(dm.module.get_parameters(), lr=config.lr, weight_decay=config.wd)

    if config.pretrain_epochs > 0:
        pretrain(config, sampler, optimizer)

    cp_path = train(config, sampler, optimizer, args.verbose, obj=args.obj)

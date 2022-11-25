import os
import torch
import torch.distributed as dist
from torch.nn import functional as F
import numpy as np
import copy
import functools
import pickle
import datetime
import argparse
import random
import re
from nltk import word_tokenize


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


def sequence_mask(lengths, maxlen=None, device='cpu', dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    mask = torch.arange(maxlen, device=device)[None, :] < lengths[:, None]
    mask.type(dtype)
    return mask


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def clip_and_log(prob):
    return torch.log(torch.clip(prob, 1e-20, 1.0))


def load_dm(cp_path, target_domain=None, args=None):
    import modules
    from config import Config_PG

    def set_attr(key, dict, func=None, ckey=None, set_val=None, default=None):
        ckey = ckey or key
        if key in dict:
            value = dict[key] if set_val is None else set_val
            if func:
                value = func(dict[key])
            config[ckey] = value
        else:
            config[ckey] = default

    basename = os.path.basename(cp_path)
    start_idx = 1
    if not 'SEED' in basename:
        basename = os.path.dirname(cp_path).split(os.path.sep)[-1]
        start_idx = 0
    else:
        basename = basename[:basename.rfind('.')]

    config_items = basename.split("_")
    _config_items = {}

    for item in config_items[start_idx:]:
        key_val = item.split("-")
        if len(key_val) == 1:
            _config_items[key_val[0]] = True
            continue
        key = key_val[0]
        val = '-'.join(key_val[1:])
        _config_items[key] = val
    config_items = _config_items

    print(config_items)

    config = Config_PG()

    if 'tod' in config_items:
        exp = 'tod'
    elif 'summ' in config_items:
        exp = 'summ'
    elif 'qa' in config_items:
        exp = 'qa'

    config.exp = exp
    config.domain = config_items[exp]

    if 'FTG' in config_items or 'MLE' in config_items:
        mode = 'ftg'
        if 'STG' in config_items:
            mode = 'stg'
    else:
        mode = 'stg'

    config.mode = mode
    config.set_exp(exp)

    if config.exp == 'tod':
        config.target_domain = target_domain
    else:
        set_attr('td', config_items, ckey='target_domain', default=None)

    set_attr('dim', config_items, func=int)
    set_attr('nlm', config_items, func=int, default=1)
    set_attr('scheme', config_items, default='max')
    set_attr('EOS', config_items, ckey='use_eos', set_val=True, default=True)
    set_attr('interm', config_items, ckey='interm_layer', set_val=True, default=False)
    set_attr('temp', config_items, ckey='temperature4train', func=float, default=1)
    set_attr('SEED', config_items, ckey='seed', func=int, default=9)
    set_attr('inj', config_items, ckey='inj_scheme', default=None)

    config.init()
    learner_type = config_items['pg']
    config.adapter_type = learner_type

    if learner_type in ['lstm', 'gru']:
        config.adapter_type = 'rnn'
        config.rnn_type = learner_type
        config.num_layers = int(config_items['nlayer'])

    if config.mode == 'ftg':
        dm = modules.DM_FTG(config)
    else:
        dm = modules.DM_STG(config)

    weights = torch.load(cp_path, map_location='cpu')
    new_weights = copy.deepcopy(weights)

    dm.set_weights(new_weights)
    return dm, config


def get_injected_tokens(tokenizer, acts, vocab_size, return_str=True, injected_tokens=None):
    ignore_tok_ids = [tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
    if injected_tokens is None:
        injected_tokens = []
        for index, _id in enumerate(acts):
            if _id < vocab_size:
                injected_tokens.append((index, _id))

    if not return_str:
        return injected_tokens
    else:
        injected_report = ""
        for (index, tok_id) in injected_tokens:
            if tok_id in ignore_tok_ids:
                break
            tok = tokenizer.decode([tok_id]).strip()
            injected_report += "{}: ({}, {}) ".format(index, tok, tok_id)
        return injected_report


def set_up_distributed_training_multi_gpu(args):
    args.device_id = args.local_rank
    torch.cuda.set_device(args.device_id)
    args.distributed_rank = args.device_id
    torch.distributed.init_process_group(backend=args.backend,
                                         init_method='env://')


def is_primary():
    return get_rank() == 0


def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo", timeout=datetime.timedelta(0, 3600))
    else:
        return dist.group.WORLD



def gather(data):
    size_list = [None for _ in range(torch.distributed.get_world_size())]
    dist.all_gather_object(size_list, data.shape[0])
    max_len = max(size_list)
    data_shape = (max_len,)
    if data.dim() == 2:
        data_shape += (data.shape[-1],)

    tot_list = [torch.zeros(data_shape, dtype=data.dtype, device=data.device) for _ in size_list]
    dist.all_gather(tot_list, data)
    return tot_list


def get_utter_len(context_length, gt_len, margin):
    PLM_max_len = 1024
    _margin = PLM_max_len - (context_length + gt_len)
    _margin = min(_margin, margin)
    return gt_len + _margin


def top_k_top_p_filtering(logits, top_k=5, top_p=0.9, filter_value=-float('Inf'), is_probs=False):
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        if not is_probs:
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        else:
            cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices,
                                                             src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def ids2text(gen_seq, tokenizer, rm_remain=False):
    if type(gen_seq) != list:
        gen_seq = gen_seq.tolist()

    eos = tokenizer.eos_token_id

    try:
        gen_seq = gen_seq[:gen_seq.index(eos)] if gen_seq[0] != eos else [eos]
    except:
        pass

    gen_answer = tokenizer.decode(gen_seq, skip_special_tokens=True)
    gen_answer = tokenize(gen_answer.strip().lower())
    if rm_remain and gen_answer.rfind('.') > 0:
        gen_answer = gen_answer[:gen_answer.rfind('.') + 1]
    return gen_answer


def build_ddp_model(model, args, is_train=False):
    print("build_model:", args.device_id)
    model = torch.nn.parallel.DistributedDataParallel(model.to(args.device_id),
                                                device_ids=[args.device_id],
                                                output_device=args.device_id,
                                                find_unused_parameters=True)
    model.cuda()
    if is_train:
        model.train()
        model.module.plm.eval()
    else:
        model.eval()
    return model


def remove_eos_token(text, eos_token):
    toks = []
    for i, tok in enumerate(text.split()):
        if tok == eos_token:
            break
        else:
            toks.append(tok)
    return ' '.join(toks)


def get_plm_path(args, root_dir='./'):
    domain = args.domain
    try:
        if args.target_domain is not None:
            domain = args.target_domain
    except:
        pass

    plm_path = f"{root_dir}/{args.exp}/ft/"

    if args.exp in ['qa', 'summ']:
        plm_path += f"{domain}_{args.seed}"

    if args.exp in ['qa', 'summ'] and args.use_eos:
        plm_path += "_eos"

    return plm_path


def tokenize(text, eos_token=None):
    if eos_token is not None:
        text = remove_eos_token(text, eos_token)
    text = word_tokenize(text)
    text = ' '.join(text)
    return text


def get_loader(tokenizer, args, mode='train', sep_token=None):
    if args.exp == 'qa':
        from data_loader import QALoader
        loader = QALoader(tokenizer, domain=args.domain, mode=mode,
                          args=args, use_eos=args.use_eos)
    elif args.exp == 'summ':
        from data_loader import SummLoader
        loader = SummLoader(tokenizer, domain=args.domain, mode=mode,
                            args=args, use_eos=args.use_eos, preseqlen=args.preseqlen, sep_token=sep_token)
    elif args.exp == 'tod':
        from data_loader import FsWozLoader
        domain = args.domain if args.target_domain is None else args.target_domain
        loader = FsWozLoader(tokenizer, domain, mode=mode, args=args)

    return loader


def setup_multi_gpu(args):
    if args.world_size is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["NCCL_BLOCKING_WAIT"] = '1'
        set_up_distributed_training_multi_gpu(args)
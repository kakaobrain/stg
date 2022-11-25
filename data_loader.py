import os
import torch
import numpy as np
import json
import random
import copy

from utils import is_primary
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

CUR_PATH = os.path.dirname(os.path.abspath(__file__))


class WOZDataUtil(object):
    def __init__(self, domain, dacts, sents):
        from tod.fswoz.utils.loader.DataReader import DataReader
        from tod.fswoz.utils.nlp.nlp import normalize
        import re

        vocab = 'tod/fswoz/utils/resource/vocab'
        vocab = os.path.join(CUR_PATH, vocab)

        container = []
        util = DataReader(None, domain, 'dt', vocab)

        for dact, sent in zip(dacts, sents):
            # dact = util.preproc_dact(dact)
            # _sent = util.delexicalise(normalize(re.sub(' [\.\?\!]$', '', sent)), dact)
            _feat = util.formatter.format(dact)
            container.append([_feat, dact, sent])

        self.data = []

        for feat, dact, sent in container:
            a, sv, _, _ = util.genFeatVec(feat, util.cardinality, util.dfs)
            felements = [util.cardinality[x + util.dfs[1]] for x in sv]
            _row = [a, felements, sent, dact]
            self.data.append(_row)

        self.util = util


class WOZDataset(Dataset):
    def __init__(self, tokenizer, domain, mode='train', max_seq=80, seperator=' & ', use_valid=False):
        if use_valid:
            if mode == 'train_valid':
                mode = 'train'
        else:
            if mode == 'test':
                mode = 'test_full'

            if mode in ['valid', 'train_valid']:
                mode = 'train'

        file_path = f'tod/fswoz/data/{domain}/{mode}.txt'
        file_path = os.path.join(CUR_PATH, file_path)
        print(file_path)

        self.raw_codes = []
        self.raw_sents = []
        self.codes = []
        self.sents = []
        self.ids = []
        with open(file_path, encoding="utf-8") as f:
            cnt = 0
            for line in f:
                self.ids.append(cnt)
                line = line.strip()
                raw_str = line.lower()
                if len(raw_str.split()) > max_seq - 1:
                    raw_str = ' '.join(raw_str.split()[:max_seq - 1])

                code_str, sent_str = raw_str.split(seperator)[:2]
                self.raw_codes.append(code_str)
                self.raw_sents.append(sent_str.strip())

                sent_str += ' ' + tokenizer.eos_token

                code_str += seperator
                code_str = code_str.strip()

                tokenized_code = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code_str))
                tokenized_sent = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent_str))

                self.codes.append(tokenized_code)
                self.sents.append(tokenized_sent)

                cnt += 1

        self.util = WOZDataUtil(domain, self.raw_codes, self.raw_sents)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, item):
        return torch.tensor(self.codes[item]), \
               torch.tensor(self.sents[item]), \
               item, self.ids[item]


class AbstractLoader:
    def __len__(self):
        return len(self.dataset)

    def get_batch(self):
        try:
            batch = next(self.iter)
        except:
            self.iter = iter(self.loader)
            batch = next(self.iter)

        return batch

    def get_loader(self, dataset, mode, batch_size, args=None):
        is_train = mode == 'train'

        if args is None:
            sampler = None
            if is_train:
                sampler = RandomSampler(dataset)
            loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        else:
            shuffle = False
            if is_train:
                shuffle = True
            sampler = DistributedSampler(dataset, shuffle=shuffle, num_replicas=args.world_size, rank=args.distributed_rank)
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers, sampler=sampler)
            if is_primary():
                print(mode + " data size: %d" % len(loader.dataset))

        return loader, sampler


class FsWozLoader(AbstractLoader):
    def __init__(self, tokenizer, domain, mode="train", batch_size=1, args=None):
        domain = domain.split('-')[0]
        self.dataset = WOZDataset(tokenizer, domain, mode=mode, use_valid=args.use_train_score if args is not None else None)
        self.aux_data = self.dataset.util.data
        self.util = self.dataset.util.util

        self.loader, self.sampler = self.get_loader(self.dataset, mode, batch_size, args)
        self.iter = iter(self.loader)

    def get_batch(self):
        try:
            batch = next(self.iter)
        except:
            self.iter = iter(self.loader)
            batch = next(self.iter)

        code, sent, idx, data_id = batch

        idx = idx.detach().cpu().numpy()[0]

        aux_data = self.aux_data[idx].copy()
        aux_data.append(data_id)

        return code, sent, aux_data


class SummDataset(Dataset):
    def __init__(self, tokenizer, domain='CNN', mode='train', use_eos=True, data=None, preseqlen=0, sep_token=None, root_dir='summ/CNN'):
        if data is None:
            with open(os.path.join(f'{root_dir}/ids.json'), 'r') as fin:
                split_info = json.load(fin)
            data = split_info[f'{mode}_ids']
        else:
            data = copy.deepcopy(data)

        self.data = data
        self.preseqlen = preseqlen
        print("preseqlen:", self.preseqlen)

        n_data = {
            'train_valid': 500,
            'valid': 500,
            'test': 15000
        }

        if domain == 'CNN2':
            n_data['train'] = 6000
        elif domain == 'CNN':
            n_data['train'] = 3000
        elif domain == 'CNN05':
            n_data['train'] = 1500
        elif domain == 'CNN01':
            n_data['train'] = 300
        elif domain == 'CNN003':
            n_data['train'] = 100
        elif domain == 'CNN001':
            n_data['train'] = 50


        if mode == 'train':
            np.random.shuffle(self.data)

        self.data = self.data[:n_data[mode]]

        self.root_dir = root_dir
        self.use_eos = use_eos
        self.tokenizer = tokenizer
        self.eos_token_id = [self.tokenizer.eos_token_id]

        if sep_token is None:
            self.sep_token_id = [self.tokenizer.sep_token_id]
        else:
            self.sep_token_id = self.tokenizer.encode(sep_token)

    def __len__(self):
        return len(self.data)

    def _get_article(self, data, trlen=None):
        article = data['article']
        if trlen is not None:
            article = article[:trlen]
        if self.use_eos:
             article += self.eos_token_id
        article += self.sep_token_id
        return article

    def _get_answer(self, data):
        answer = data['abstract']
        if self.use_eos:
            answer += self.eos_token_id
        return answer

    def __getitem__(self, idx):
        data_id = self.data[idx]
        fpath = os.path.join(self.root_dir, f'{data_id}.json')
        with open(fpath, 'r') as f:
            data = json.load(f)

        _data = copy.deepcopy(data)
        article = self._get_article(data)

        input_seq = np.array(article)
        gt_tok = self._get_answer(data)
        gt_len = len(gt_tok)

        seq_len = len(article) + gt_len + self.preseqlen
        if seq_len > 1024: # gpt2 ctx len
            article = self._get_article(_data, trlen=1024-seq_len)
            input_seq = np.array(article)
            seq_len = len(article) + gt_len + self.preseqlen

        assert seq_len <= 1024

        gt_summ = self.tokenizer.decode(data['abstract']).strip().lower()
        return input_seq, gt_len, (gt_summ, torch.tensor(gt_tok))


class SummDataset4Pretrain(SummDataset):
    def __getitem__(self, idx):
        data_id = self.data[idx]
        fpath = os.path.join(self.root_dir, f'{data_id}.json')
        with open(fpath, 'r') as f:
            data = json.load(f)

        text = self.tokenizer.encode(self.tokenizer.pad_token) * 1024
        article = self._get_article(data)
        answer = self._get_answer(data)

        content = article + answer
        text[:len(content)] = content
        text = torch.tensor(text)
        sample = {'article': text, 'sum_idx': len(data['article'])}
        return sample


class SummLoader(AbstractLoader):
    def __init__(self, tokenizer, domain='CNN', mode="train", batch_size=1, args=None, use_eos=False, data=None, preseqlen=0, sep_token=None):
        self.dataset = SummDataset(tokenizer, domain, mode=mode, use_eos=use_eos, data=data, preseqlen=preseqlen, sep_token=sep_token)
        self.loader, self.sampler = self.get_loader(self.dataset, mode, batch_size, args)
        self.iter = iter(self.loader)


class QADataset(Dataset):
    def __init__(self, tokenizer, domain='1', mode='train', use_eos=False, data=None, root_dir='qa/data'):
        if data is None:
            fpath = f'{root_dir}/{mode}.json'
            with open(fpath, 'r') as f:
                data = json.load(f)
        else:
            data = copy.deepcopy(data)

        self.data = data

        k = int(domain)
        smallset_domain = {
            '05': 0.5,
            '01': 0.1,
            '005': 0.05
        }
        if domain in smallset_domain:
            k = smallset_domain[domain]
        n_data = {
            'train': int(k * 1000),
            'train_valid': 500,
            'valid': 500,
            'test': 12000
        }

        if mode == 'train':
            np.random.shuffle(self.data)

        self.data = self.data[:n_data[mode]]

        self.use_eos = use_eos
        self.tokenizer = tokenizer
        self.eos_token_id = [self.tokenizer.eos_token_id]
        self.sep_token_id = [self.tokenizer.sep_token_id]
        self.q_tokens = self.tokenizer.encode('?') + self.tokenizer.encode(' ?')
        self.q_token_id = self.tokenizer.encode(' ?')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        if self.use_eos:
            passage_query = data['passage'] + self.eos_token_id
            passage_query += data['query']
            if not data['query'][-1] in self.q_tokens:
                passage_query += self.q_token_id
            passage_query += self.eos_token_id
            answer = data['answer'] + self.eos_token_id
            postfix = " " + self.tokenizer.eos_token
            query_text = data['query_text'].strip()
            if not query_text.endswith('?'):
                query_text += " ?"
            texts = {
                'passage': data['passage_text'] + postfix,
                'query': query_text + postfix,
                'answer': data['answer_text'] + postfix
            }
        else:
            passage_query = data['passage']
            passage_query += data['query']
            if not data['query'][-1] in self.q_tokens:
                passage_query += self.q_token_id
            answer = data['answer']
            texts = {
                'passage': data['passage_text'],
                'query': data['query_text'],
                'answer': data['answer_text']
            }

        length = len(answer)

        passage_query = torch.tensor(passage_query)
        answer = torch.tensor(answer)

        return passage_query, length, (answer, texts)


class QALoader(AbstractLoader):
    def __init__(self, tokenizer, domain='1', mode="train", batch_size=1, args=None, use_eos=False, data=None):
        self.dataset = QADataset(tokenizer, domain, mode=mode, use_eos=use_eos, data=data)
        self.loader, self.sampler = self.get_loader(self.dataset, mode, batch_size, args)
        self.iter = iter(self.loader)


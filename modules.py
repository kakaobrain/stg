import os
import copy
import torch
import numpy as np
from torch.nn import functional as F

import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation_utils import BeamSearchScorer

from utils import dict_to_cpu, clip_and_log
from config import Config

from abc import ABC, abstractmethod
import termcolor
from nltk.tokenize import word_tokenize
from utils import top_k_top_p_filtering
import models

CUR_PATH = os.path.dirname(os.path.abspath(__file__))


class AbstractDM(ABC, torch.nn.Module):
    def __init__(self, config: Config, plm=None, tokenizer=None):
        super().__init__()
        self.config = config

        print("LM path:", self.config.ft_lm_path)

        self.plm = plm or AutoModelForCausalLM.from_pretrained(self.config.ft_lm_path, local_files_only=True)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.config.ft_lm_path, local_files_only=True)
        self.no_grad_plm()

    def no_grad_plm(self):
        self.plm.eval()

    def get_parameters(self):
        params = list(self.policy.parameters())
        return params

    def init_hidden(self, n_rounds=1):
        if self.config.adapter_type == 'rnn':
            device = self.get_device()
            return self.policy.init_hidden(device, n_rounds=n_rounds)
        else:
            return None

    def get_device(self):
        return next(self.plm.parameters()).device

    def get_lm_head_weights(self):
        dict_ = dict_to_cpu(self.plm.lm_head.state_dict())
        return dict_

    def set_weights(self, weights):
        self.policy.load_state_dict(weights)

    def print_dialogue(self, dialogue, colored=True, print_injection_info=True):
        if colored:
            colored = termcolor.colored
        else:
            colored = lambda x, _: x

        results = []

        utter = dialogue['sequences'][0]
        acts = dialogue['acts'][0]

        if print_injection_info:
            injected_tokens = []
            for index, _id in enumerate(acts):
                if _id < self.config.vocab_size:
                    injected_tokens.append((index, _id))

            injected_report = ""
            for (index, tok_id) in injected_tokens:
                if tok_id >= self.tokenizer.eos_token_id:
                    continue
                tok = self.tokenizer.decode([tok_id]).strip()
                injected_report += "{}: ({}, {}) ".format(index, tok, tok_id)

            print(colored('[' + injected_report.strip() + ']', 'red'))

        utter = utter.tolist()
        try:
            utter = utter[:utter.index(self.tokenizer.eos_token_id) + 1]
        except:
            pass

        next_utterance = self.tokenizer.decode(utter)
        next_utterance = next_utterance.strip().lower()
        if next_utterance:
            print(colored('PREDICT: {}'.format(next_utterance), 'green'))

        results.append(next_utterance)

        return results

    def plm_gen(self, input_ids: torch.LongTensor,
                max_length: int = 1024,
                utter_length: int = 128,
                **model_kwargs):

        eos_token_id = pad_token_id = self.tokenizer.eos_token_id
        sequence_lengths, unfinished_sequences, cur_len = self.plm._init_sequence_length_for_generation(
            input_ids, max_length
        )
        seq_len = torch.zeros_like(unfinished_sequences)
        start_len = cur_len
        use_greedy_sample = 'greedy' in self.config.scheme
        p_tokens = []
        while (cur_len - start_len) < utter_length:
            model_inputs = self.plm.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self.plm(**model_inputs)
            lm_next_token_logits = outputs.logits[:, -1].detach() / self.config.temperature4plm

            if not use_greedy_sample:
                lm_next_token_logits = top_k_top_p_filtering(lm_next_token_logits,
                                                             top_k=self.config.top_k, top_p=self.config.top_p)

            lm_dist = F.softmax(lm_next_token_logits, dim=-1)

            if use_greedy_sample:
                next_tokens = torch.argmax(lm_dist, dim=-1, keepdim=True)
            else:
                next_tokens = torch.multinomial(lm_dist, num_samples=1)

            p_next_tokens = lm_dist.gather(1, next_tokens)
            p_tokens.append(p_next_tokens)

            next_tokens = next_tokens.squeeze(1)
            next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            cur_len = cur_len + 1

            sequence_lengths, unfinished_sequences = self.plm._update_seq_length_for_generation(
                sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
            )
            seq_len += unfinished_sequences

            if unfinished_sequences.max() == 0:
                break

            model_kwargs = self.plm._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.plm.config.is_encoder_decoder
            )

        scores = torch.cat(p_tokens, dim=1)
        input_ids = input_ids[:, start_len:]
        return scores, input_ids, seq_len

    def _cat_and_sample(self, dist1, dist2, dist=None, use_filter=True, use_greedy_sample=False):
        if dist is None:
            dist = torch.cat([dist1, dist2], dim=-1)

        if not use_greedy_sample and use_filter:
            dist = top_k_top_p_filtering(dist, is_probs=True, filter_value=0.0,
                                         top_k=self.config.top_k, top_p=self.config.top_p)

        if use_greedy_sample:
            next_token = torch.argmax(dist, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(dist, num_samples=1)

        dist_ids = torch.div(next_token, self.config.vocab_size, rounding_mode='floor')
        do_not_inject = torch.less(next_token, self.config.vocab_size)
        p_token = dist.gather(1, next_token)
        next_tokens = next_token % self.config.vocab_size
        return next_tokens, p_token, do_not_inject, dist_ids

    def get_inj_prob(self, inj_probs):
        if not self.policy.training and self.config.fixed_inj_prob is not None:
            _inj_probs = torch.zeros_like(inj_probs)
            _inj_probs[:, 0] = 1 - self.config.fixed_inj_prob
            _inj_probs[:, 1] = self.config.fixed_inj_prob
            inj_probs = _inj_probs
        return inj_probs

    def _forward_stg_multi(self, plm_dist, inj_probs, calib_lm_dists, sample_scheme_dict):
        inj_probs = self.get_inj_prob(inj_probs)
        if sample_scheme_dict['use_cat'] or sample_scheme_dict['use_mix']:
            lm_next_token_probs = plm_dist * inj_probs[:, 0, None]

            calib_next_token_probs = []
            for i, calib_lm_dist in enumerate(calib_lm_dists):
                calib_next_token_probs.append(calib_lm_dist * inj_probs[:, i + 1, None])
            calib_next_token_probs = torch.cat(calib_next_token_probs, dim=-1)

            if sample_scheme_dict['use_mix']:
                mixture = lm_next_token_probs + calib_next_token_probs
                next_tokens, p_token, do_not_inject, dist_ids = \
                    self._cat_and_sample(None, None, dist=mixture, use_greedy_sample=False)
            else:
                next_tokens, p_token, do_not_inject, dist_ids = \
                    self._cat_and_sample(lm_next_token_probs, calib_next_token_probs, use_greedy_sample=False)

            p_inj_act = torch.where(do_not_inject, inj_probs[:, 0, None], torch.gather(inj_probs, 1, dist_ids))
            p_token = p_token / p_inj_act
        else:
            calib_next_token_probs = torch.cat(calib_lm_dists, dim=-1)

            if sample_scheme_dict['use_max'] or sample_scheme_dict['use_greedy']:
                inj_act = torch.argmax(inj_probs, dim=-1, keepdims=True)
            else:
                inj_act = torch.multinomial(inj_probs, num_samples=1)

            do_not_inject = torch.eq(inj_act, 0)

            _inj_act = torch.where(do_not_inject, 1, inj_act)

            fr = (_inj_act - 1) * self.config.vocab_size
            to = fr + self.config.vocab_size

            calib_lm_dist = []
            for i in range(len(fr)):
                _fr = fr[i]
                _to = to[i]
                calib_lm_dist.append(calib_next_token_probs[i, _fr:_to])

            calib_lm_dist = torch.stack(calib_lm_dist)

            p_inj_act = inj_probs.gather(1, inj_act)

            if sample_scheme_dict['use_greedy']:
                lm_next_token = torch.argmax(plm_dist, dim=-1, keepdim=True)
                calib_next_token = torch.argmax(calib_lm_dist, dim=-1, keepdim=True)
            else:
                lm_next_token = torch.multinomial(plm_dist, num_samples=1)
                calib_next_token = torch.multinomial(calib_lm_dist, num_samples=1)

            p_plm_token = plm_dist.gather(1, lm_next_token)
            p_calib_token = calib_lm_dist.gather(1, calib_next_token)

            next_tokens = torch.where(do_not_inject, lm_next_token, calib_next_token)
            p_token = torch.where(do_not_inject, p_plm_token, p_calib_token)

        injection_mask = torch.logical_not(do_not_inject)
        p_plm_token = plm_dist.gather(1, next_tokens)

        _p_actions = [p_inj_act.unsqueeze(1), p_token.unsqueeze(1),
                      p_plm_token.unsqueeze(1), inj_probs.unsqueeze(1)]
        _p_actions = torch.cat(_p_actions, dim=-1)

        return next_tokens, _p_actions, injection_mask, p_plm_token

    def plm_forward(self, input_ids, **model_kwargs):
        # prepare model inputs
        model_inputs = self.plm.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self.plm(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )

        plm_hidden_states = torch.cat(outputs.hidden_states[-self.config.nlm:], dim=-1)
        plm_hidden_states = plm_hidden_states.detach()

        plm_next_token_logits = outputs.logits[:, -1].detach()

        plm_next_token_logits /= self.config.temperature4plm

        return plm_hidden_states, plm_next_token_logits, outputs

    def get_init_rnn_context(self, plm_hidden_states, prev_state):
        if self.config.adapter_type == 'rnn':
            init_plm_hidden_states = plm_hidden_states[:, :-1]
            if self.config.is_stg:
                prev_state = self.policy.get_init_ctx(init_plm_hidden_states, prev_state)
            else:
                prev_state = self.policy.get_init_ctx(init_plm_hidden_states, prev_state)
        return prev_state

    def _forward_naive_injector(self, plm_dist, calib_dist, sample_scheme_dict, calib_tokens=None, plm_tokens=None):
        if calib_tokens is None:
            if sample_scheme_dict['use_greedy']:
                calib_tokens = torch.argmax(calib_dist, dim=-1, keepdim=True)
            else:
                calib_tokens = torch.multinomial(calib_dist, num_samples=1)

        if plm_tokens is None:
            if sample_scheme_dict['use_greedy']:
                plm_tokens = torch.argmax(plm_dist, dim=-1, keepdim=True)
            else:
                plm_tokens = torch.multinomial(plm_dist, num_samples=1)

        p_plm_tokens = plm_dist.gather(1, plm_tokens)
        p_calib_tokens = calib_dist.gather(1, calib_tokens)

        if self.config.inj_scheme == 'max':
            do_not_inject = p_calib_tokens > p_calib_tokens
            next_tokens = torch.where(do_not_inject, plm_tokens, calib_tokens)
            p_tokens = torch.where(do_not_inject, p_plm_tokens, p_calib_tokens)
        elif self.config.inj_scheme == 'mix':
            if sample_scheme_dict['use_cat']:
                next_tokens, p_tokens, do_not_inject, _ = \
                    self._cat_and_sample(plm_dist, calib_dist, dist=None, use_filter=not self.policy.training)
            else:
                mixture = (plm_dist + calib_dist) / 2
                next_tokens, p_tokens, do_not_inject, _ = \
                    self._cat_and_sample(None, None, dist=mixture, use_filter=not self.policy.training,
                                         use_greedy_sample=sample_scheme_dict['use_greedy'])
        elif self.config.inj_scheme == 'random':
            do_not_inject = random.choices([True, False], k=len(calib_tokens))
            do_not_inject = torch.BoolTensor(do_not_inject).to(calib_tokens.device).unsqueeze(-1)
            next_tokens = torch.where(do_not_inject, plm_tokens, calib_tokens)
            p_tokens = torch.where(do_not_inject, p_plm_tokens, p_calib_tokens)

        return next_tokens, p_tokens, do_not_inject

    def forward_step(self,
                     plm_last_state,
                     prev_state,
                     plm_next_token_logits,
                     sample_scheme_dict):

        extra_stat_dict = dict(inj_prob=None)
        if self.config.is_stg:
            use_filter = not sample_scheme_dict['use_greedy']
            use_filter = use_filter and (sample_scheme_dict['use_cat'] or sample_scheme_dict['use_max'])
            if use_filter:
                plm_next_token_logits = \
                    top_k_top_p_filtering(plm_next_token_logits, top_k=self.config.top_k, top_p=self.config.top_p)

            plm_dist = F.softmax(plm_next_token_logits, dim=-1)
            input_critic, (inj_probs, calib_lm_dist), prev_state = self.policy.sample(plm_last_state, prev_state,
                                                                                      use_filter=use_filter)
            next_tokens, p_action_items, injection_mask, p_plm_tokens = \
                self._forward_stg_multi(plm_dist, inj_probs, calib_lm_dist, sample_scheme_dict)

            extra_stat_dict['inj_prob'] = inj_probs[:, 1, None]
        else:
            input_critic, calib_tokens, calib_dist, prev_state, calib_logtis = \
                self.policy.sample(plm_last_state, prev_state, return_logits=True,
                                   use_greedy_sample=sample_scheme_dict['use_greedy'])

            if self.config.inj_scheme is None:
                plm_dist = F.softmax(plm_next_token_logits, dim=-1)
                next_tokens = calib_tokens
                injection_mask = torch.ones_like(next_tokens, dtype=torch.bool)
                p_plm_tokens = plm_dist.gather(1, next_tokens)
                p_action_items = p_plm_tokens.unsqueeze(1)
            else:
                plm_dist = F.softmax(plm_next_token_logits, dim=-1)

                next_tokens, p_tokens, do_not_inject = \
                    self._forward_naive_injector(plm_dist, calib_dist, sample_scheme_dict, calib_tokens=calib_tokens)

                injection_mask = torch.logical_not(do_not_inject)
                p_plm_tokens = plm_dist.gather(1, next_tokens)

                if self.config.is_mle:
                    p_action_items = calib_logtis.unsqueeze(1)
                else:
                    p_action_items = p_tokens.unsqueeze(1)

        return prev_state, p_plm_tokens, next_tokens, p_action_items, injection_mask, input_critic, extra_stat_dict

    def _forward(self, input_ids: torch.LongTensor,
                prev_state: torch.FloatTensor,
                max_length: int = 1024,
                utter_length: int = 128,
                **model_kwargs):

        eos_token_id = pad_token_id = self.tokenizer.eos_token_id

        sequence_lengths, unfinished_sequences, cur_len = self.plm._init_sequence_length_for_generation(
            input_ids, max_length
        )

        observations = []
        injection_masks = []
        p_plm_tokens = []
        prev_states = []
        inputs_critic = []
        extra_stats_dict = dict(inj_probs=list(), entropies=list())

        seq_len = torch.zeros_like(unfinished_sequences)

        start_len = cur_len

        has_started = False

        # sample scheme for STI
        sample_scheme_dict = {
            'use_greedy': not self.policy.training and 'greedy' in self.config.scheme,
            'use_max': not self.policy.training and 'max' in self.config.scheme,
            'use_cat': not self.policy.training and 'cat' in self.config.scheme,
            'use_mix': not self.policy.training and 'mix' in self.config.scheme
        }

        # auto-regressive generation
        while (cur_len - start_len) < utter_length:
            plm_hidden_states, plm_next_token_logits, outputs = self.plm_forward(input_ids, **model_kwargs)
            plm_last_state = plm_hidden_states[:, -1]

            if not has_started:
                prev_state = self.get_init_rnn_context(plm_hidden_states, prev_state)
                has_started = True

            prev_state, p_plm_token, next_tokens, p_action_items, injection_mask, input_critic, extra_stat_dict = \
                                    self.forward_step(plm_last_state, prev_state, plm_next_token_logits, sample_scheme_dict)

            observations.append(p_action_items)
            inputs_critic.append(input_critic.unsqueeze(1))
            prev_states.append(prev_state)
            p_plm_tokens.append(p_plm_token)
            injection_masks.append(injection_mask)

            extra_stats_dict['inj_probs'].append(extra_stat_dict['inj_prob']) if not extra_stat_dict['inj_prob'] is None else None

            next_tokens = next_tokens.squeeze(1)
            next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            cur_len = cur_len + 1

            # update sequence length
            sequence_lengths, unfinished_sequences = self.plm._update_seq_length_for_generation(
                sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
            )
            seq_len += unfinished_sequences

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # update model kwargs
            model_kwargs = self.plm._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.plm.config.is_encoder_decoder
            )

        if len(observations) > 0:
            observations = torch.cat(observations, dim=1)
            injection_masks = torch.cat(injection_masks, dim=-1)
            p_plm_tokens = torch.cat(p_plm_tokens, dim=1)
            inputs_critic = torch.cat(inputs_critic, dim=1)
            if self.config.is_stg:
                extra_stats_dict['inj_probs'] = torch.cat(extra_stats_dict['inj_probs'], dim=1)

            if self.config.is_mle:
                critic_value = None
            else:
                critic_value = self.policy.critic(inputs_critic).squeeze(-1)

        return input_ids, prev_states, observations, injection_masks, p_plm_tokens, seq_len, critic_value, extra_stats_dict

    def forward(self, inputs, prev_state=None,
                max_length=1024,
                utter_length=128, is_mle=False):

        self.no_grad_plm()

        if prev_state is None:
            prev_state = self.init_hidden(len(inputs))

        if is_mle:
            return self.policy(inputs, prev_state)
        else:
            return self._forward(inputs, prev_state, maxin_length=max_length, utter_length=utter_length)

    def beam_gen(self, input_ids, num_beams=5,
                    max_length=1024, do_early_stopping=True, **model_kwargs):
        with torch.no_grad():
            batch_size = input_ids.shape[0]
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                do_early_stopping=do_early_stopping,
                device=self.plm.device
            )

            input_ids, model_kwargs = self.plm._expand_inputs_for_generation(
                input_ids, expand_size=num_beams,
                is_encoder_decoder=self.plm.config.is_encoder_decoder, **model_kwargs
            )

            return self.beam_search(input_ids, beam_scorer,
                                    max_length=max_length, **model_kwargs)


    def beam_search(self, input_ids, beam_scorer, max_length=1024, **model_kwargs):
        eos_token_id = pad_token_id = self.tokenizer.eos_token_id

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        batch_beam_size, cur_len = input_ids.shape

        has_started = False

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        prev_state = self.init_hidden(len(input_ids))
        injection_masks = []
        is_stg = self.config.is_stg or self.config.inj_scheme is not None
        while cur_len < max_length:
            plm_hidden_states, plm_next_token_logits, outputs = self.plm_forward(input_ids, **model_kwargs)
            plm_last_state = plm_hidden_states[:, -1]

            if not has_started:
                prev_state = self.get_init_rnn_context(plm_hidden_states, prev_state)
                has_started = True

            if is_stg:
                next_token_scores, prev_state, plm_dist, calib_dist, prior = self.beam_step(plm_last_state, prev_state, plm_next_token_logits)
            else:
                next_token_scores, prev_state = self.beam_step(plm_last_state, prev_state, plm_next_token_logits)


            next_token_scores = next_token_scores + beam_scores[:, None]

            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]


            if is_stg:
                p_token_plm = plm_dist.gather(1, beam_next_tokens[:, None])
                p_token_calib = calib_dist.gather(1, beam_next_tokens[:, None])
                if prior is not None:
                    injection_mask = torch.where(prior[:, 0, None] > prior[:, 1, None],
                                                 torch.zeros_like(p_token_plm),
                                                 torch.ones_like(p_token_calib))
                else:
                    injection_mask = torch.where(p_token_plm > p_token_calib,
                                                 torch.zeros_like(p_token_plm),
                                                 torch.ones_like(p_token_calib))
            else:
                injection_mask = torch.zeros_like(beam_next_tokens).unsqueeze(-1)

            injection_masks.append(injection_mask)

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            cur_len = cur_len + 1

            model_kwargs = self.plm._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.plm.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self.plm._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        idx = torch.argmax(beam_scores)
        injection_masks = torch.cat(injection_masks, -1)

        return sequence_outputs["sequences"], sequence_outputs["sequence_scores"], injection_masks[idx]


class DM_STG(AbstractDM):
    def __init__(self, config: Config):
        super(DM_STG, self).__init__(config)

        if config.adapter_type == 'rnn':
            self.policy = models.MA_PGN_RNN_STG(config)
        else:
            self.policy = models.PGN(config)

        self.policy.load_lm_head(copy.deepcopy(self.get_lm_head_weights()))
        self.policy.train()

    def beam_step(self, plm_last_state, prev_state, plm_next_token_logits):
        plm_dist = F.softmax(plm_next_token_logits, dim=-1)
        input_critic, (inj_probs, calib_lm_dists), prev_state = self.policy.sample(plm_last_state, prev_state,
                                                                                  use_filter=False)
        inj_probs = self.get_inj_prob(inj_probs)
        plm_dist = plm_dist * inj_probs[:, 0, None]
        calib_dist = 0
        for i, calib_lm_dist in enumerate(calib_lm_dists):
            calib_dist += calib_lm_dist * inj_probs[:, i + 1, None]

        return clip_and_log(plm_dist + calib_dist), prev_state, plm_dist, calib_dist, None


class DM_FTG(AbstractDM):
    def __init__(self, config: Config):
        super(DM_FTG, self).__init__(config)

        if config.adapter_type == 'rnn':
            self.policy = models.PGN_FTI_RNN(config)
        else:
            self.policy = models.PGN_FTI(config)
        self.policy.load_lm_head(copy.deepcopy(self.get_lm_head_weights()))
        self.policy.train()

    def beam_step(self, plm_last_state, prev_state, plm_next_token_logits):
        calib_next_token_logits, _, prev_state = self.policy(plm_last_state, state_t=prev_state)

        if self.config.inj_scheme is None:
            _, _, calib_dist, prev_state = \
                self.policy.sample(plm_last_state, prev_state, return_logits=False, use_greedy_sample=False)
            next_token_scores = F.log_softmax(calib_next_token_logits, dim=-1)
            return next_token_scores, prev_state
        else:
            plm_dist = F.softmax(plm_next_token_logits, dim=-1)
            calib_dist = F.softmax(calib_next_token_logits, dim=-1)
            if self.config.inj_scheme == 'max':
                dist = torch.maximum(plm_dist, calib_dist)
            elif self.config.inj_scheme == 'mix':
                dist = (plm_dist + calib_dist)/2
            elif self.config.inj_scheme == 'random':
                vocab_size = plm_next_token_logits.shape[-1]
                use_plm = random.choices([True, False], k=vocab_size)
                use_plm = torch.BoolTensor(use_plm).to(plm_next_token_logits.device)
                dist = torch.where(use_plm, plm_dist, calib_dist)
            next_token_scores = clip_and_log(dist)
            return next_token_scores, prev_state, plm_dist, calib_dist, None


class PLM_wrapper(AbstractDM):
    def __init__(self, config: Config, plm=None, tokenizer=None):
        super(PLM_wrapper, self).__init__(config, plm=plm, tokenizer=tokenizer)

    def beam_step(self, plm_last_state, prev_state, plm_next_token_logits):
        return F.log_softmax(plm_next_token_logits, dim=-1), prev_state


class DialogueSampler:
    def __init__(self,
                 dm: AbstractDM,
                 config: Config,
                 mode='train',
                 args=None):

        assert mode in ['train', 'test']

        self.dm = dm
        self.config = config
        self.mode = mode
        self.loader = None

        self.tokenizer = dm.module.tokenizer

        if mode == 'train':
            if self.config.exp == 'tod':
                from data_loader import FsWozLoader
                from tod.fswoz.utils.loader.GentScorer import GentScorer
                domain = self.config.domain if self.config.target_domain is None else self.config.target_domain
                self.loader = FsWozLoader(self.tokenizer, domain, mode='train', args=args)
                self.valid_loader = FsWozLoader(self.tokenizer, domain, mode='valid', args=args)
                if args.use_train_score:
                    self.train_valid_loader = FsWozLoader(self.tokenizer, domain, mode='train_valid',
                                                          args=args)

                self.scorer = GentScorer(os.path.join(CUR_PATH, 'tod/fswoz/utils/resource/detect.pair'))
                self.score_func = self.fswoz_reward_func
            elif self.config.exp == 'summ':
                from data_loader import SummLoader
                from rouge_score import rouge_scorer
                self.loader = SummLoader(self.tokenizer, domain=self.config.domain, mode='train',
                                         use_eos=self.config.use_eos, args=args)
                self.valid_loader = SummLoader(self.tokenizer, domain=self.config.domain, mode='valid',
                                               use_eos=self.config.use_eos, args=args)
                if args.use_train_score:
                    self.train_valid_loader = SummLoader(self.tokenizer, domain=self.config.domain, mode='train_valid',
                                                   use_eos=self.config.use_eos, args=args, data=self.loader.dataset.data)
                self.metrics = ['rougeL']
                self.scorer = rouge_scorer.RougeScorer(self.metrics, use_stemmer=True)
                self.score_func = self.summ_reward_func
            elif self.config.exp == 'qa':
                from data_loader import QALoader
                from qa.src.Evaluation.bleu.bleu import Bleu
                from qa.src.Evaluation.rouge.rouge import Rouge
                self.loader = QALoader(self.tokenizer, domain=self.config.domain, mode='train',
                                       use_eos=self.config.use_eos, args=args)
                self.valid_loader = QALoader(self.tokenizer, domain=self.config.domain, mode='valid',
                                             use_eos=self.config.use_eos, args=args)
                if args.use_train_score:
                    self.train_valid_loader = QALoader(self.tokenizer, domain=self.config.domain, mode='train_valid',
                                                 use_eos=self.config.use_eos, args=args, data=self.loader.dataset.data)
                self.scorer = [Bleu(), Rouge()]
                self.score_func = self.qa_reward_func

    def get_batch(self, n_rounds, loader=None):
        if loader is None:
            loader = self.loader
        input, output, aux_data = loader.get_batch()
        input = input.repeat(n_rounds, 1)

        return input, output, aux_data

    def sample(self, batch=None, n_rounds=1,
               utter_length=128, max_length=1024,
               is_valid=False):

        if batch is None:
            loader = self.loader if not is_valid else self.valid_loader
            batch = self.get_batch(n_rounds, loader)

        input_tokens, _, aux_data = batch

        input_tokens = input_tokens.cuda()
        init_len = input_tokens.shape[-1]

        sequences, prev_state, obs, injection_masks, p_plm_tokens, seq_length, ext_values, extra_stats_dict = \
            self.dm(input_tokens, prev_state=None, max_length=max_length, utter_length=utter_length)

        output_tokens = sequences[:, init_len:]
        acts = torch.ones_like(output_tokens) * self.tokenizer.vocab_size
        acts = (torch.logical_not(injection_masks) * acts) + (injection_masks * output_tokens)

        output_sents = output_tokens.detach().cpu().numpy()
        results = {
            'sequences': output_sents,
            'acts': acts,
            'obs': obs,
            'injection_masks': injection_masks,
            'seq_lengths': seq_length,
            'log_p_plm_tokens': clip_and_log(p_plm_tokens),
            'inj_probs': extra_stats_dict['inj_probs'],
            'ext_values': ext_values,
        }

        if self.mode == 'train':
            results['ext_rewards'] = self.score_func(output_sents, aux_data, is_valid=is_valid),

        return results

    def _tokenize(self, text):
        n_eos_token = len(self.tokenizer.eos_token)
        last_token = text[-n_eos_token:]
        if self.config.use_eos and last_token == self.tokenizer.eos_token:
            text = text[:-n_eos_token]
            text = ' '.join(word_tokenize(text)) + ' ' + last_token
        else:
            text = ' '.join(word_tokenize(text))
        return text

    def qa_reward_func(self, outputs, aux_data, is_valid=False):
        (_, texts) = aux_data
        gt = self._tokenize(texts['answer'][0])

        def calc_score(gt, gen):
            gts, res = {'0': [gt]}, {'0': [gen]}
            score = 0.0
            for scorer in self.scorer:
                _score, _ = scorer.compute_score(gts, res)
                score += np.mean(_score)
            score /= len(self.scorer)
            return score

        scores = []
        for i, encoded in enumerate(outputs):
            encoded = encoded.tolist()

            try:
                encoded = encoded[:encoded.index(self.tokenizer.eos_token_id)+1]
            except:
                pass

            gen = self.tokenizer.decode(encoded, skip_special_tokens=not self.config.use_eos)
            gen = gen.strip().lower()
            gen = self._tokenize(gen)
            score = calc_score(gt, gen)
            scores.append(score)

        reward = scores

        return reward

    def summ_reward_func(self, outputs, aux_data, is_valid=False):
        (gt, _) = aux_data
        gt = gt[0].strip().lower()
        gt = self._tokenize(gt)
        scores = []

        for i, encoded in enumerate(outputs):
            gen = self.tokenizer.decode(encoded, skip_special_tokens=not self.config.use_eos)
            gen = gen.strip().lower()
            gen = self._tokenize(gen)

            # print("GT:", gt)
            # print("GEN:", gen)
            # print("-" * 50)

            score = self.scorer.score(gt, gen)
            score = [score[metric].fmeasure for metric in self.metrics]
            scores.append(np.mean(score))

        reward = scores
        return reward

    def fswoz_reward_func(self, outputs, aux_data, is_valid=False):
        assert len(outputs) == 1
        # outputs = outputs[0]
        a, felements, ref_sent, dact, _ = aux_data

        gens = []

        dact = self.loader.util.preproc_dact(dact)

        ref_sent = ' '.join(word_tokenize(ref_sent))
        ref_sent = self.loader.util.delexicalise(ref_sent, dact)

        # print("DACT:", dact)
        # print("REF:", ref_sent)
        for i, encoded in enumerate(outputs):
            gen_str = self.tokenizer.decode(encoded, skip_special_tokens=True)
            gen_str = gen_str.strip().lower()
            gen_str = ' '.join(word_tokenize(gen_str))

            gen_str = gen_str.replace('watts','watt -s').replace('televisions','television -s').replace('ports', 'port -s').replace('includes', 'include -s').replace('restaurants','restaurant -s').replace('kids','kid -s').replace('childs','child -s').replace('prices','price -s').replace('range','range -s').\
                replace('laptops','laptop -s').replace('familys','family -s').replace('specifications','specification -s').replace('ratings','rating -s').replace('products','product -s').\
                    replace('constraints','constraint -s').replace('drives','drive -s').replace('dimensions','dimension -s')

            gen_str = self.loader.util.delexicalise(gen_str, dact)
            # print("GEN:",gen_str)
            gens.append(gen_str)

        reward = [self.scorer.scoreSBLEU([[[gen], [ref_sent]]]) for gen in gens]

        return reward

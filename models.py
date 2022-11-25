import torch
from torch.nn import functional as F
from config import Config
from utils import top_k_top_p_filtering
import numpy as np


def activate(x):
    x = F.relu(x)
    return x


def build_ctx_rnn(rnn_type, in_dim, out_dim, num_layer, dropout=0):
    if rnn_type == 'lstm':
        ctx_rnn = torch.nn.LSTM(in_dim, out_dim, num_layer,
                                batch_first=True, dropout=dropout)
    else:
        ctx_rnn = torch.nn.GRU(in_dim, out_dim, num_layer,
                               batch_first=True, dropout=dropout)
    return ctx_rnn


class Agent(torch.nn.Module):
    def __init__(self, config: Config):
        super(Agent, self).__init__()
        self.config = config
        self.num_actions = config.vocab_size

        self.lm_fc = torch.nn.Linear(config.dim, config.lm_dim)
        self.lm_head = torch.nn.Linear(config.lm_dim, self.num_actions, bias=False)

    def load_lm_head(self, weights):
        self.lm_head.load_state_dict(weights)

    def forward(self, x_t, lm_h_t):
        x_t = x_t[..., -self.config.lm_dim:]
        _x_t = self.lm_fc(lm_h_t)
        x_t = x_t + _x_t
        logits_t = self.lm_head(x_t)
        return logits_t


class PGN(torch.nn.Module):
    def __init__(self, config: Config):
        super(PGN, self).__init__()

        self.config = config
        self.num_actions = config.vocab_size

        self.critic = torch.nn.Linear(config.dim, 1, bias=False)
        self.lm_head = torch.nn.Linear(config.lm_dim, self.num_actions, bias=False)

    def load_lm_head(self, weights):
        self.lm_head.load_state_dict(weights)

    def get_init_ctx(self, xs, init_state):
        return init_state

    def forward(self, x_t, state_t=None):
        h_t = activate(self.fc(x_t))
        logits_t = self.lm_head(x_t + self.lm_fc(h_t))

        inj_logits_t = self.inj_head(h_t)
        h_t = h_t.unsqueeze(1)
        return logits_t, inj_logits_t, h_t

    def init_hidden(self, device, n_rounds=1):
        if self.config.adapter_type == 'rnn':
            prev_h = torch.zeros(self.config.num_layers, n_rounds, self.config.dim, device=device)
            if self.config.rnn_type == 'lstm':
                prev_c = torch.zeros(self.config.num_layers, n_rounds, self.config.dim, device=device)
                prev_state = (prev_h, prev_c)
            else:
                prev_state = prev_h
            return prev_state
        else:
            return None

    def _sample_token(self, logits):
        if not self.training:
            logits = top_k_top_p_filtering(logits, top_k=self.config.top_k, top_p=self.config.top_p)

        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        p_token = probs.gather(1, token)
        return token, p_token

    def _build_ctx_rnn(self, in_dim, out_dim):
        return build_ctx_rnn(self.config.rnn_type, in_dim, out_dim, self.config.num_layers)

    # FIXME
    def sample(self, x_t, prev_state):
        logits, inj_logits, cur_state = self.forward(x_t, state_t=prev_state)

        inj_probs = F.softmax(inj_logits, dim=-1)

        if not self.training:
            logits /= self.config.temperature4calib
            logits = top_k_top_p_filtering(logits, top_k=self.config.top_k, top_p=self.config.top_p)

        probs = F.softmax(logits, dim=-1)

        return None, (inj_probs, probs), cur_state


class PGN_RNN(PGN):
    def __init__(self, config: Config):
        super(PGN_RNN, self).__init__(config)

        self.config = config
        self.num_actions = config.vocab_size

        self.ctx_rnn = self._build_ctx_rnn(self.config.lm_dim * self.config.nlm, self.config.dim)

        self.lm_fc = torch.nn.Linear(config.dim, config.lm_dim)
        self.lm_head = torch.nn.Linear(config.lm_dim, self.num_actions, bias=False)

        self.inj_fc, self.inj_head = self._build_injector(2)
        self.critic = torch.nn.Linear(config.dim * 3, 1, bias=False)

    def get_init_ctx(self, xs, init_state):
        h_t, init_state = self.ctx_rnn(xs, init_state)
        return init_state

    def get_ctx(self, x_t, state_t):
        should_squeeze = False
        if len(x_t.shape) == 2:
            x_t = x_t.unsqueeze(1)
            should_squeeze = True
        h_t, state_t = self.ctx_rnn(x_t, state_t)
        if should_squeeze:
            h_t = h_t.squeeze(1)
        return h_t, state_t

    def _build_injector(self, n_generator):
        n_channel = (n_generator + 1)
        inj_fc = None
        if self.config.interm_layer:
            inj_fc = torch.nn.Linear(self.config.dim * n_channel, self.config.dim * n_channel)
        inj_head = torch.nn.Linear(self.config.dim * n_channel, n_generator)
        return inj_fc, inj_head

    def _forward_injector(self, inj_h_t):
        if self.inj_fc is not None:
            inj_h_t = activate(self.inj_fc(inj_h_t))
        inj_logits_t = self.inj_head(inj_h_t)
        return inj_logits_t

    def _forward_lm(self, x_t, lm_h_t):
        x_t = x_t[..., -self.config.lm_dim:]
        lm_h_t = self.lm_fc(lm_h_t)
        x_t = x_t + lm_h_t

        logits_t = self.lm_head(x_t)
        return logits_t

    def forward(self, x_t, state_t):
        h_t, state_t = self.get_ctx(x_t, state_t)
        logits_t = self._forward_lm(x_t, h_t)

        return logits_t, h_t, state_t

    # FIXME
    def sample(self, x_t, prev_state, plm_token):
        pass


class MA_PGN_RNN(PGN_RNN):
    def __init__(self, config: Config):
        super(MA_PGN_RNN, self).__init__(config)
        self.config = config
        self.num_actions = config.vocab_size

        self.ctx_rnn = self._build_ctx_rnn(self.config.lm_dim * self.config.nlm, self.config.dim)
        self.agents = torch.nn.ModuleList([Agent(config) for _ in range(config.n_agent)])

        self.inj_fc, self.inj_head = self._build_injector(config.n_agent + 1)
        self.critic = torch.nn.Linear(config.dim * (config.n_agent + 2), 1, bias=False)

    def load_lm_head(self, weights):
        for agent in self.agents:
            agent.load_lm_head(weights)

    def h_transform(self, h_t):
        return h_t

    def _forward_lm(self, x_t, lm_h_t):
        lm_h_t = self.h_transform(lm_h_t)
        logits = [agent(x_t, lm_h_t) for agent in self.agents]
        return logits

    def _get_init_ctx(self, ctx_rnn, xs, init_state):
        _, init_state = ctx_rnn(xs, init_state)
        return init_state

    def _get_ctx(self, ctx_rnn, x_t, state_t):
        should_squeeze = False
        if len(x_t.shape) == 2:
            x_t = x_t.unsqueeze(1)
            should_squeeze = True
        h_t, state_t = ctx_rnn(x_t, state_t)

        if should_squeeze:
            h_t = h_t.squeeze(1)

        return h_t, state_t

    def get_init_ctx(self, xs, init_state):
        xs = self.feat_transform_lm(xs)
        return self._get_init_ctx(self.ctx_rnn, xs, init_state)

    def get_init_ctx_lm(self, xs, init_state):
        xs = self.feat_transform_lm(xs)
        return self._get_init_ctx(self.ctx_rnn, xs, init_state)

    def get_ctx_lm(self, x_t, state_t, embed_tokens=None):
        x_t = self.feat_transform_lm(x_t)
        return self._get_ctx(self.ctx_rnn, x_t, state_t)

    def get_init_ctx_inj(self, xs, init_state):
        xs = self.feat_transform_inj(xs)
        return self._get_init_ctx(self.ctx_rnn, xs, init_state)

    def get_ctx_inj(self, x_t, state_t, embed_tokens=None):
        x_t = self.feat_transform_inj(x_t, embed_tokens=embed_tokens)
        return self._get_ctx(self.ctx_rnn, x_t, state_t)

    def forward4mle(self, plm_logits, xs, init_state, idx):
        hs, state_t = self.get_ctx(xs, init_state)
        logits = self._forward_lm(xs, hs)

        plm_dist = F.softmax(plm_logits, dim=-1)[:, idx:-1]
        inj_logits = self._forward_injector(hs)[:, idx:-1]
        inj_probs = F.softmax(inj_logits, dim=-1)

        dist = plm_dist * inj_probs[:, :, 0, None]
        for i, _logits in enumerate(logits):
            _logits = _logits[:, idx:-1]
            calib_dist = F.softmax(_logits, dim=-1)
            dist += calib_dist * inj_probs[:, :, i+1, None]
        return dist

    def forward(self, x_t, state_t):
        h_t, state_t = self.get_ctx_lm(x_t, state_t)
        logits_t = self._forward_lm(x_t, h_t)

        return logits_t, state_t


class MA_PGN_RNN_STG(MA_PGN_RNN):
    def __init__(self, config: Config):
        super(MA_PGN_RNN_STG, self).__init__(config)
        self.config = config
        self.num_actions = config.vocab_size

        self.ctx_rnn = self._build_ctx_rnn(self.config.lm_dim * self.config.nlm, self.config.dim)

        self.agents = torch.nn.ModuleList([Agent(config) for _ in range(config.n_agent)])
        self.critic = torch.nn.Linear(config.dim, 1, bias=False)

    def feat_transform_lm(self, x):
        return x

    def feat_transform_inj(self, x):
        return x

    def forward(self, x_t, state_t):
        h_t, state_t = self.get_ctx(x_t, state_t)
        logits_t = self._forward_lm(x_t, h_t)

        return logits_t, state_t

    def _build_injector(self, n_generator):
        inj_fc = None
        if self.config.interm_layer:
            inj_fc = torch.nn.Linear(self.config.dim, self.config.dim)
        inj_head = torch.nn.Linear(self.config.dim, n_generator)
        return inj_fc, inj_head

    def forward_logits(self, x_t, prev_state):
        h_t, cur_state = self.get_ctx(x_t, prev_state)
        logits = self._forward_lm(x_t, h_t)
        inj_logits = self._forward_injector(h_t)
        return logits, inj_logits, cur_state

    def sample(self, x_t, prev_state, use_filter=True):
        h_t, cur_state = self.get_ctx(x_t, prev_state)
        logits = self._forward_lm(x_t, h_t)
        inj_logits = self._forward_injector(h_t)

        inj_probs = F.softmax(inj_logits, dim=-1)  # n_batch x (n_agent + 1)

        probs = []

        for _logits in logits:
            if not self.training:
                _logits /= self.config.temperature4calib
                if use_filter:
                    _logits = top_k_top_p_filtering(_logits, top_k=self.config.top_k, top_p=self.config.top_p)

            _probs = F.softmax(_logits, dim=-1)
            probs.append(_probs)

        input4critic = h_t

        return input4critic, (inj_probs, probs), cur_state


class PGN_FTI(PGN):
    def __init__(self, config: Config):
        super(PGN_FTI, self).__init__(config)

    def forward(self, x_t, state_t=None):
        h_t = activate(self.fc(x_t))
        logits_t = self.lm_head(x_t + self.lm_fc(h_t))

        h_t = h_t.unsqueeze(1)
        return logits_t, h_t

    def sample(self, x_t, prev_state, return_logits=False, use_greedy_sample=False):
        logits, h_t, cur_state = self.forward(x_t, state_t=prev_state)

        if not self.training:
            logits /= self.config.temperature4calib
            if not use_greedy_sample:
                logits = top_k_top_p_filtering(logits, top_k=self.config.top_k, top_p=self.config.top_p)

        calib_lm_probs = F.softmax(logits, dim=-1)

        if use_greedy_sample:
            act = torch.argmax(calib_lm_probs, dim=-1, keepdim=True)
        else:
            act = torch.multinomial(calib_lm_probs, num_samples=1)

        if return_logits:
            return h_t, act, calib_lm_probs, cur_state, logits
        else:
            return h_t, act, calib_lm_probs, cur_state


class PGN_FTI_RNN(PGN_FTI):
    def __init__(self, config: Config):
        super(PGN_FTI_RNN, self).__init__(config)

        self.ctx_rnn = self._build_ctx_rnn(self.config.lm_dim * self.config.nlm, self.config.dim)
        self.lm_fc = torch.nn.Linear(config.dim, config.lm_dim)
        self.critic = torch.nn.Linear(config.dim, 1, bias=False)

    def get_init_ctx(self, xs, init_state):
        hs, init_state = self.ctx_rnn(xs, init_state)
        return init_state

    def forward(self, x_t, state_t=None):
        if state_t is not None:
            should_squeeze = False
            if x_t.dim() == 2:
                x_t = x_t.unsqueeze(1)
                should_squeeze = True
            h_t, state_t = self.ctx_rnn(x_t, state_t)
            if should_squeeze:
                x_t = x_t.squeeze(1)
                h_t = h_t.squeeze(1)

        logits_t = self.lm_head(x_t[:, -self.config.lm_dim:] + self.lm_fc(h_t))
        return logits_t, h_t, state_t



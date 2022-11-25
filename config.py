import random
import os
import torch
import numpy as np

GPU_USABLE = torch.cuda.is_available()

GPT_DIM_TABLE = {
    'small': 768,
    'medium': 1024,
    'large': 1280
}

MODEL_TABLE = {
    'dialogpt': 'microsoft/DialoGPT-{}',
    'gpt2': 'gpt2-{}'
}


class Config:
    def __init__(self):
        self.exp = 'summ' # ['tod', 'summ', 'qa']
        self.domain = 'CNN' # ['hotel', 'restaurant', 'tv', 'laptop'], ['CNN']
        self.target_domain = None

        self.mode = 'stg'
        self.model_name = 'gpt2' # ['dialogpt', 'gpt2']
        self.gpt_size = 'medium' # ['small', 'medium', 'large', 'xl']

        self.min_turn = 1
        self.max_turn = 1
        # for optimizer
        self.lr = 0.00025
        self.alpha = 0.95
        self.eps = 0.01

        self.nlm = 1

        # train
        self.gamma = 1

        self.print_freq = 50
        self.save_record_freq = 100

        # for test
        self.top_k = 0
        self.top_p = 0.9

        self.temperature4plm = 1.0
        self.temperature4calib = 1.0

        self.max_utter_len = None
        self.vocab_size = None
        self.lm_dim = None
        self.num_actions = None
        self.results_path = None
        self.ft_lm_path = None
        self.use_eos = True

        self.seed = 9

        self.init()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, val):
        self.__dict__[key] = val

    def init(self):
        if self.exp == 'summ':
            self.max_utter_len = 105
            self.vocab_size = 50259
        elif self.exp == 'tod':
            self.max_utter_len = 80
            self.vocab_size = 50257
        elif self.exp in ['qa']:
            self.max_utter_len = 95
            self.vocab_size = 50258
        self.init_dir()
        self.set_gpt_size(self.gpt_size)
        self.num_actions = self.vocab_size

    def init_dir(self):
        root_dir = os.path.dirname(os.path.realpath(__file__))
        self.results_path = os.path.join(root_dir, self.exp)
        self.ft_lm_path = os.path.join(self.results_path, 'ft')
        self.model_path = os.path.join(self.results_path, self.domain)

        if self.exp in ['qa', 'summ']:
            domain = self.domain
            if self.target_domain is not None:
                domain = self.target_domain
            if self.use_eos:
                self.ft_lm_path = os.path.join(self.ft_lm_path, f'{domain}_{self.seed}_eos')
            else:
                self.ft_lm_path = os.path.join(self.ft_lm_path, f'{domain}_{self.seed}')
        else:
            self.ft_lm_path = os.path.join(self.ft_lm_path, self.domain)

        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.ft_lm_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

    def get_model_name(self):
        return MODEL_TABLE[self.model_name].format(self.gpt_size)

    def set_gpt_size(self, size):
        self.gpt_size = size
        self.lm_dim = GPT_DIM_TABLE[self.gpt_size]

    def sample_n_turn(self):
        if self.exp == 'tod':
            n_turn = 1
        else:
            n_turn = random.randint(self.min_turn, self.max_turn)
        return n_turn

    def get_experiment_name(self):
        name = self._get_name()

        if self.exp == 'tod':
            base_domain = ""
            domain = self.domain
            if self.target_domain is not None:
                base_domain = "base-{}_".format(self.domain)
                domain = self.target_domain
            exp_domain = "{}-{}_".format(self.exp, domain) + base_domain
            name = exp_domain + name
        else:
            if self.target_domain is None:
                name = "{}-{}_".format(self.exp, self.domain) + name
            else:
                name = "{}-{}_td-{}_".format(self.exp, self.domain, self.target_domain) + name

        if not self.mode.startswith('stg'):
            name = "{}_".format(self.mode.upper()) + name

        return name




class Config_PG(Config):
    def __init__(self):
        super(Config_PG, self).__init__()

        self.learner = 'pg'
        self.adapter_type = 'rnn'  # ['mlp', 'rnn']
        self.rnn_type = 'lstm'  # ['lstm', 'gru']
        self.algorithm = 'ac'  # ['ac', 'ppo']
        self.num_layers = 2
        self.scheme = 'sample'
        self.baseline = 'critic'

        self.score_alpha = 1
        self.pretrain_epochs = 0

        self.gamma = 1
        self.num_rounds = 1

        # for PPO algorithm
        self.eps_clip = 0.2
        
        self.n_agent = 1
        
        self.plm_gen = False
        self.interm_layer = False
        self.det = True
        self.inj_scheme = None
        self.nlm = 1
        self.do_valid_ppl = False
        self.do_valid_gen = True
        self.save_freq = 5
        self.step = 0  # for exp_decay
        self.scheduler = None
        self.int_scale = 0.1
        self.ext_scale = 1

        # for eval
        self.fixed_inj_prob = None

    def set_exp(self, exp,
                dim=None,
                gamma=None,
                score_alpha=None,
                scheme=None,
                lr=None,
                wd=None,
                pretrain_epochs=None,
                num_layers=None,
                num_batch=None):
        self.exp = exp

        self.dim = dim or 512
        self.num_layers = num_layers or self.num_layers
        self.wd = wd or 1e-5
        self.lr = lr or 1e-5

        self.scheme = scheme or self.scheme  # decoding scheme during in validation
        self.gamma = gamma if gamma is not None else self.gamma
        self.score_alpha = score_alpha if gamma is not None else self.score_alpha

        self.pretrain_epochs = pretrain_epochs if pretrain_epochs is not None else self.pretrain_epochs

        if self.exp == 'qa':
            self.print_freq = 1

            self.valid_freq = 64
            self.gen_margin = 5

            epochs = {
                '005': 50,
                '01': 40,
                '05': 20,
                '1': 10,
                '2': 5
            }
            self.num_epochs = epochs[self.domain] * 2
            self.num_batch = num_batch if num_batch is not None else 16
            self.pretrain_num_batch = 128

            self.fixed_inj_prob = None
            self.interm_layer = True

        elif self.exp == 'summ':
            self.print_freq = 1

            # self.valid_freq = 100
            self.valid_freq = 64

            self.gen_margin = 10

            epochs = {
                'CNN001': 25,
                'CNN003': 20,
                'CNN01': 15,
                'CNN05': 8,
                'CNN': 4,
                'CNN2': 2
            }

            self.num_epochs = epochs[self.domain]
            self.num_batch = num_batch if num_batch is not None else 16
            self.pretrain_num_batch = 64

            self.fixed_inj_prob = None
            self.interm_layer = True

        elif self.exp == 'tod':
            self.print_freq = 1
            self.valid_freq = 5
            self.gen_margin = 5
            # self.num_epochs = 30
            self.num_epochs = 60 # overfitting 체크용
            self.num_batch = num_batch if num_batch is not None else 10
            self.pretrain_num_batch = 10

            self.fixed_inj_prob = None
            self.interm_layer = True

        print(f'MODE:{self.mode}')

    @property
    def is_stg(self):
        return 'stg' in self.mode

    @property
    def is_ftg(self):
        return 'ftg' in self.mode

    @property
    def is_mle(self):
        return self.mode.startswith('mle')

    @property
    def use_greedy_baseline(self):
        return 'greedy' == self.baseline

    def _get_name(self):
        name = f"SEED-{self.seed}_"

        learner_type = self.adapter_type
        if self.adapter_type == 'rnn':
            learner_type = self.rnn_type

        tmpl = "{}-{}_{}-{}_dim-{}_bs-{}_nr-{}_lr-{}"
        name += tmpl.format(self.learner, learner_type, self.model_name, self.gpt_size, self.dim,
                            self.num_batch, self.num_rounds, self.lr)


        if self.adapter_type == 'rnn':
            name += f"_nlayer-{self.num_layers}"

        if not self.mode.startswith('mle'):
            name += f"_algo-{self.algorithm}"
            if self.plm_gen:
                name += "_GLM"
            if self.pretrain_epochs > 0:
                name += f"_PRE-{self.pretrain_epochs}"
            if self.gamma < 1:
                name += f"_gamma-{self.gamma}"
            if self.baseline == 'greedy':
                name += f'greedyBL'

        if self.scheduler is not None:
            name += f"_LRS-{self.scheduler}"

        if self.use_eos:
            name += "_EOS"

        if self.nlm > 1:
            name += f"_nlm-{self.nlm}"

        if self.is_stg:
            name += f"_scheme-{self.scheme}"
            name += f"_temp-{self.temperature4plm}"
            if self.interm_layer:
                name += "_interm"
        else:
            if self.inj_scheme is not None:
                name += f"_inj-{self.inj_scheme}"

        return name


if __name__ == "__main__":
    config = Config()
    print(config.get_experiment_name())
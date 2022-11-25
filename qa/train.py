import argparse
from datetime import datetime
import os
import time

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from data_loader import QALoader
from utils import set_seed, top_k_top_p_filtering, str2bool


def sample_seq(model, context, length, device, temperature=1, top_k=0, top_p=0.0):
    """ Generates a sequence of tokens
        Args:
            model: gpt/gpt2 model
            context: tokenized text using gpt/gpt2 tokenizer
            length: length of generated sequence.
            device: torch.device object.
            temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """

    context = torch.tensor(context, dtype=torch.long, device=device)
    generated = context
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated}
            outputs = model( **inputs)
            next_token_logits = outputs[0][:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated


def generate_sample(loader, tokenizer, model, num=1, eval_step=False, length=100, temperature=1, top_k=10, top_p=0.5, device=torch.device('cuda')):
    for i in range(num):
        batch = loader.get_batch()
        inputs, _, (labels, _) = batch
        context = inputs.tolist()
        summary = labels.tolist()
        generated_text = sample_seq(model, context, length, device, temperature, top_k, top_p)
        generated_text = generated_text[0, inputs.shape[-1]:].tolist()
        text = tokenizer.convert_ids_to_tokens(generated_text, skip_special_tokens=True)
        text = tokenizer.convert_tokens_to_string(text)

        print('passage & query', end='\n\n')
        print(tokenizer.decode(context[0]), end='\n\n')
        print("generated  answer", end='\n\n')
        print(text, end='\n\n')

        if eval_step==False:
            print('actual answer', end='\n\n')
            print(tokenizer.decode(summary[0]), end='\n\n')


def train(args, model, tokenizer, train_loader, valid_loader):
    loss_fct = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=80000)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = range(int(args.num_train_epochs))

    best_ppl = 15
    for _ in train_iterator:
        for step in range(len(train_loader.loader)):
            inputs, _, (labels, _) = train_loader.get_batch()
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()

            _inputs = torch.cat([inputs, labels], dim=-1)
            logits = model(_inputs)[0]

            idx = inputs.shape[-1]-1

            # only consider loss on reference summary just like seq2seq models
            shift_logits = logits[..., idx:-1, :].contiguous()
            shift_labels = labels.contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss/args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                print("step:{}, loss: {}".format(step, loss.item()))
                if (step + 1)/args.gradient_accumulation_steps == 1.0:
                    print('After 1st update: ', end='\n\n')
                    model.eval()
                    # generate_sample(valid_loader, tokenizer, model, num=2, eval_step=False, device=args.device)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                model.eval()

                results = evaluate(args, model, valid_loader, global_step)
                ppl = results['perplexity'].detach().cpu().numpy().tolist()
                print("PPL", ppl)
                if best_ppl > ppl:
                    print('Scored!! {0:.2f}, Saving trained model...'.format(ppl))
                    model_file = os.path.join(args.model_dir, 'pytorch_model.bin'.format(ppl))
                    print('model path:', model_file)
                    torch.save(model.state_dict(), model_file)

                    best_ppl = ppl

                print('After', global_step+1, 'updates: ', end='\n\n')
                # generate_sample(valid_loader, tokenizer, model, num=2, eval_step=True, device=args.device)


""" Returns perplexity score on validation dataset.
	Args:
		args: dict that contains all the necessary information passed by user while training
		model: finetuned gpt/gpt2 model
		eval_dataset: GPT21024Dataset object for validation data
		global_step: no. of times gradients have backpropagated
		ignore_index: token not considered in loss calculation
"""
def evaluate(args, model, loader, global_step=None):
    eval_output_dir = args.model_dir

    loss_fct = CrossEntropyLoss()

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for i in range(len(loader.loader)):
        inputs, _, (labels, _) = loader.get_batch()
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        model.train()

        _inputs = torch.cat([inputs, labels], dim=-1)

        idx = inputs.shape[-1] - 1

        with torch.no_grad():
            logits = model(_inputs)[0]
            shift_logits = logits[..., idx:-1, :].contiguous()
            shift_labels = labels.contiguous()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }
    print("perplexity:", perplexity.item())

    if global_step:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as f:
            for key in sorted(result.keys()):
                f.write('\n\n')
                f.write("time = %s, %s = %s, step = %s\n" % (datetime.now().strftime("%d/%m/%Y %H:%M:%S"), key, str(result[key]), str(global_step)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-5, type=float, required=False, help="learning rate")
    parser.add_argument("--seed", default=9, choices=[9, 99, 999, 9999], type=int, required=False, help="seed to replicate results")
    parser.add_argument("--n_gpu", default=1, type=int, required=False, help="no of gpu available")
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int, required=False, help="gradient_accumulation_steps")
    parser.add_argument("--batch_size", default=8, type=int, required=False, help="batch_size")
    parser.add_argument("--num_workers", default=4, type=int, required=False, help="num of cpus available")
    parser.add_argument("--device", default='cuda', required=False, help="torch.device object")
    parser.add_argument("--num_train_epochs", default=5, type=int, required=False, help="no of epochs of training")
    parser.add_argument("--model_dir", default="./qa/log_dir/ft", type=str, required=False, help="path to save trained model")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="max gradient norm.")
    parser.add_argument("--model_name_or_path", default="gpt2-medium", type=str, help="The model checkpoint for weights initialization.")
    parser.add_argument("--domain", default="1", type=str)
    parser.add_argument("--use_eos", default=True, type=str2bool)

    args = parser.parse_args()

    set_seed(args.seed)

    if args.use_eos:
        model_dir = os.path.join(args.model_dir, f"{args.domain}_{args.seed}_eos")
    else:
        model_dir = os.path.join(args.model_dir, f"{args.domain}_{args.seed}")

    os.makedirs(model_dir, exist_ok=True)

    print(args.use_eos, model_dir)

    args.model_dir = model_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    special_tokens = {'sep_token': '<|sep|>'}
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.save_pretrained(args.model_dir)

    train_loader = QALoader(tokenizer, domain=args.domain, mode='train', use_eos=args.use_eos)
    valid_loader = QALoader(tokenizer, domain=args.domain, mode='valid', use_eos=args.use_eos)

    print(f"n_train:{len(train_loader.loader)}, n_valid:{len(valid_loader.loader)}")

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, local_files_only=True)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    start = time.time()
    train(args, model, tokenizer, train_loader, valid_loader)
    print('total time: ', (time.time()-start)/60, " minutes", end='\n\n')

    config_file = os.path.join(args.model_dir, 'config.json')
    model.config.to_json_file(config_file)


if __name__ == '__main__':
    main()

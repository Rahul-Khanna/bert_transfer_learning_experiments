# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License

# Copyright (c) 2019 wenhu chen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Not sure how to properly cite this, but essentially code coming from:
https://github.com/wenhuchen/Table-Fact-Checking/blob/master/code/run_BERT.py

Which in turn is heavily influenced by the huggingface's examples

My main changes:
1. Broke up code for better readability and understanding
2. Changed the model training and evaluation steps
3. Caching features, so that once they're created you don't need to re-create them for future
   experiments
4. Removed some extraneous functionality (multi-gpu, fp16)
    * Though did keep the gradient_accumulation_steps concept (just commented it out for now)
        * I'm interested to explore that more
5. Changed model saving and loading
6. Changed model (frozen BERT/ RoBERTa -- just tune final linear projection)

Driver for running frozen BERT and frozen RoBERTa experiments.
"""

from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import argparse
import csv
import logging
import os
import random
import sys
import pickle
# import io
# import json

from prepare_data_for_bert import (QqpProcessor, convert_examples_to_features)
from fact_verification_model import FactVerificationClf

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from tqdm import tqdm, trange
from torch.nn import NLLLoss
from sklearn.metrics import f1_score
from transformers import (BertTokenizer, RobertaTokenizer)
from pprint import pprint
logger = logging.getLogger(__name__)

# Directly ported from
# https://github.com/wenhuchen/Table-Fact-Checking/blob/master/code/run_BERT.py
def simple_accuracy(preds, labels):
    return (preds == labels).mean()

# Directly ported from
# https://github.com/wenhuchen/Table-Fact-Checking/blob/master/code/run_BERT.py
def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

# Directly ported from
# https://github.com/wenhuchen/Table-Fact-Checking/blob/master/code/run_BERT.py
def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)

def load_raw_data(args, train=True):
    """
        Loads pre-processed data and creates InputExamples from this data
        This step is needed when creating features for BERT
        This concept is derived from the example scripts found here:
        https://github.com/huggingface/transformers/tree/master/examples
    """
    processor = QqpProcessor()
    if train:
        args.data_dir = os.path.join(args.data_dir, "tsv_data_{}".format(args.scan))
        logger.info("Datasets will be loaded from {}\n".format(args.data_dir))
        examples = processor.get_train_examples(args.data_dir)
    else:
        if args.do_eval:
            args.data_dir = os.path.join(args.data_dir, "tsv_data_{}".format(args.scan))
        logger.info("Datasets will be loaded from {}\n".format(args.data_dir))
        examples = processor.get_dev_examples(args.data_dir, dataset=args.test_set)

    return examples, processor

def build_features(args, examples, label_list, tokenizer, train=True):
    """
        Builds the corresponding feature set for whatever portion of data is sent in
        Attempts to load a cached version of the data if available and user allows it

        convert_examples_to_features function is directly ported from:
        https://github.com/wenhuchen/Table-Fact-Checking/blob/master/code/run_BERT.py

        and can be found in prepare_data_for_bert.py
    """
    if train:
        if os.path.exists("default_features/training.pickle") and args.load_cached_features:
            logger.info("Loading pickle file of features")
            with open("default_features/training.pickle", "rb") as f:
                features = pickle.load(f)
        else:
            features = convert_examples_to_features(
                examples, label_list, args.max_seq_length, tokenizer, fact_place=args.fact, balance=args.balance)

            logger.info("Saving pickle file of features")
            with open("default_features/training.pickle", "wb") as f:
                pickle.dump(features, f)
    else:
        if os.path.exists("default_features/{}.pickle".format(args.test_set)) and args.load_cached_features:
            logger.info("Loading pickle file of features")
            with open("default_features/{}.pickle".format(args.test_set), "rb") as f:
                features = pickle.load(f)
        else:
            features = convert_examples_to_features(
                examples, label_list, args.max_seq_length, tokenizer, fact_place=args.fact, balance=False)
            logger.info("Saving pickle file of features")
            with open("default_features/{}.pickle".format(args.test_set), "wb") as f:
                pickle.dump(features, f)

    return features

def build_data_loader(features, batch_size, train=True):
    """
        Wrapper function for code found in:
        https://github.com/wenhuchen/Table-Fact-Checking/blob/master/code/run_BERT.py

        But is a recommended concept from multiple Medium posts online. Using this
        method of sending data into the model speeds up training and eval process.
    """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if train:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader

def eval_and_save(args, global_step, period_info, model, optimizer, device, processor, label_list, num_labels, 
                  tokenizer, tr_loss):
    """
        Wrapper function for evaluating a model after period number of steps, and then checkpointing
        that model.

        Code partly taken from:
        https://github.com/wenhuchen/Table-Fact-Checking/blob/master/code/run_BERT.py
        
        I have different save functionality though, and my evaluate function signature is
        different.
    """
    
    output_dir = os.path.join(args.output_dir, 'save_step_{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_model_file = os.path.join(output_dir, "checkpoint.pt")
    info = evaluate(args, model, device, processor, label_list, num_labels, tokenizer, tr_loss,
                    global_step, save_dir=args.output_dir)
    
    period_info.append(info)

    save_model_checkpoint(model, optimizer, global_step, period_info, output_model_file)

    return period_info

def save_model_checkpoint(model, optimizer, global_step, period_info, file_name):
    """
        Function to create a checkpoint storing model and optimizer progress,
        the number of steps taken and the stats so far
    """
    output = {
              "model"       : model.state_dict(),
              "optimizer"   : optimizer.state_dict(),
              "global_step" : global_step + 1,
              "period_info" : period_info
            }
    torch.save(output, file_name)

def load_model_checkpoint(model, optimizer, file_name):
    """
        Function to load a model checkpoint and then load its contents into the relevant objects.
    """
    checkpoint = torch.load(file_name)
    starting_step = checkpoint['global_step']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    period_info = checkpoint['period_info']

    return model, optimizer, starting_step, period_info

def main():
    """
        Driver that has two modes, training and evaluation. 

        In training you will automatically
        evaluate and save your model every period number of steps taken.
        
        In eval mode you must provide a load_dir.

        Idea and parts of code taken from:
        https://github.com/wenhuchen/Table-Fact-Checking/blob/master/code/run_BERT.py
    """
    logger.info("Running %s" % ' '.join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--scan",
                        default="horizontal",
                        choices=["vertical", "horizontal"],
                        type=str,
                        help="The direction of linearizing table cells.")
    parser.add_argument("--data_dir",
                        default="processed_datasets",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir",
                        default="outputs",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_dir",
                        type=str,
                        help="The output directory where the model checkpoints will be loaded during evaluation")
    parser.add_argument("--fact",
                        default="first",
                        choices=["first", "second"],
                        type=str,
                        help="Whether to put fact in front.")
    parser.add_argument("--test_set",
                        default="dev",
                        choices=["dev", "test", "simple_test", "complex_test", "small_test"],
                        help="Which test set is used for evaluation",
                        type=str)
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--balance",
                        action='store_true',
                        help="balance between + and - samples for training.")
    parser.add_argument("--model_type",
                        default="bert",
                        type=str,
                        choices=["bert", "roberta"])
    parser.add_argument("--model_name",
                        default="bert-base-multilingula-cased",
                        type=str,
                        help="Model Type you'd like to use")
    parser.add_argument('--period',
                        type=int,
                        default=500)
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=6,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=5,
                        type=float,
                        help="Total number of training epochs to perform.")
    # parser.add_argument("--warmup_proportion",
    #                     default=0.1,
    #                     type=float,
    #                     help="Proportion of training to perform linear learning rate warmup for. "
    #                          "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--load_cached_features",
                        action="store_true",
                        help="store pre_computed features if possible")

    # General environment set up
    args = parser.parse_args()
    pprint(vars(args))
    sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.output_dir = "{}_fact-{}_{}".format(args.output_dir, args.fact, args.scan)

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    logger.info("device: {} n_gpu: {}".format(
        device, n_gpu))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    # args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # Now we start setting up our Bert related models
    if args.model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)
    elif args.model_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)

    model = FactVerificationClf(args.model_name, args.model_type)
    
    # if you have provided a checkpoint from which we should load from, we do so
    loaded_model = False
    if args.load_dir:
        checkpoint = os.path.join(args.load_dir, "checkpoint.pt")
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        model, optimizer, global_step, period_info = load_model_checkpoint(model, optimizer, checkpoint)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                  state[k] = v.to(device)
        
        tr_loss = list(period_info[-1].values())[0]["loss"]
        
        loaded_model = True
    
    # With everything generally set up we are ready to move to the main functionality
    loss_funct = NLLLoss(reduction='mean')
    model.to(device)

    # Training Functionality
    if args.do_train:
        logging.info("Creating file structure for output")
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Outputs will be saved to {}".format(args.output_dir))
        
        # Getting training data together
        train_examples, processor = load_raw_data(args)
    
        label_list = processor.get_labels()
        num_labels = len(label_list)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        if not loaded_model:
            global_step = 0
            period_info = []
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            tr_loss = 0
    
        tmp_global_step = 0
        proceed = False
        train_features = build_features(args, train_examples, label_list, tokenizer)
        train_dataloader = build_data_loader(train_features, args.train_batch_size)
        
        # Data is ready, so now we start training
        model.train()

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("Training epoch {} ...".format(epoch))
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                # Allows you to skip over training points your model was already trained till
                # Useful in the event of a crash
                if tmp_global_step >= global_step or proceed:
                    proceed = True
                else:
                    tmp_global_step += 1
                
                if proceed:
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    log_probs = model(input_ids, segment_ids, input_mask)
                    loss = loss_funct(log_probs, label_ids)
                    
                    # if args.gradient_accumulation_steps > 1:
                    #     loss = loss / args.gradient_accumulation_steps

                    tr_loss += loss.item()

                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    global_step += 1

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # if (step + 1) % args.gradient_accumulation_steps == 0:
                    #     total_norm = 0.0
                    #     for n, p in model.named_parameters():
                    #         if p.grad is not None:
                    #             param_norm = p.grad.data.norm(2)
                    #             total_norm += param_norm.item() ** 2
                    #     total_norm = total_norm ** (1. / 2)
                    #     preds = torch.argmax(log_probs, axis=1) == label_ids
                    #     acc = torch.sum(preds).float() / preds.size(0)
                    #     writer.add_scalar('train/gradient_norm', total_norm, global_step)
                    #     if args.fp16:
                    #         # modify learning rate with special warm up BERT uses
                    #         # if args.fp16 is False, BertAdam is used that handles this automatically
                    #         lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                    #         for param_group in optimizer.param_groups:
                    #             param_group['lr'] = lr_this_step
                    #     optimizer.step()
                    #     optimizer.zero_grad()
                    #     model.zero_grad()
                    #     global_step += 1

                    # If we have reached period number of steps, we evaluate our progress so far and
                    # create a checkpoint
                    if (global_step) % args.period == 0 and global_step > 0:
                        model.eval()
                        torch.set_grad_enabled(False)
                        period_info = eval_and_save(args, global_step, period_info, model, optimizer, device, processor,
                                                    label_list, num_labels, tokenizer, tr_loss)
                        model.train() 
                        torch.set_grad_enabled(True)
                        tr_loss = 0
                else:
                    tmp_global_step += 1

        # one final eval and save before exiting
        model.eval()
        torch.set_grad_enabled(False)
        eval_and_save(args, global_step, period_info, model, optimizer, device, processor,
                                    label_list, num_labels, tokenizer, tr_loss)

    # Eval mode assumes a model has already been loaded in above
    # It ensures that this happens by checking the loaded_model boolean.
    if args.do_eval and loaded_model:
        if not args.do_train:
            global_step = 0
        load_step = int(os.path.split(args.load_dir)[1].replace('save_step_', ''))
        print("load_step = {}".format(load_step))
        model.eval()
        torch.set_grad_enabled(False)
        evaluate(args, model, device, None, ["0","1"], 2, tokenizer, tr_loss,
                 global_step, save_dir=args.output_dir, load_step=load_step)
    else:
        logger.info("Need a model to load")


def evaluate(args, model, device, processor, label_list, num_labels, tokenizer, tr_loss, global_step,
             save_dir=None, load_step=0, loss_funct=NLLLoss(reduction='mean')):
    """
        Evaluates a passed in model on a passed in dataset. Dataset is passed in as a command line
        argument, so is pulled from args.test_set

        Inspiration and partial code from: 
        https://github.com/wenhuchen/Table-Fact-Checking/blob/master/code/run_BERT.py

    """

    # Load dataset
    eval_examples, processor = load_raw_data(args, train=False)
    features = build_features(args, eval_examples, label_list, tokenizer, train=False)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    eval_dataloader = build_data_loader(features, args.eval_batch_size, train=False)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # running eval
    batch_idx = 0
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    temp = []
    with torch.no_grad():
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

        
            log_probs = model(input_ids, segment_ids, input_mask)
            tmp_eval_loss = loss_funct(log_probs, label_ids)
            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(log_probs.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], log_probs.detach().cpu().numpy(), axis=0)

            # labels = label_ids.detach().cpu().numpy().tolist()
            # start = batch_idx*args.eval_batch_size
            # end = start+len(labels)
            # batch_range = list(range(start, end))
            # csv_names = [eval_examples[i][0].guid.replace("{}-".format(args.test_set), "") for i in batch_range]
            # facts = [eval_examples[i][0].text_b for i in batch_range]
            # labels = label_ids.detach().cpu().numpy().tolist()
            # assert len(csv_names) == len(facts) == len(labels)

            # temp.extend([(x, y, z) for x, y, z in zip(csv_names, facts, labels)])
            # batch_idx += 1

    # Computing final eval_loss, finding predicitions
    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)

    # evaluation_results = OrderedDict()
    # for x, y in zip(temp, preds):
    #     c, f, l = x
    #     if not c in evaluation_results:
    #         evaluation_results[c] = [{'fact': f, 'gold': int(l), 'pred': int(y)}]
    #     else:
    #         evaluation_results[c].append({'fact': f, 'gold': int(l), 'pred': int(y)})

    # print("save_dir is {}".format(save_dir))
    # output_eval_file = os.path.join(save_dir, "{}_eval_results.json".format(args.test_set))
    # with io.open(output_eval_file, "w", encoding='utf-8') as fout:
    #     json.dump(evaluation_results, fout, sort_keys=True, indent=4)

    # Understanding performance
    result = compute_metrics(preds, all_label_ids.numpy())
    loss = tr_loss/args.period if args.do_train and global_step > 0 else None

    log_step = global_step if args.do_train and global_step > 0 else load_step
    result['eval_loss'] = eval_loss
    result['global_step'] = log_step
    result['loss'] = loss

    # Ouputing performance
    output_eval_metrics = os.path.join(save_dir, "eval_metrics.txt")
    output = {log_step : result}
    with open(output_eval_metrics, "a") as writer:
        writer.write("***** Eval results {}*****\n".format(args.test_set))
        writer.write(str(output))
        writer.write("\n")
        logger.info("***** Eval results {}*****".format(args.test_set))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    if args.do_train:
        return output

if __name__ == "__main__":
    main()

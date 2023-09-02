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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import pandas as pd
import torch
import multiprocessing
import ast, astunparse
from tqdm import tqdm
from collections import defaultdict
from statistics import mean, median
from sklearn.metrics import recall_score,precision_score,f1_score

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

from model import Model

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
cpu_cont = 16

def get_example(item):
    url1,url2,tokenizer,args,cache,url_to_code = item
    if url1 in cache:
        code1 = cache[url1].copy()
    else:
        try:
            code = ' '.join(url_to_code[url1].split())
        except:
            code = ""
        code1 = tokenizer.tokenize(code)
    if url2 in cache:
        code2 = cache[url2].copy()
    else:
        try:
            code = ' '.join(url_to_code[url2].split())
        except:
            code = ""
        code2 = tokenizer.tokenize(code)
        
    return convert_examples_to_features(code1,code2,url1,url2,tokenizer,args,cache)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 url1,
                 url2

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.url1=url1
        self.url2=url2
        
        
def convert_examples_to_features(code1_tokens,code2_tokens,url1,url2,tokenizer,args,cache):
    """convert examples to token ids"""
    code1_tokens = code1_tokens[:args.block_size-4]
    code1_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code1_tokens+[tokenizer.sep_token]
    code2_tokens = code2_tokens[:args.block_size-4]
    code2_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code2_tokens+[tokenizer.sep_token]  
    
    code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = args.block_size - len(code1_ids)
    code1_ids += [tokenizer.pad_token_id]*padding_length
    
    code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = args.block_size - len(code2_ids)
    code2_ids += [tokenizer.pad_token_id]*padding_length
    
    source_tokens = code1_tokens+code2_tokens
    source_ids = code1_ids+code2_ids
    return InputFeatures(source_tokens,source_ids,url1,url2)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, submission_id, url_to_code, other_url, problem_id, pool=None):
        self.examples = []

        data = []
        cache = {}
        if submission_id not in url_to_code:
            raise ValueError("Unable to predict since the code snippet does not exist")
        for url2 in other_url:
            #if problem_id[url1] == problem_id[url2]:
                #continue
            #later include problem_id in data
            data.append((submission_id,url2,tokenizer, args,cache,url_to_code))
            #test
            #if len(data) == 100:
                #break
                
        self.examples = pool.map(get_example,tqdm(data,total=len(data)))
       

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        #call this 100 times, returned 100 times
        return torch.tensor(self.examples[item].input_ids)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def removeComments(source):
    '''Remove comments'''
    return list(filter(None, re.sub('//.*|/\*(.|\n)*?\*/', '', '\n'.join(source)).split('\n')))

def preprocess(file):
    '''Preprocess'''
    if file[-2:] == "py":
        with open(file, encoding = "utf-8") as f:
            ans = []
            can_parse = True
            try:
                lines = astunparse.unparse(ast.parse(f.read())).split('\n')
            except:
                can_parse = False
                pass
            if can_parse:
                for line in lines:
                    if line.lstrip()[:1] not in ("'", '"'):
                        ans.append(line)
            else:
                for line in f:
                    if line.lstrip()[:1] not in ("'", '"'):
                        ans.append(line)

    else:      
        with open(file, encoding = "utf-8") as f:
            ans = []
            for line in f:
                if line != "\n":
                    ans.append(line)
            
            removed = removeComments(ans)
            ans = []
            for line in removed:
                if all(x == " " for x in line):
                    continue
                ans.append(line)
                
    return ans

def evaluate(args, model, tokenizer, data_file, pool):
    """ Evaluate the model """
    eval_output_dir = args.output_dir
    
    code_ids = []
    problem_ids = {}
    
    #postfix = data_file.split('/')[-1].split('.txt')[0]
    index_filename = data_file
    logger.info("Creating features from index file at %s ", index_filename)
    url_to_code = {}
    
    with open('/'.join(index_filename.split('/')[:-1])+'/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url_to_code[js['idx']] = js['func']
            code_ids.append(js['idx'])
            problem_ids[js['idx']] = js['Task']
        
    file_name = data_file.split("/")[-1]
    submission_id = file_name.split(".")[0]
        
    directory = "/content/drive/MyDrive/CZ4079_FYP/Similar_Problems_Pipeline/dataset"

    filelist = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            filelist.append(os.path.join(root,file))
            
    filelist= [x for x in filelist if submission_id in x]
    print(filelist)
    assert len(filelist) == 1

    json_string = " ".join(preprocess(filelist[0]))
    print(json_string)
    
    url_to_code[str(submission_id)] = json_string
    
    
    eval_dataset = TextDataset(tokenizer, args, submission_id, url_to_code, code_ids, problem_ids, pool)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)

    # Eval!
    logger.info("***** Running evaluation and finding similar codes *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()
    logits = defaultdict(list)
    cur_idx = 0
    for i, batch in tqdm(enumerate(eval_dataloader), total = len(eval_dataloader)):
        inputs = batch.to(args.device)
        with torch.no_grad():
            cos_sim = model(inputs)
            similarity_scores = cos_sim.cpu().numpy()
            for res in similarity_scores:
                logits[problem_ids[code_ids[cur_idx]]].append(res)
                cur_idx += 1
        #if cur_idx % args.eval_batch_size == 0 and (cur_idx // args.eval_batch_size) % 10 == 0:
            #logger.info(f"  Processed {cur_idx} pairs")
        nb_eval_steps += 1
        
    problem_sim_scores = []
    for pid in logits:
        problem_sim_scores.append((mean(logits[pid]), pid, len(logits[pid])))
        
    problem_sim_scores.sort(reverse = True)
    
    result = []
    for i in range(min(30, len(problem_sim_scores))):
        result.append([f"{i+1}", "*" + problem_sim_scores[i][1], round(problem_sim_scores[i][0],3), problem_sim_scores[i][2]])
            
    return result
                                                
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--source_code_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--find_similar", action='store_true',
                        help="Whether to find similar problems to input.")    
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    pool = multiprocessing.Pool(cpu_cont)
    
    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path) 

    model = Model(model,config,tokenizer,args)
    logger.info("Training/evaluation parameters %s", args)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)      
            
    if args.find_similar:
        checkpoint_prefix = 'model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))       
        result = evaluate(args, model, tokenizer, args.source_code_file, pool = pool)
        print(result)
        logger.info("***** Test results *****") 
        df = pd.DataFrame(result, columns = ["Rank", "Problem ID", "Similarity", "Samples"])
        file_name = args.source_code_file.split("/")[-1]
        source_id = file_name.split(".")[0]
                
        print("results is saved at /outputs directory")
        df.to_csv(f"/content/drive/MyDrive/CZ4079_FYP/Similar_Problems_Pipeline/outputs/similarity_output_{source_id}.csv", index = False)

if __name__ == "__main__":
    main()

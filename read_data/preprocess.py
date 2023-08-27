'''
SSMix
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import numpy as np
from transformers import *
from datasets import load_dataset, load_metric, concatenate_datasets
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from datasets import load_from_disk

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "cb":("premise","hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "trec": ("text", None),
    "anli": ("premise", "hypothesis"),
    "cola":("sentence",None),
    "rotten":("text",None),
   "agnews":("text",None),
  "imdb":("text",None),
 "subj":("text",None),
 "amazon":("text",None),
 "dbpedia":("content",None),
 "yahoo":("text",None),
 "email":("text",None),
 "thunews":("text",None)
}

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    #f1 = float(f1_score(y_true=labels, y_pred=preds))
    return {
        "accuracy": acc,
    #    "f1": f1,
    }

def simple_accuracy(predictions, references):
    return float((predictions == references).mean())

class PreProcessor:
    def __init__(self, args=None, task_name='sst2', tokenizer=None, max_len=128, seed_num=0):
        set_seed(seed_num)
        self.args = args
        self.task_name = task_name
        self.tokenizer = tokenizer
        if 'trec' in self.task_name:
            self.task_name = 'trec'
           # self.datasets = load_dataset('trec')
            self.datasets=load_from_disk("dataset/dataset/trec/trec")
        elif self.task_name == 'anli':
           # self.datasets = load_dataset('anli')
             self.datasets=load_from_disk("dataset/dataset/anli/anli")
        elif self.task_name == 'imdb':
            #self.datasets=load_from_disk("dataset/dataset/data2/imdb")
               self.datasets=load_dataset('imdb')
        elif self.task_name == 'rotten':
             #self.datasets=load_from_disk("dataset/dataset/data2/rotten_tomatoes")
             self.datasets=load_dataset("rotten_tomatoes")
             if self.args.eda ==1:
                if self.args.seed==0:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/eda/eda_rotten0.csv'})
                elif self.args.seed==1:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/eda/eda_rotten1.csv'})
                elif self.args.seed==2:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/eda/eda_rotten2.csv'})
                elif self.args.seed==3:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/eda/eda_rotten3.csv'})
                elif self.args.seed==4:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/eda/eda_rotten4.csv'})
                else:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/eda/eda_rotten5.csv'})
                self.datasets['train']=da['train']

             if self.args.bt ==1:
                if self.args.seed==0:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_rotten0.csv'})
                elif self.args.seed==1:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_rotten1.csv'})
                elif self.args.seed==2:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_rotten2.csv'})
                elif self.args.seed==3:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_rotten3.csv'})
                elif self.args.seed==4:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_rotten4.csv'})
                else:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_rotten5.csv'})
                self.datasets['train']=da['train']
        elif self.task_name== 'agnews':
             self.datasets=load_from_disk("dataset/dataset/data2/ag_news")
        elif self.task_name == 'subj':
             self.datasets=load_from_disk("/kaggle/input/cbrunt3/dataset/dataset/data2/subj")
             #self.datasets=load_dataset('SetFit/subj')
             if self.args.eda ==1:
                if self.args.seed==0:
                   da=load_dataset("csv", data_files={'train':'/kaggle/input/cbrunt3/dataset/dataset/data2/eda/cb_subj0.csv'})
                elif self.args.seed==1:
                   da=load_dataset("csv", data_files={'train':'/kaggle/input/cbrunt3/dataset/dataset/data2/eda/cb_subj1.csv'})
                elif self.args.seed==2:
                   da=load_dataset("csv", data_files={'train':'/kaggle/input/cbrunt3/dataset/dataset/data2/eda/cb_subj2.csv'})
                elif self.args.seed==3:
                   da=load_dataset("csv", data_files={'train':'/kaggle/input/cbrunt3/dataset/dataset/data2/eda/cb_subj3.csv'})
                elif self.args.seed==4:
                   da=load_dataset("csv", data_files={'train':'/kaggle/input/cbrunt3/dataset/dataset/data2/eda/cb_subj4.csv'})
                else:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/eda/cb_subj5.csv'})
                self.datasets['train']=da['train']

             if self.args.bt ==1:
                if self.args.seed==0:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_subj0.csv'})
                elif self.args.seed==1:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_subj1.csv'})
                elif self.args.seed==2:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_subj2.csv'})
                elif self.args.seed==3:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_subj3.csv'})
                elif self.args.seed==4:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_subj4.csv'})
                else:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_subj5.csv'})
                self.datasets['train']=da['train']
        elif self.task_name == 'amazon':
             #self.datasets=load_from_disk("dataset/dataset/data2/amazon_counterfactual")
                self.datasets=load_dataset('SetFit/amazon_counterfactual_en')
        elif self.task_name == 'qnli':
             self.datasets=load_from_disk("dataset/dataset/data2/qnli")
        elif self.task_name== 'dbpedia':
             self.datasets=load_from_disk("dataset/dataset/data2/dbpedia_14")
        elif self.task_name == 'mrpc':
             #self.datasets=load_from_disk("dataset/dataset/data2/mrpc")
               self.datasets=load_dataset('glue','mrpc')
        elif self.task_name == 'rte':
             #self.datasets=load_from_disk("dataset/dataset/data2/rte")
                self.datasets=load_dataset('glue','rte')
        elif self.task_name=='cb':
             self.datasets=load_from_disk("dataset/dataset/super_glue_cb")
        elif self.task_name == 'sst2':
             self.datasets=load_dataset('glue','sst2')
             if self.args.eda ==1:
                if self.args.seed==0:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/eda/eda_sst0.csv'})
                elif self.args.seed==1:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/eda/eda_sst1.csv'})
                elif self.args.seed==2:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/eda/eda_sst2.csv'})
                elif self.args.seed==3:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/eda/eda_sst3.csv'})
                elif self.args.seed==4:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/eda/eda_sst4.csv'})
                else: 
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/eda/eda_sst5.csv'})
                self.datasets['train']=da['train']

             if self.args.bt ==1:
                if self.args.seed==0:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_sst0.csv'})
                elif self.args.seed==1:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_sst1.csv'})
                elif self.args.seed==2:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_sst2.csv'})
                elif self.args.seed==3:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_sst3.csv'})
                elif self.args.seed==4:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_sst4.csv'})
                else:
                   da=load_dataset("csv", data_files={'train':'dataset/dataset/data2/bt/bt_sst5.csv'})
                self.datasets['train']=da['train']
        elif self.task_name== 'yahoo':
             self.datasets=load_from_disk("dataset/dataset/data2/yahoo_answer")
        elif self.task_name=='thunews':
             self.datasets=load_from_disk("thunews")
        elif self.task_name == 'email':
             self.datasets=load_dataset("csv", data_files={'train':'email/train.csv','validation':'email/dev.csv'},encoding='gbk')
        else: # glue task
        # self.datasets = load_dataset("csv", data_files={'train':'SST-2/SST-2/train.tsv','validation':'SST-2/SST-2/dev.tsv'},delimiter="\t")
          #  self.datasets = load_dataset("glue", self.task_name)
            self.datasets=load_from_disk("dataset/dataset/super_glue_cb")
        self.max_length = max_len

        self.sentence1_key, self.sentence2_key, self.train_dataset, self.eval_dataset, \
        self.test_dataset, self.compute_metrics, self.num_labels, self.eval_key = None, None, None, None, \
                                                                                  None, None, None, None

        self.get_label_info()
        self.preprocess_dataset()
        self.get_metric()

    def get_label_info(self):
        # Labels
        if self.task_name == 'trec':
            if self.args.dataset == 'trec-fine':
                label_list = self.datasets["train"].features["label-fine"].names
            elif self.args.dataset == 'trec-coarse':
                label_list = self.datasets["train"].features["label-coarse"].names
            self.num_labels = len(label_list)   # 6
        elif self.args.dataset == 'anli':
            label_list = self.datasets["train_r1"].features["label"].names
            label_list = self.datasets["train_r1"].features["label"].names
            self.num_labels = len(label_list)   # 6
        elif self.args.dataset == 'anli':
            label_list = self.datasets["train_r1"].features["label"].names
            self.num_labels = len(label_list)
        elif self.args.dataset == 'subj' or self.args.dataset == 'amazon' or self.args.dataset == 'yahoo' or self.args.dataset =='email':
             label_list = self.datasets["train"].unique('label')
             self.num_labels = len(label_list)
        elif self.args.eda==1 or self.args.bt ==1:
             label_list = self.datasets["train"].unique('label')
             self.num_labels = len(label_list)
           
        else:
            label_list = self.datasets["train"].features["label"].names
            #label_list = self.datasets["train"].unique('label')
            self.num_labels = len(label_list)

    def preprocess_dataset(self):
        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key],
                                                                                    examples[self.sentence2_key])
            )
            result = self.tokenizer(*args, padding='max_length', max_length=self.max_length, truncation=True)
            return result

        self.sentence1_key, self.sentence2_key = task_to_keys[self.task_name]
        self.datasets = self.datasets.map(preprocess_function, batched=True, load_from_cache_file=True)
        self.datasets.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
        if self.task_name == 'anli':
            self.train_dataset = concatenate_datasets([self.datasets['train_r1'], self.datasets['train_r2'], self.datasets['train_r3']])
        else:
            self.train_dataset = self.datasets['train']

        if self.task_name == 'anli':
            self.eval_dataset = {
                'test_r1': self.datasets['test_r1'],
                'test_r2': self.datasets['test_r2'],
                'test_r3': self.datasets['test_r3'],
                'val_r1': self.datasets['dev_r1'],
                'val_r2': self.datasets['dev_r2'],
                'val_r3': self.datasets['dev_r3'],
            }
        else:
            if self.task_name == 'mnli':
                self.eval_key = 'validation_matched'
            elif self.task_name== 'trec' or self.task_name == 'imdb' or self.task_name=='agnews' or self.task_name=='subj' or self.task_name=='amazon' or self.task_name == 'dbpedia' or self.task_name == 'yahoo':
                self.eval_key = 'test'
            else:
                self.eval_key = 'validation'
            self.eval_dataset = self.datasets[self.eval_key]

    def get_metric(self):
        # Get the metric function
        if self.task_name == 'trec' or self.task_name == 'anli':
            self.compute_metrics = None
            return
       # metric = load_metric("glue", self.task_name)

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            if self.task_name is not None:
                #result = metric.compute(predictions=preds, references=p.label_ids)
                #result={"accuracy": simple_accuracy(predictions=preds, references=p.label_ids)}
                #result={"matthews_correlation":matthews_corrcoef(p.label_ids, preds)}
                result=acc_and_f1(preds=preds,labels=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            else:
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

        self.compute_metrics = compute_metrics
    def get_accuracy(self, preds, label_ids):
        if self.task_name == 'trec' or self.task_name == 'anli':
            predicted = torch.argmax(preds, dim=1)
            correct = (predicted == label_ids).sum()
            total_sample = len(label_ids)
            return float(correct) / total_sample
        return self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))



import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup, set_seed

from classification_model import ClassificationModel
from trainer import Trainer
from read_data import get_data
from read_data.dataset import NLUDataset
import torch.nn.functional as F 

def _convert_cuda(inputs,args, targets=None):
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device=args.device)
        if targets is not None:
            targets = targets.to(device=args.device, non_blocking=True)
        return inputs, targets

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class LabelSmoothingCrossEntropy1(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, device,n_labels,smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy1, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.device=device
        self.n_labels=n_labels
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        num=x.shape[0]
        label = np.zeros((num,self.n_labels))
        label[range(num),target.cpu()] = 1
        smooth_label=F.softmax(x,dim=-1)*self.smoothing+torch.tensor(label).to(self.device)*self.confidence
        nll_loss=-(smooth_label*logprobs).sum(dim=-1)
        return nll_loss.mean()


def run_train(args):
    # load data
    loader = dict()
    train_labeled_set, test_set, n_labels, tokenizer, eval_fn = get_data(args)
    if args.dataset=='email' or args.dataset=='thunews':
           train_labeled_set, test_set, n_labels, tokenizer, eval_fn = get_data(args,max_seq_len=256)
    args.eval_fn = eval_fn
    if args.low_resource==1:
       lisq=dict()
       lisq['input_ids']=[]
       lisq['attention_mask']=[]
       lisq['token_type_ids']=[]
       lisq['label']=[]
       num=args.num
       for k in range(0,n_labels):
           k_sum=0
           for i in train_labeled_set:
              if i[1]==k:
                k_sum=k_sum+1
                lisq['input_ids'].append(i[0]['input_ids'])
                lisq['attention_mask'].append(i[0]['attention_mask'])
                lisq['token_type_ids'].append(i[0]['token_type_ids'])
                lisq['label'].append(i[1])
              if k_sum==num:
                 break
         
       lisq['input_ids']=torch.stack(lisq['input_ids'])
       lisq['attention_mask']=torch.stack(lisq['attention_mask'])
       lisq['token_type_ids']=torch.stack(lisq['token_type_ids'])
       train_labeled_set=NLUDataset(args,lisq,lisq['label'], mode='eval')
       print('train:'+str(len(train_labeled_set)))
    

    loader['labeled_trainloader'] = DataLoader(dataset=train_labeled_set,
                                               batch_size=args.batch_size, shuffle=True)
    if args.dataset == 'anli':
        loader['test_loader'] = dict()
        for key in ["test_r1", 'test_r2', 'test_r3', 'val_r1', 'val_r2', 'val_r3']:
            loader['test_loader'][key] = DataLoader(dataset=test_set[key], batch_size=1024, shuffle=False)
    else:
        loader['test_loader'] = DataLoader(dataset=test_set, batch_size=1024, shuffle=False)

    print("total number of labels: ", n_labels)

    # Configure epoch/step number
    loader_length = len(loader['labeled_trainloader'])
    args.steps = 0
    if loader_length < 500:
        args.checkpoint = loader_length
    else:
        args.checkpoint = 500
    print(f"{args.dataset}: loader-length: {loader_length}, "
          f"total of {(5 * loader_length) // args.checkpoint} checkpoints for aug(*3/5 for normal)")

    # load model
    model = ClassificationModel(pretrained_model=args.pretrained_model, num_labels=n_labels).to(args.device)
    model = nn.DataParallel(model)

    # Load pretrained models for augmentation
    if args.aug_mode == 'normal':
        args.epochs = 3
        assert args.optimizer_lr == 5e-05
        if args.eda==1:
           args.epochs=10
           args.optimizer_lr=1e-05
           checkpoint_name = f"{args.dataset}-{args.seed}"
           checkpoint_name = f'{args.checkpoint_path}/{checkpoint_name}/best.pt'
           model.load_state_dict(torch.load(checkpoint_name, map_location=args.device))
        if args.bt==1:
           args.epochs=10
           args.optimizer_lr=1e-05
           checkpoint_name = f"{args.dataset}-{args.seed}"
           checkpoint_name = f'{args.checkpoint_path}/{checkpoint_name}/best.pt'
           model.load_state_dict(torch.load(checkpoint_name, map_location=args.device))
    else:
        checkpoint_name = f"{args.dataset}-{args.seed}"
        if 'trec' in args.dataset:
            if args.dataset == 'trec-fine':
                checkpoint_name = f"f-trec-{args.seed}"
            else:
                checkpoint_name = f"c-trec-{args.seed}"
        elif args.dataset == 'anli':
            checkpoint_name = f"{args.dataset}-{args.anli_round}-{args.seed}"

        checkpoint_name = f'{args.checkpoint_path}/{checkpoint_name}/best.pt'

        model.load_state_dict(torch.load(checkpoint_name, map_location=args.device))
        args.epochs=5
        if args.evo ==1:
          args.epochs = 10
                        
          score_l=[]
          with torch.no_grad():
             for batch_idx, (inputs, targets, length) in enumerate(loader['labeled_trainloader']):
                inputs, targets = _convert_cuda(inputs,args, targets)
                outputs = model(inputs=inputs)#采用ssmix baseline
                outmax=[]
                for i in range(len(outputs)):
                    al,a=outputs[i].sort(descending=True)
                    if a[0]==targets[i]:
                        outmax.append(a[1].item())
                    else:
                        outmax.append(a[0].item())
                score_l.append((outputs[np.arange(len(outputs)),targets]-outputs[np.arange(len(outputs)),outmax]))
          score_list=[]
          for i in range(len(score_l)):
            for j in range(len(score_l[i])):
                 score_list.append(score_l[i][j].item())
          judge=np.median(np.array(score_list))
          easy_ids,easy_att,easy_token=[],[],[]
          hard_ids,hard_att,hard_token=[],[],[]
          easy_label=[]
          hard_label=[]
          data_easy=dict()
          data_hard=dict()
          with torch.no_grad():
            for batch_idx, (inputs, targets, length) in enumerate(loader['labeled_trainloader']):
                inputs, targets = _convert_cuda(inputs,args, targets)
                outputs = model(inputs=inputs)
                outmax=[]
                for i in range(len(outputs)):
                    al,a=outputs[i].sort(descending=True)
                    if a[0]==targets[i]:
                        outmax.append(a[1].item())
                    else:
                        outmax.append(a[0].item())
                score=outputs[np.arange(len(outputs)),targets]-outputs[np.arange(len(outputs)),outmax]
                for i in range(len(score)):
                    if score[i]>judge:
                        easy_ids.append(inputs['input_ids'][i])
                        easy_att.append(inputs['attention_mask'][i])
                        easy_token.append(inputs['token_type_ids'][i])
                        easy_label.append(targets[i].item())
                        
                    else:
                        hard_ids.append(inputs['input_ids'][i])
                        hard_att.append(inputs['attention_mask'][i])
                        hard_token.append(inputs['token_type_ids'][i])
                        hard_label.append(targets[i].item())
          data_easy['input_ids']=torch.stack(easy_ids).cpu()
          data_easy['attention_mask']=torch.stack(easy_att).cpu()
          data_easy['token_type_ids']=torch.stack(easy_token).cpu()
          data_hard['input_ids']=torch.stack(hard_ids).cpu()
          data_hard['attention_mask']=torch.stack(hard_att).cpu()
          data_hard['token_type_ids']=torch.stack(hard_token).cpu()
          easydata=NLUDataset(args,data_easy,easy_label, mode='eval')
          harddata=NLUDataset(args,data_hard,hard_label, mode='eval')
          loader['labeled_trainloader_easy'] = DataLoader(dataset=easydata,
                                               batch_size=32, shuffle=True)
          loader['labeled_trainloader_hard']=DataLoader(dataset=harddata,
                                               batch_size=32, shuffle=True)


    print(args)
    # Configure optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-4  # 10^-4 good at mixup paper
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.optimizer_lr, eps=1e-8)

    # warmup for 10% of total training step.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=(loader_length * args.epochs) // 10,
                                                num_training_steps=loader_length * args.epochs)
    criterion = nn.CrossEntropyLoss(reduction="none")
    #criterion = LabelSmoothingCrossEntropy1(device=args.device,n_labels=n_labels)
    #criterion = LabelSmoothingCrossEntropy()
    if args.evo == 1:
       #criterion = LabelSmoothingCrossEntropy1(device=args.device,n_labels=n_labels,smoothing=args.smooth)
          criterion = nn.CrossEntropyLoss(reduction="none")
           #criterion = LabelSmoothingCrossEntropy()
    trainer = Trainer(args=args, model=model, optimizer=optimizer, criterion=criterion, loader=loader,
                      n_labels=n_labels, tokenizer=tokenizer, scheduler=scheduler)
    trainer.run_train()


def parse_argument():
    parser = argparse.ArgumentParser(description='train classification model')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased', help='pretrained model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    parser.add_argument('--verbose', action='store_true', help='description T/F for printing out the logs')
    parser.add_argument('--verbose_level', type=int, default=2000,
                        help='description level for printing out the logs - print every n batch')
    parser.add_argument('--verbose_show_augment_example', action='store_true',
                        help='Print out examples of augmented text at first epoch, '
                             'and also print out initial test accuracy')
    parser.add_argument('--seed', type=int, help='Set seed number')
    parser.add_argument('--optimizer_lr', type=float, default=5e-05, help='Set learning rate for optimizer')
    parser.add_argument('--naive_augment', action='store_true', help='Augment without original data')
    parser.add_argument('--dataset', type=str, default='trec', help='Dataset to use')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/checkpoints',
                        help='Directory path to save checkpoint')
    parser.add_argument('--anli_round', type=int, default=1, choices=[1, 2, 3],
                        help='dataset to load for ANLI round.')
    parser.add_argument('--low_resource',type=int,default=0)
    parser.add_argument('--num',type=int)
    parser.add_argument('--evo',type=int,default=0)
    parser.add_argument('--per',type=float,default=1)
    parser.add_argument('--eda',type=int,default=0)
    parser.add_argument('--bt',type=int,default=0)
    parser.add_argument('--smooth',type=float,default=0.1) 
    # Training mode. AUG_MODE must be one of following modes
    subparsers = parser.add_subparsers(title='augmentation', description='Augmentation mode', dest='aug_mode')

    # NORMAL (No augmentation)
    subparsers.add_parser('normal')
    subparsers.default = 'normal'

    # SSMIX
    sp_ss = subparsers.add_parser('ssmix')
    sp_ss.add_argument('--ss_winsize', type=int, default=10,
                       help='Percent of window size. 10 means 10% for augmentation')
    sp_ss.add_argument('--ss_no_saliency', action='store_true',
                       help='Excluding saliency constraint in SSMix')
    sp_ss.add_argument('--ss_no_span', action='store_true',
                       help='Excluding span constraint in SSMix')

    # TMIX
    sp_hidden = subparsers.add_parser('tmix')
    sp_hidden.add_argument('--hidden_alpha', type=float, default=0.2,
                           help='mixup alpha value for l=np.random.beta(alpha, alpha) when getting lambda probability')

    # EMBEDMIX
    sp_embed = subparsers.add_parser('embedmix')
    sp_embed.add_argument('--embed_alpha', type=float, default=0.2,
                          help='mixup alpha value for l=np.random.beta(alpha, alpha) when getting lambda probability')

    # UNK
    sp_unk = subparsers.add_parser('unk')
    sp_unk.add_argument('--unk_winsize', type=float, default=10,
                        help='Percent of window size. 10 means 10% for augmentation')

    args, _ = parser.parse_known_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset configuration
    args.nli_dataset = args.dataset in ['mnli', 'mrpc', 'qqp', 'qnli', 'rte', 'anli','cb']

    return args


def main():
    args = parse_argument()
    set_seed(args.seed)
    run_train(args)


if __name__ == '__main__':
    print(f"\nTrain pipeline start \n"
          f"CUDA available: {torch.cuda.is_available()}, number of GPU: {torch.cuda.device_count()}")
    main()

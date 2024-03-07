import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from augmentation import Augment


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

def cross_entropy(logits, target):
    p = F.softmax(logits, dim=1)
    log_p = -torch.log(p)
    loss = target*log_p
    # print(target,p,log_p,loss)
    batch_num = logits.shape[0]
    return loss.sum()/batch_num

def cos_simi(x,y):
    x=x.float()
    y=y.float()
    num=torch.matmul(x,y.T)
#     denom=np.linalg.norm(x,axis=1).dot(np.linalg.norm(y,axis=1).T)
    norm_x = torch.norm(x)  # 计算向量 x 的范数
    norm_y = torch.norm(y, dim=1)  # 计算向量 y 的范数
    denom= norm_x * norm_y  # 计算余弦相似度
#     denom=(norm_x.unsqueeze(1) * norm_y.unsqueeze(0))
#     denom=np.linalg.norm(x).dot(np.linalg.norm(y,axis=1).T)
    return num / denom

def belong(input,data,device):
    for i in range(len(data)):
        if torch.equal(input,data[i]):
            return True
    return False


def aum(inputs,targets,length,easydata,harddata,device):
    inputid=[]
    inputat=[]
    inputtt=[]
    tarr=[]
    lenr=[]
    inputid1=[]
    inputat1=[]
    inputtt1=[]
    tarr1=[]
    lenr1=[]
    for i in range(len(inputs['input_ids'])):
        input=inputs['input_ids'][i]
        if belong(input, easydata.input_ids,device):
            data=harddata
        elif belong(input,harddata.input_ids,device):
            data=easydata
        coss=cos_simi(input,data.input_ids)
        coss=coss.cpu()
#     coss-=np.eye(coss.shape[0])
        inputid.append(data.input_ids[np.argmax(coss)])
        inputat.append(data.attn_mask[np.argmax(coss)])
        inputtt.append(data.ttids[np.argmax(coss)])
        tarr.append(torch.tensor(data.label_list[np.argmax(coss)]))
        lenr.append(data.tokenized_length[np.argmax(coss)])
        inputid1.append(data.input_ids[np.argmin(coss)])
        inputat1.append(data.attn_mask[np.argmin(coss)])
        inputtt1.append(data.ttids[np.argmax(coss)])
        tarr1.append(torch.tensor(data.label_list[np.argmin(coss)]))
        lenr1.append(data.tokenized_length[np.argmin(coss)])
    input_right=dict()
    input_right['input_ids']=torch.stack(inputid)
    input_right['attention_mask']=torch.stack(inputat)
    input_right['token_type_ids']=torch.stack(inputtt)
    target_right=torch.stack(tarr)
    length_right=torch.stack(lenr)
    input_right1=dict()
    input_right1['input_ids']=torch.stack(inputid1)
    input_right1['attention_mask']=torch.stack(inputat1)
    input_right1['token_type_ids']=torch.stack(inputtt1)
    target_right1=torch.stack(tarr1)
    length_right1=torch.stack(lenr1)
#         input_right['input_ids']=torch.stack([data.input_ids[np.argmax(row)] for num_row,row in enumerate(coss)])
#         input_right['attention_mask']=torch.stack([data.attn_mask[np.argmax(row)] for num_row,row in enumerate(coss)])
#         input_right['token_type_ids']=torch.stack([data.ttids[np.argmax(row)] for num_row,row in enumerate(coss)])
#         target_right=torch.stack([torch.tensor(data.label_list[np.argmax(row)]) for num_row, row in enumerate(coss)])
#         length_right=torch.stack([data.tokenized_length[np.argmax(row)] for num_row, row in enumerate(coss)])
#         inputr.append(input)
    return (inputs,targets,length,input_right,target_right,length_right),(inputs,targets,length,input_right1,target_right1,length_right1)
        


#def cos_simi(x,y):
#    num=torch.matmul(x,y.T)
#    denom=np.linalg.norm(x,axis=1).dot(np.linalg.norm(y,axis=1).T)
#    return num / denom
def get_similarity(inputs,targets,length):
    input=inputs['input_ids']
    coss=cos_simi(input,input)
    coss-=np.eye(coss.shape[0])
    input_right=dict()
    for key in inputs:
        input_right[key]=torch.stack([inputs[key][np.argmax(row)] for num_row,row in enumerate(coss)])
    target_right=torch.stack([targets[np.argmax(row)] for num_row, row in enumerate(coss)])
    length_right=torch.stack([length[np.argmax(row)] for num_row, row in enumerate(coss)])
    return inputs,targets,length,input_right,target_right,length_right

def get_similarity_num(inputs,targets,length,num):
    input=inputs['input_ids']
    coss=cos_simi(input,input)
    coss-=np.eye(coss.shape[0])
    input_right=dict()
    #for key in inputs:
    #    input_right[key]=torch.stack([inputs[key][np.argmax(row)] for num_row,row in enumerate(coss)])
    #target_right=torch.stack([targets[np.argmax(row)] for num_row, row in enumerate(coss)])
    #length_right=torch.stack([length[np.argmax(row)] for num_row, row in enumerate(coss)])
    num_right=torch.stack([num[np.argmax(row)] for num_row, row in enumerate(coss)])
    return num,num_right

class Trainer:

    def __init__(self, args, model, optimizer, criterion, loader, n_labels, tokenizer, scheduler):
        self.args = args
        self.args.model = model
        self.args.optimizer = optimizer
        self.args.criterion = criterion
        self.loader = loader
        self.n_labels = n_labels
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.augment = Augment(self.args)

        self.best_acc = 0
        self.init_acc = 0

    def _convert_cuda(self, inputs, targets=None):
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device=self.args.device)
        if targets is not None:
            targets = targets.to(device=self.args.device, non_blocking=True)
        return inputs, targets

    def _get_loss(self, inputs1, targets1, targets2=None, ratio=None, **kwargs):
        output = self.args.model(inputs=inputs1, **kwargs)
        ll=[]
        l2=[]
        for i in range(0,len(output)):
            ll.append(output[i][0].item())
            l2.append(output[i][1].item())
        pdi=pd.DataFrame()
        pdi['yi']=ll
        pdi['er']=l2
        pdi.to_csv('test1.csv')
       # if self.args.tree == 1:
        #   loss =cross_entropy(output,targets1)
        #else:
        loss = self.args.criterion(output, targets1)
        if targets2 is not None:
            loss = loss * ratio + self.args.criterion(output, targets2) * (1 - ratio)
        loss = loss.mean().float()
        return loss

    def _report_and_forward(self, batch_idx, epoch, loss):
        self.args.steps += 1
        if self.args.verbose and (self.args.steps % self.args.checkpoint) == 0:
            print('epoch {}, step {}, loss {}, best_acc {}'.format(epoch, batch_idx, loss.item(), self.best_acc))

        torch.nn.utils.clip_grad_norm_(self.args.model.parameters(), 1)
        self.args.optimizer.zero_grad()
        loss.backward()
        self.args.optimizer.step()
        self.scheduler.step()
        if self.args.steps % self.args.checkpoint == 0:
            self.validate_and_save(epoch)

    def _split_labeled_batch(self, data):
        inputs, targets, lengths = data
        if len(inputs['input_ids']) % 2 != 0:
            # Skip odd-numbered batch
            return None, None, None, None, None, None
        inputs_left, inputs_right = dict(), dict()

        # To handle case of leftovers, we should not set split_size to half of BATCH size but as half of INPUT size.
        half_size = len(inputs['input_ids']) // 2
        for key in inputs.keys():
            inputs_left[key], inputs_right[key] = torch.split(inputs[key], half_size)

        targets_left, targets_right = torch.split(targets, half_size)
        length_left, length_right = torch.split(lengths, half_size)

        return inputs_left, targets_left, length_left, inputs_right, targets_right, length_right

    def show_augment_example(self,
                             inputs_left, targets_left,
                             inputs_right, targets_right,
                             inputs_aug_left, ratio_left,
                             inputs_aug_right, ratio_right,
                             length_left, length_right):
        def reconstruct_text(input_ids):
            out = []
            for input_id in input_ids:
                pad_id = 0
                pad_removed = [x for x in input_id if x != pad_id]
                out.append(self.tokenizer.decode(pad_removed))
            return out

        left_text, right_text, aug_left_text, aug_right_text = [reconstruct_text(x['input_ids']) for x in [
            inputs_left, inputs_right, inputs_aug_left, inputs_aug_right]]

        print("\n\nShowing length distribution ...")
        print(f"{length_left} \n& {length_right}\n->{inputs_aug_left['attention_mask'].sum(dim=1)}",
              f"\n->{inputs_aug_right['attention_mask'].sum(dim=1)}")
        print("\n\nShowing augment example ...")

        for idx, (l_t, r_t, a_l_t, a_r_t, t_l, t_r, r_l, r_r) in enumerate(zip(left_text, right_text, aug_left_text,
                                                                               aug_right_text, targets_left,
                                                                               targets_right,
                                                                               ratio_left, ratio_right)):
            if t_l.shape == torch.Size([]) and t_r.shape == torch.Size([]):
                t_l = t_l.item()
                t_r = t_r.item()
            print(f"Example #{idx}")
            print(
                f"<<left_text (label {t_l})>>\n->{l_t}\n<<right_text (label {t_r})>>\n->{r_t}")
            print(f"<<left_mixup of ratio {r_l}>>\n->{a_l_t}\n<<right_mixup of ratio {r_r}>>\n->{a_r_t}\n")

    def train_augment(self, epoch, **kwargs):
        self.args.model.train()
        print(f"TRAIN_{self.args.aug_mode}: epoch {epoch} train start")

        for batch_idx, data in enumerate(self.loader['labeled_trainloader']):
            # prepare data
            splitted = self._split_labeled_batch(data)
            inputs_left, targets_left, length_left, inputs_right, targets_right, length_right = splitted
            if inputs_left is None:  # data was odd - skip this batch
                continue

            inputs_left, targets_left = self._convert_cuda(inputs_left, targets_left)
            inputs_right, targets_right = self._convert_cuda(inputs_right, targets_right)

            loss_list = []
            if not self.args.naive_augment:
                loss_list.append(self._get_loss(inputs_left, targets_left))
                loss_list.append(self._get_loss(inputs_right, targets_right))

            inputs_aug_left, ratio_left, inputs_aug_right, ratio_right = self.augment(*splitted, **kwargs)
            loss_list.append(self._get_loss(inputs1=inputs_aug_left,
                                            targets1=targets_left,
                                            targets2=targets_right,
                                            ratio=ratio_left))
            loss_list.append(self._get_loss(inputs1=inputs_aug_right,
                                            targets1=targets_right,
                                            targets2=targets_left,
                                            ratio=ratio_right))

            self._report_and_forward(batch_idx, epoch, torch.mean(torch.stack(loss_list)))
            if batch_idx == 0 and (inputs_aug_left is not None):
                if self.args.verbose_show_augment_example:  # show augment example at the first step
                    self.show_augment_example(inputs_left, targets_left,
                                              inputs_right, targets_right,
                                              inputs_aug_left, ratio_left,
                                              inputs_aug_right, ratio_right,
                                              length_left, length_right)

        print(f"epoch {epoch} train end")



    def train_augment1(self, epoch, **kwargs):
        self.args.model.train()
        print(f"TRAIN_{self.args.aug_mode}: epoch {epoch} train start")
#         if epoch<5:
        for batch_idx, data in enumerate(self.loader['labeled_trainloader_easy']):
            # prepare data
            inputs,targets,length=data
            #splitted = self._split_labeled_batch(data)
            splitted = get_similarity(inputs,targets,length)
            inputs_left, targets_left, length_left, inputs_right, targets_right, length_right = splitted
            if inputs_left is None:  # data was odd - skip this batch
                continue
            inputs_left, targets_left = self._convert_cuda(inputs_left, targets_left)
            inputs_right, targets_right = self._convert_cuda(inputs_right, targets_right)
            loss_list = []
            loss_list.append(self._get_loss(inputs_left, targets_left))
            #loss_list.append(self._get_loss(inputs_right, targets_right))
            inputs_aug_left, ratio_left, inputs_aug_right, ratio_right = self.augment(*splitted, **kwargs)
            loss_list.append(self._get_loss(inputs1=inputs_aug_left,
                                            targets1=targets_left,
                                            targets2=targets_right,
                                            ratio=ratio_left))
            #loss_list.append(self._get_loss(inputs1=inputs_aug_right,
             #                               targets1=targets_right,
              #                              targets2=targets_left,
               #                             ratio=ratio_right))

            self._report_and_forward(batch_idx, epoch, torch.mean(torch.stack(loss_list)))
#         else:
        for batch_idx, data in enumerate(self.loader['labeled_trainloader_hard']):
            # prepare data
            inputs,targets,length=data
            #splitted = self._split_labeled_batch(data)
            splitted = get_similarity(inputs,targets,length)
            inputs_left, targets_left, length_left, inputs_right, targets_right, length_right = splitted
            if inputs_left is None:  # data was odd - skip this batch
                continue
            inputs_left, targets_left = self._convert_cuda(inputs_left, targets_left)
            inputs_right, targets_right = self._convert_cuda(inputs_right, targets_right)
            loss_list = []
            loss_list.append(self._get_loss(inputs_left, targets_left))
            #loss_list.append(self._get_loss(inputs_right, targets_right))
            inputs_aug_left, ratio_left, inputs_aug_right, ratio_right = self.augment(*splitted, **kwargs)
            loss_list.append(self._get_loss(inputs1=inputs_aug_left,
                                            targets1=targets_left,
                                            targets2=targets_right,
                                            ratio=ratio_left))
            #loss_list.append(self._get_loss(inputs1=inputs_aug_right,
             #                               targets1=targets_right,
              #                              targets2=targets_left,
               #                             ratio=ratio_right))

            self._report_and_forward(batch_idx, epoch, torch.mean(torch.stack(loss_list)))
        print(f"epoch {epoch} train end")
    def train_hidden(self, epoch):
        self.args.model.train()
        print(f"TRAIN_{self.args.aug_mode}, epoch {epoch} start!")

        for batch_idx, data in enumerate(self.loader['labeled_trainloader']):
            if self.args.aug_mode == 'tmix':
                mix_layer = random.choice([7, 9, 12])
                mix_layer = mix_layer - 1
                mix_embedding = False
                l = np.random.beta(self.args.hidden_alpha, self.args.hidden_alpha)  # experimenting with 0.2 and 0.4
                l = max(l, 1 - l)  # lambda
            elif self.args.aug_mode == 'embedmix':
                mix_embedding = True
                mix_layer = -1
                l = np.random.beta(self.args.embed_alpha, self.args.embed_alpha)  # experimenting with 0.2 and 0.4
                l = max(l, 1 - l)  # lambda
            else:
                raise RuntimeError('Invalid mixup')

            i_1, t_1, l_1, i_2, t_2, l_2 = self._split_labeled_batch(data)
            if i_1 is None:
                continue

            i_1, t_1 = self._convert_cuda(i_1, t_1)
            i_2, t_2 = self._convert_cuda(i_2, t_2)
            loss_list = []

            # get orig data loss
            if not self.args.naive_augment:
                loss_list.append(self._get_loss(inputs1=i_1, targets1=t_1))
                loss_list.append(self._get_loss(inputs1=i_2, targets1=t_2))

            loss_list.append(
                self._get_loss(inputs1=i_1, targets1=t_1, targets2=t_2, ratio=l, inputs2=i_2, mixup_lambda=l,
                               mixup_layer=mix_layer, mix_embedding=mix_embedding))
            loss_list.append(
                self._get_loss(inputs1=i_2, targets1=t_2, targets2=t_1, ratio=l, inputs2=i_1, mixup_lambda=l,
                               mixup_layer=mix_layer, mix_embedding=mix_embedding))
            self._report_and_forward(batch_idx, epoch, torch.mean(torch.stack(loss_list)))

        print(f"epoch {epoch} train end")


    def train_hidden1(self, epoch):
        self.args.model.train()
        print(f"TRAIN_{self.args.aug_mode}, epoch {epoch} start!")

        for batch_idx, data in enumerate(self.loader['labeled_trainloader']):
            if self.args.aug_mode == 'tmix':
                mix_layer = random.choice([7, 9, 12])
                mix_layer = mix_layer - 1
                mix_embedding = False
                l = np.random.beta(self.args.hidden_alpha, self.args.hidden_alpha)  # experimenting with 0.2 and 0.4
                l = max(l, 1 - l)  # lambda
            elif self.args.aug_mode == 'embedmix':
                mix_embedding = True
                mix_layer = -1
                l = np.random.beta(self.args.embed_alpha, self.args.embed_alpha)  # experimenting with 0.2 and 0.4
                l = max(l, 1 - l)  # lambda
            else:
                raise RuntimeError('Invalid mixup')
            inputs,targets,length=data
            splitted = get_similarity(inputs,targets,length)
            #splitted,splitted1=aum(inputs,targets,length,self.args.easy,self.args.hard,self.args.device)
            i_1, t_1, l_1, i_2, t_2, l_2 = splitted
            #i_11,t_11,l_11,i_21,t_21,l_21=splitted1
            if i_1 is None:
                continue

            i_1, t_1 = self._convert_cuda(i_1, t_1)
            i_2, t_2 = self._convert_cuda(i_2, t_2)
            #i_11, t_11 = self._convert_cuda(i_11, t_11)
            #i_21, t_21 = self._convert_cuda(i_21, t_21)

            loss_list = []
            # get orig data loss
            if not self.args.naive_augment:
                loss_list.append(self._get_loss(inputs1=i_1, targets1=t_1))
                loss_list.append(self._get_loss(inputs1=i_2, targets1=t_2))

            loss_list.append(
                self._get_loss(inputs1=i_1, targets1=t_1, targets2=t_2, ratio=l, inputs2=i_2, mixup_lambda=l,
                               mixup_layer=mix_layer, mix_embedding=mix_embedding))
            #loss_list.append(
             #   self._get_loss(inputs1=i_11, targets1=t_11, targets2=t_21, ratio=l, inputs2=i_21, mixup_lambda=l,
              #                 mixup_layer=mix_layer, mix_embedding=mix_embedding))
            loss_list.append(
                self._get_loss(inputs1=i_2, targets1=t_2, targets2=t_1, ratio=l, inputs2=i_1, mixup_lambda=l,
                               mixup_layer=mix_layer, mix_embedding=mix_embedding))

            self._report_and_forward(batch_idx, epoch, torch.mean(torch.stack(loss_list)))


        print(f"epoch {epoch} train end")


    def train_normal(self, epoch):
        self.args.model.train()
        print(f"TRAIN_NORMAL: epoch {epoch} train start")
        if self.args.tree==1:
           #ed=pd.DataFrame()
           #hd=pd.DataFrame()
           for batch_idx, data in enumerate(self.loader['labeled_trainloader_easy']):
               inputs, targets, length= data
               tensor_list = []
               for item in targets:
                   item = item.strip('[]')  # 去除方括号
                   numbers = np.fromstring(item, sep=' ')  # 将字符串转换为NumPy数组
                   tensor = torch.from_numpy(numbers)  # 将NumPy数组转换为PyTorch的tensor类型
                   tensor_list.append(tensor)
               result_tensor = torch.stack(tensor_list)  # 将tensor列表堆叠为一个整体的tensor
               targets=result_tensor
               inputs, targets = self._convert_cuda(inputs, targets)
               loss = self._get_loss(inputs, targets)

               self._report_and_forward(batch_idx, epoch, loss)

        
               #inputs, targets = self._convert_cuda(inputs, targets)
              # splitted = get_similarity_num(inputs,targets,length,num)
               #num_left,num_right=splitted
               #k1,k2=task_to_keys[self.args.dataset]
               #ed[k1]=self.args.orida['train'][num_left][k1]+self.args.orida['train'][num_right][k1]
               #ed['parsing1']=self.args.orida['train'][num_left]['parsing1']+self.args.orida['train'][num_right]['parsing1']
               #if k2 is not None:
                  # ed[k2]=self.args.orida['train'][num_left][k2]+self.args.orida['train'][num_right][k2]
                  # ed['parsing2']=self.args.orida['train'][num_left]['parsing2']+self.args.orida['train'][num_right]['parsing2']
              # ed['label']=self.args.orida['train'][num_left]['label']+self.args.orida['train'][num_right]['label']
           #ed.to_csv('easy'+self.args.dataset+str(self.args.seed)+'.csv')
           for batch_idx, data in enumerate(self.loader['labeled_trainloader_hard']):
               #inputs, targets, length, num= data
               inputs, targets, length= data
               tensor_list = []
               for item in targets:
                   item = item.strip('[]')  # 去除方括号
                   numbers = np.fromstring(item, sep=' ')  # 将字符串转换为NumPy数组
                   tensor = torch.from_numpy(numbers)  # 将NumPy数组转换为PyTorch的tensor类型
                   tensor_list.append(tensor)
               result_tensor = torch.stack(tensor_list)  # 将tensor列表堆叠为一个整体的tensor
               targets=result_tensor
               inputs, targets = self._convert_cuda(inputs, targets)
               loss = self._get_loss(inputs, targets)

               self._report_and_forward(batch_idx, epoch, loss)
               #inputs, targets = self._convert_cuda(inputs, targets)
               #splitted = get_similarity_num(inputs,targets,length,num)
               #num_left,num_right=splitted
               #k1,k2=task_to_keys[self.args.dataset]
               #hd[k1]=self.args.orida['train'][num_left][k1]+self.args.orida['train'][num_right][k1]
               #hd['parsing1']=self.args.orida['train'][num_left]['parsing1']+self.args.orida['train'][num_right]['parsing1']
               #if k2 is not None:
                  # hd[k2]=self.args.orida['train'][num_left][k2]+self.args.orida['train'][num_right][k2]
                  # hd['parsing2']=self.args.orida['train'][num_left]['parsing2']+self.args.orida['train'][num_right]['parsing2']
              # hd['label']=self.args.orida['train'][num_left]['label']+self.args.orida['train'][num_right]['label']
           #hd.to_csv('hard'+self.args.dataset+str(self.args.seed)+'.csv')

        else: 
          for batch_idx, data in enumerate(self.loader['labeled_trainloader']):
            inputs, targets, length = data
            if self.args.tree==1:
               tensor_list = []
               for item in targets:
                   item = item.strip('[]')  # 去除方括号
                   numbers = np.fromstring(item, sep=' ')  # 将字符串转换为NumPy数组
                   tensor = torch.from_numpy(numbers)  # 将NumPy数组转换为PyTorch的tensor类型
                   tensor_list.append(tensor)
               result_tensor = torch.stack(tensor_list)  # 将tensor列表堆叠为一个整体的tensor
               targets=result_tensor

            inputs, targets = self._convert_cuda(inputs, targets)
            loss = self._get_loss(inputs, targets)

            self._report_and_forward(batch_idx, epoch, loss)

        print(f"epoch {epoch} train end")

    def validate(self, loader, mode=''):
        self.args.model.eval()
        preds, label_ids, loss_total, total_sample = None, None, 0, 0

        with torch.no_grad():
            for batch_idx, (inputs, targets, length) in enumerate(loader):
                inputs, targets = self._convert_cuda(inputs, targets)

                outputs = self.args.model(inputs=inputs)
                #cre=torch.nn.CrossEntropyLoss(reduction='none')
                loss = torch.mean(self.args.criterion(outputs, targets))
                preds = outputs if preds is None else nested_concat(preds, outputs, dim=0)
                label_ids = targets if label_ids is None else nested_concat(label_ids, targets, dim=0)

                loss_total += loss.item() * len(inputs['input_ids'])
                total_sample += inputs['input_ids'].shape[0]

            if not ('trec' in self.args.dataset or 'anli' in self.args.dataset):
                preds, label_ids = nested_numpify(preds), nested_numpify(label_ids)

            val = self.args.eval_fn(preds, label_ids)

            if 'trec' in self.args.dataset or self.args.dataset == 'anli':
                acc_total = val
                print(f"{mode} acc: {acc_total}")
            else:
                if self.args.dataset == 'cola':
                   acc_total = val['matthews_correlation']
                else:
                   acc_total = val['accuracy']

            loss_total = float(loss_total) / total_sample

        if mode == 'Initial':
            self.init_acc = acc_total
        else:
            self.best_acc = max(self.best_acc, acc_total)

        if not self.args.dataset == 'anli':
            print(f"Init_acc {self.init_acc}, Best_acc: {self.best_acc}, "
                  f"better than init {self.init_acc <= self.best_acc}")
        return loss_total, acc_total

    def run_train(self):
        # validate for first loaded model
        if self.args.verbose_show_augment_example and self.args.aug_mode != 'normal' and self.args.dataset != 'anli':
            test_loss, test_acc = self.validate(loader=self.loader['test_loader'], mode="Initial")
            print(f"Initial test, test acc {test_acc}")

        for epoch in range(self.args.epochs):
            if self.args.aug_mode == 'normal':
                self.train_normal(epoch=epoch)
            #elif self.args.aug_mode == 'normal' and self.args.eda ==1:
             #     self.train_normal1(epoch=epoch)
            elif self.args.aug_mode in ['ssmix', 'unk'] and self.args.evo ==0:
                self.train_augment(epoch=epoch)
            elif self.args.aug_mode == 'ssmix' and self.args.evo ==1:
                self.train_augment1(epoch=epoch)
            elif self.args.aug_mode in ['tmix', 'embedmix'] and self.args.evo ==1:
                self.train_hidden1(epoch=epoch)
            elif self.args.aug_mode in ['tmix', 'embedmix'] and self.args.evo ==0:
                self.train_hidden(epoch=epoch)
            else:
                raise NotImplementedError('Invalid augmentation mode')
        if self.args.checkpoint == 500:
            # validate for last epoch ONLY IF checkpoint is bigger than 500.
            # Otherwise it is already checkpointed for last epoch, so no need.
            self.validate_and_save(epoch)

    def validate_and_save(self, epoch):
        print(f"validating step {self.args.steps}, epoch {epoch}")
        if self.args.dataset == 'anli':
            val = self.best_acc
            val_name, test_name = f'val_r{self.args.anli_round}', f'test_r{self.args.anli_round}'
            val_loss, val_acc = self.validate(loader=self.loader['test_loader'][val_name],
                                              mode=f"{val_name} stats")
            self.best_acc = max(val, val_acc)
            if val < self.best_acc:
                test_loss, test_acc = self.validate(loader=self.loader['test_loader'][test_name],
                                                    mode=f'{test_name} stats')
        else:
            val_loss, val_acc = self.validate(loader=self.loader['test_loader'], mode="Test Stats")

        # save example: mrpc0-500, mrpc2-1500 ..
        if self.args.aug_mode == 'normal' and val_acc == self.best_acc and self.args.eda==0 and self.args.bt==0 and self.args.tree==0:  # save checkpoint for only normal
            self._save_model('best')

       # if self.args.aug_mode == 'ssmix' and val_acc == self.best_acc :
        #    self._save_model('best')   

    def _save_model(self, mode='best'):
        if self.args.dataset == 'anli':
            checkpoint = f"{self.args.dataset}-{self.args.anli_round}-{self.args.seed}"
        elif 'trec' in self.args.dataset:
            checkpoint = 'f-' if self.args.dataset == 'trec-fine' else 'c-'  # coarse, fine
            checkpoint = checkpoint + f'trec-{self.args.seed}'
        elif  self.args.aug_mode=='ssmix':
            checkpoint = f"{self.args.dataset}-{self.args.seed}-{self.args.aug_mode}-{self.args.evo}"             
        else:
            checkpoint = f"{self.args.dataset}-{self.args.seed}"

        print(f"Saving {checkpoint}/{mode} ...")
        checkpoint_path = os.path.join(self.args.checkpoint_path, checkpoint, f'{mode}.pt')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.args.model.state_dict(), checkpoint_path)


def nested_concat(tensors, new_tensors, dim=0):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, dim) for t, n in zip(tensors, new_tensors))
    return torch.cat((tensors, new_tensors), dim=dim)


def nested_numpify(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    return tensors.cpu().numpy()

import torch
import numpy as np
import random
import torch.nn.functional as F
import math
import pdb

def load_pretrained_weights(model, pretrained_path):

    
    if pretrained_path == None or len(pretrained_path)<1:
        
        return model
    print(f"checkpoint :{pretrained_path}")
    
    pretrained_dict = torch.load(pretrained_path,map_location=torch.device('cpu'))

    model_dict = model.state_dict()

    for k, v in model_dict.items():
        if k in pretrained_dict:
            model_dict[k]=pretrained_dict[k]
        else:
            print(f'{k}不匹配')

    model.load_state_dict(model_dict)
    
    return model

class AverageMeter:
    def __init__(self) -> None:
        self.total = 0
        self.avg = 0
        self.nums = 0
    
    def update(self,m,nums):
        self.total += m
        self.nums += nums
        
        if self.nums!=0:
            self.avg = self.total/self.nums
def calculate_perplexity(pred, tag, masks):

    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.float)
    if not isinstance(tag, torch.Tensor):
        tag = torch.tensor(tag, dtype=torch.long)

    log_probs = F.log_softmax(pred, dim=-1)


    nll_loss = F.nll_loss(log_probs, tag, reduction='none')
    
    idx = masks==1

    
    nll_loss = nll_loss[idx]
    if nll_loss.shape[0]>1:
        nll_loss = nll_loss.mean()

        ppl = torch.exp(nll_loss).item()
    else:
        ppl = torch.tensor(0.)
    
    return ppl

int2bin={}
for n in range(256):
    s = format(n, '08b')
    tmp = []
    for c in s:
        if c=='1':
            tmp.append(1)
        elif c=='0':
            tmp.append(0)
        else:
            print('int2bin err!')
    int2bin[n] = tmp
    


def string_to_binary_array(s):

    binary_data = s.encode('utf-8')

    binary_array = []

    for byte in binary_data:

        binary_array += int2bin[byte]
    return binary_array
        
def convert_tokens_to_string(tokens):

        tokens = bytes(tokens)
        while tokens:
            try:
                tokens = tokens.decode("utf-8")
                return tokens

            except UnicodeDecodeError:

                tokens = tokens[:-1]
        return ""  




#############timm#################
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    
def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:

        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)
    
def adaptive_clip_grad(parameters, clip_factor=0.01, eps=1e-3, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is None:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()
        max_norm = unitwise_norm(p_data, norm_type=norm_type).clamp_(min=eps).mul_(clip_factor)
        grad_norm = unitwise_norm(g_data, norm_type=norm_type)
        clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
        new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
        p.grad.detach().copy_(new_grads)
def dispatch_clip_grad(parameters, value: float, mode: str = 'norm', norm_type: float = 2.0):

    if mode == 'norm':
        torch.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)
    elif mode == 'value':
        torch.nn.utils.clip_grad_value_(parameters, value)
    elif mode == 'agc':
        adaptive_clip_grad(parameters, value, norm_type=norm_type)
    else:
        assert False, f"Unknown clip mode ({mode})."

class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
            self,
            loss,
            optimizer,
            clip_grad=None,
            clip_mode='norm',
            parameters=None,
            create_graph=False,
            need_update=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if need_update:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  
                dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

######################################


###############Multiple CrossEntropyLoss##########################
criterion = torch.nn.CrossEntropyLoss(reduction='none')


vector_space = torch.eye(256+3)
for x in range(10):
    idx = ord(str(x)) 
    
    for N in range(10):
        y = math.exp(-(x-N)**2/1.5)
        idx_N = ord(str(N))
        vector_space[idx][idx_N] = y
    vector_space[idx] /= vector_space[idx][48:48+10].sum()
vector_space = vector_space.contiguous()


def calc_reg_loss(pred,target,type,loss_reg,total):
    
    if type == 0:
        idx = pred>1
        formula = lambda x: (x-1).exp()
    elif type ==1:
        idx = (pred>=0)&(pred<=1)
        formula = lambda x: x**2
    elif type ==2:
        idx = (pred>=-1)&(pred<0)
        formula = lambda x: -1*x**2
    else:
        idx = pred<-1
        formula = lambda x: -1*(-(x+1)).exp()
    pred = pred[idx]
    if pred.numel()<1:

        return loss_reg,total

    target = target[idx]
    # pred = pred[idx]
    pred = formula(pred)
    loss = (target-pred)**2
    # loss = loss.mean()
    loss_reg = loss_reg+loss.sum()
    total = total+loss.numel()

    return loss_reg,total


def cross_entropy_multiple_Loss(pred,pred_reg ,target_id,masks,label_positions,target_reg,num_tokens_masks):
    
  

    loss = criterion(pred.permute(0, 2, 1), target_id)

    length = (masks>-1).sum()


    loss_nums = []

    pred_logits = pred[0].softmax(dim=-1)# pred shape=batch,length,dim
    target_logits = vector_space[target_id[0].cpu()].to(pred_logits.device) 

    mask1 = masks==0
    loss_lm = loss[mask1].mean()
    

    loss_label = []

    num_tokens_masks[...] = False 
    for i, (start, end) in enumerate(label_positions):
        start, end = start-1, end-1 
        if start<=0 and end<=0:
            continue
        if start<0:
            start = 0
        if end>loss.shape[-1]:
            end = loss.shape[-1]

        num_tokens_masks[:,start] = True  


        num_tensor_pred = pred_logits[start:end,:].to(pred.device).float()
        num_tensor_tag = target_logits[start:end,:].to(pred.device).float()
        

        kl_loss = F.kl_div(torch.log(num_tensor_pred), num_tensor_tag, reduction='none')
        kl_loss = kl_loss.sum(dim=-1,keepdim=True)

        weight = torch.arange(1,end-start+1).flip(dims=[0])
        weight = weight.unsqueeze(-1).to(kl_loss.device)

        weight = weight/weight.sum(dim=0)
        kl_loss = (kl_loss*weight).sum()
        
        
        loss_label.append(kl_loss)
        
            



    if len(loss_label)>0:

        loss_label = torch.stack(loss_label).mean()
    else:
        loss_label = 0
        print(f"########{label_positions}")
    

    loss = loss_lm*0.2+loss_label*0.8
   

    target_reg = target_reg[num_tokens_masks]
    pred_reg = pred_reg.squeeze(-1)[num_tokens_masks]


    formula = lambda x: x**0.5
    target_reg = torch.where(target_reg>=0,formula(target_reg),target_reg)

    formula = lambda x: -x.abs()**0.5
    target_reg = torch.where(target_reg<0,formula(target_reg),target_reg)


    
    loss_reg = ((target_reg-pred_reg)**2).mean()
    length2 = target_reg.numel()



    return loss,length,loss_reg,length2



if __name__ =="__main__":

    print(111)
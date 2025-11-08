

import torch
from torch.nn import Module, Linear, Dropout, LayerNorm, Identity
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import config as cfg 
from torch.utils.checkpoint import checkpoint  
import random

def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)   
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()   
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ChannelNorm(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.norm = LayerNorm(d_model)
        
    def forward(self,x):
        
        x = self.norm(x.transpose(-2,-1))
        return x.transpose(-2,-1)



class AttentionPatch(Module):

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = (head_dim) ** -0.5
        

     
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1,padding=0,bias=False)
        self.attn_drop = Dropout(attention_dropout)
        
        self.proj = nn.Conv2d(dim, dim, kernel_size=1,padding=0)
        self.proj_drop = Dropout(projection_dropout)

       
        
    def attn_softmax_mask(self,attn,mask,dim=-1):

        mask_value = -torch.finfo(attn.dtype).max

        attn.masked_fill_(~mask,mask_value)

        attn = attn.softmax(dim=dim)
        
        
        return attn
    
   
        
    def forward(self, x,mask):
        B,D,G,P = x.shape

        x = self.qkv(x)

        qkv = rearrange(x,'b (qkv hnum hdim) group patch -> qkv (b hnum ) group patch hdim',qkv=3,hnum=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_softmax_mask(attn,mask,dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v 
        x = rearrange(x,'(b hnum ) group patch hdim -> b (hnum hdim) group patch',b=B)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class BLOCK(Module):
    def __init__(self, d_model,nhead, dim_feedforward=512, dropout=0.1,
                drop_path_rate=0.1,attention_dropout=0.1):
        super(BLOCK, self).__init__()

        self.s1 = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.s2 = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.pre_norm = ChannelNorm(d_model)
        self.norm1 = ChannelNorm(d_model)
        self.self_attn = AttentionPatch(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)
        self.dropout_cnn = Dropout(0.1)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()
        self.activation = F.gelu
        self.cov1_delegate = torch.nn.Conv2d(in_channels=d_model,out_channels=d_model,kernel_size=1,padding=0,groups=1,bias=True)
        kernel_size = 33
        self.cov3 = torch.nn.Conv1d(in_channels=d_model,out_channels=dim_feedforward,kernel_size=kernel_size,padding=kernel_size-1,groups=1)
        self.padding = kernel_size-1
        self.cov4 = torch.nn.Conv1d(in_channels=dim_feedforward,out_channels=d_model,kernel_size=1,padding=0)
     
     
    def mc_attention(self,x,mask,patch_size):
        
        B,D,L = x.shape
        x = rearrange(x,'B D (G P) -> B D G P',P=patch_size)
        x_delegate = self.cov1_delegate(x)
        x_delegate = x_delegate.permute(0,1,3,2).reshape(B,D,-1,patch_size)
        x_delegate = torch.cat([x,x_delegate],dim=-1) 
        x_delegate = self.self_attn(x_delegate,mask)
        x = x_delegate[...,:patch_size]
        x_delegate = x_delegate[...,-patch_size:].reshape(B,D,patch_size,-1).permute(0,1,3,2)
        x = (x+x_delegate)*0.5
        x = x.flatten(-2,-1)

        


        return x

  

    def forward(self, src,mask,patch_size,*args, **kwargs):

        
        B,D,L= src.shape  
        x = self.pre_norm(src)
        x = self.mc_attention(x,mask,patch_size)
        src = src + self.drop_path(x)
 
        src = self.norm1(src)
        src = self.dropout_cnn(src)
        x = self.cov3(src)[...,:-self.padding]
        x = self.activation(x)
        x = self.dropout3(x)
        x = self.cov4(x)
        x = self.dropout4(x)
        x = self.drop_path2(x)
        src = src+x
        
        return src
    

 
class CoordConv(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CoordConv, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels,kernel_size,stride,padding,bias=False,groups=in_channels//8)
        self.scale = nn.Parameter(torch.zeros(1),requires_grad=True)
        
    @staticmethod
    def sinusoidal_embedding(n_channels, dim,padding_idx=False):
        pe = torch.FloatTensor([[p / (50000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
    def get_sinusoidal_embedding(self,d_model, max_seq_len):  
        positions = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(50000.0)) / d_model))   
        pos_encoding = torch.zeros(max_seq_len, d_model, dtype=torch.float32)  
        pos_encoding[:, 0::2] = torch.sin(positions * div_term)    
        pos_encoding[:, 1::2] = torch.cos(positions * div_term)   
        return pos_encoding 
           
    def forward(self, x):
        B,D,L = x.shape
        positional_emb = self.get_sinusoidal_embedding(D,L).transpose(-2,-1).unsqueeze(0).to(x.device).to(x.dtype)
        positional_emb = self.conv(positional_emb)
        x = x +  positional_emb*self.scale
        return x
                
        
 

class BBT(Module):
 
    def __init__(self,
                 embedding_dim=1024,
                 num_classes =cfg.vocab_size,                
                 num_layers=20,
                 num_heads=4,
                 mlp_ratio=2,
                 dropout_rate=0.15,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.15,
                 sequence_length=None,
                 *args, **kwargs):
        super(BBT, self).__init__()
        
        self.codebook = nn.Parameter(torch.empty(num_classes,embedding_dim),
                                          requires_grad=True)
        nn.init.kaiming_uniform_(self.codebook)
        
        dim_feedforward = int(embedding_dim * mlp_ratio)
        
        self.coord_conv = CoordConv(embedding_dim,embedding_dim)
        
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        self.blocks = nn.ModuleList([
            BLOCK(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = Linear(embedding_dim, num_classes)
    def gen_mask(self,start,end,patch_size,device):
        
        index = torch.arange(start,end)
        
        
        
        index = index.reshape(-1,1,patch_size)
        
        global_index = torch.cat([index,index.flatten(-2,-1).permute(1,0).reshape(-1,1,patch_size)],dim=-1)
        global_mask = global_index.permute(0,2,1)-global_index
        global_mask = global_mask>-1
        global_mask = global_mask.unsqueeze(0).cuda(device)
        return global_mask

    def gen_positional_weight(self,length,device):
        w = torch.arange(length)+1
        w = w.unsqueeze(0).unsqueeze(0).cuda(device)
        return w
        


    

    def forward(self,x) -> torch.Tensor:

        x = self.embedding(x)
        
        b,n,d = x.shape
        
        x = rearrange(x,'b n d -> b d n')
        pad_num = 0

        patch_size = random.randint(cfg.patch_size[0],cfg.patch_size[1])
        if n%patch_size!=0:
            pad_num = patch_size-n%patch_size
        
                
        x = F.pad(x, (0, pad_num), mode='constant', value=0)
        mask = self.gen_mask(0,x.shape[-1],patch_size=patch_size,device=x.device)

        x = self.coord_conv(x)
        for blk in self.blocks:

            x = blk(x,mask,patch_size)

        x = rearrange(x,'B D L -> B L D')
        x = x[:,:n,:] 
        x = self.norm(x)


        x = self.fc(x)                
        return x
    def embedding(self,x):

        x = F.one_hot(x, num_classes=cfg.vocab_size).to(self.codebook.dtype).to(self.codebook.device)
        x = x@self.codebook
        
        return x
    
          
        
        
class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.reduction = reduction

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        if self.reduction=='none':
            return loss
        else:
            return loss.mean()

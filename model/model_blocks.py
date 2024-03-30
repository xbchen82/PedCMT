import torch
from torch import nn
#from model.MultiHeadAttention import MultiHeadAttention
from torch import Tensor
import math
from einops import repeat
from einops.layers.torch import Rearrange

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PositionalEncoding(nn.Module): # 位置编码
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # 生成一个max_len行1列的张量
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # 生成一个d_model/2行1列的张量
        pe = torch.zeros(max_len, 1, d_model) # 生成一个【max_len，1，d_model】的张量
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 0, 1::2] = torch.cos(position * div_term) # 奇数列
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(0)] # 位置编码
        return self.dropout(x) 


class EmbedPosEnc(nn.Module):
    def __init__(self, input_size, d_model):
        super(EmbedPosEnc, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)
        #self.embedding = MultiScaleCNN(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model) # 位置编码

        self.arrange1 = Rearrange('b s e -> s b e')  # 重排列
        self.arrange2 = Rearrange('s b e -> b s e') # 重排列

    def forward(self, x, token):
        b = x.shape[0] # 获取批次大小
        y = self.embedding(x) # 嵌入
        token = repeat(token, '() s e -> b s e', b=b) # 重复token
        y = torch.cat([token, y], dim=1) # 拼接
        return self.arrange2(self.pos_enc(self.arrange1(y))) # 位置编码


class AttentionBlocks(nn.Module):
    def __init__(self, d_model, num_heads, rate=0.3, layer_norm_eps=1e-5):
        super(AttentionBlocks, self).__init__()

        self.att = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True) # 多头注意力
        self.drop = nn.Dropout(rate) 
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps) # 归一化

    def forward(self, x, y=None):
        y = x if y is None else y # 如果y为空，则y=x
        att_out, att_w = self.att(x, y, y) # 多头注意力
        att_out = self.drop(att_out) # dropout
        y = self.norm(x + att_out) # 归一化
        return y


import torch.nn.functional as F


class Time_att(nn.Module): # 在时间维度上进行注意力
    def __init__(self, dims):
        super(Time_att, self).__init__()
        self.linear1 = nn.Linear(dims, dims, bias=False)
        self.linear2 = nn.Linear(dims, 1, bias=False)
        self.time = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        y = self.linear1(x.contiguous())  
        y = self.linear2(torch.tanh(y)) 
        beta = F.softmax(y, dim=-1)
        c = beta * x
        return self.time(c.transpose(-1, -2)).transpose(-1, -2).contiguous().squeeze()

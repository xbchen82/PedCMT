import torch
from torch import nn
import numpy as np
from model.model_blocks import EmbedPosEnc, AttentionBlocks, Time_att
from model.FFN import FFN
from model.BottleNecks import Bottlenecks
from einops import repeat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.sigma_cls = nn.Parameter(torch.ones(1, 1, requires_grad=True, device=device), requires_grad=True) # 生成一个可训练的分类损失参数
        nn.init.kaiming_normal_(self.sigma_cls, mode='fan_out') # 初始化参数
        self.sigma_reg = nn.Parameter(torch.ones(1, 1, requires_grad=True, device=device), requires_grad=True) # 生成一个可训练的回归损失参数
        nn.init.kaiming_normal_(self.sigma_reg, mode='fan_out')  # 初始化参数

        d_model = args.d_model 
        hidden_dim = args.dff
        modal_nums = 2
        self.num_layers = args.num_layers
        self.token = nn.Parameter(torch.ones(1, 1, d_model)) # 生成一个可训练的token @@绿色token

        self.bbox_embedding = EmbedPosEnc(args.bbox_input, d_model) # 张量嵌入以及生成位置编码
        self.bbox_token = nn.Parameter(torch.ones(1, 1, d_model))   # 生成一个可训练的bbox_token

        self.vel_embedding = EmbedPosEnc(args.vel_input, d_model)  # 张量嵌入以及生成位置编码
        self.vel_token = nn.Parameter(torch.ones(1, 1, d_model))   # 生成一个可训练的vel_token

        self.bbox_att = nn.ModuleList() # 生成一个空的ModuleList
        self.bbox_ffn = nn.ModuleList()
        self.vel_att = nn.ModuleList()
        self.vel_ffn = nn.ModuleList()
        self.cross_att = nn.ModuleList()
        self.cross_ffn = nn.ModuleList()

        for _ in range(self.num_layers):
            self.bbox_att.append(AttentionBlocks(d_model, args.num_heads)) # 添加AttentionBlocks
            self.bbox_ffn.append(FFN(d_model, hidden_dim)) # 添加FFN
            self.vel_att.append(AttentionBlocks(d_model, args.num_heads))
            self.vel_ffn.append(FFN(d_model, hidden_dim))
            self.cross_att.append(AttentionBlocks(d_model, args.num_heads)) # 添加AttentionBlocks
            self.cross_ffn.append(FFN(d_model, hidden_dim))

        self.dense = nn.Linear(modal_nums * d_model, 4) # 全连接层
        self.bottlenecks = Bottlenecks(d_model, args) # Bottlenecks
        self.time_att = Time_att(dims=args.num_bnks) # Time_att
        self.endp = nn.Linear(modal_nums * d_model, 4) # 全连接层
        self.relu = nn.ReLU()
        self.last = nn.Linear(args.num_bnks, 1) # 全连接层
        self.sigmoid = nn.Sigmoid() # sigmoid激活函数

    def forward(self, bbox, vel):
        '''
            :bbox       :[b, 4, 32]
            :vel        :[b, 2, 32]
        '''
        '''
            bbox: [64, 16, 4]
            vel: [64, 16, 2]
        '''
        b = bbox.shape[0]
        token = repeat(self.token, '() s e -> b s e', b=b) # 重复token，使尺寸匹配

        bbox = self.bbox_embedding(bbox, self.bbox_token) # 张量嵌入以及生成位置编码
        vel = self.vel_embedding(vel, self.vel_token) # 张量嵌入以及生成位置编码

        bbox = self.bbox_att[0](bbox) # bbox的自注意力
        token = torch.cat([token, bbox[:, 0:1, :]], dim=1)  # 拼接token和bbox
        vel = self.vel_att[0](vel) # vel的自注意力
        token = torch.cat([token, vel[:, 0:1, :]], dim=1) # 拼接token和vel
        token = self.cross_att[0](token) # token的交叉注意力
        token_new = token[:, 0:1, :] # 取出token的第一个元素
        bbox = torch.cat([token_new, bbox[:, 1:, :]], dim=1) # 拼接token_new和bbox
        vel = torch.cat([token_new, vel[:, 1:, :]], dim=1) # 拼接token_new和vel
        bbox = self.bbox_ffn[0](bbox) # bbox的FFN
        vel = self.vel_ffn[0](vel) # vel的FFN
        token = self.cross_ffn[0](token)[:, 0:1, :] # token的FFN

        for i in range(self.num_layers - 1):
            bbox = self.bbox_att[i + 1](bbox)
            token = torch.cat([token, bbox[:, 0:1, :]], dim=1) 
            vel = self.vel_att[i + 1](vel)
            token = torch.cat([token, vel[:, 0:1, :]], dim=1)
            token = self.cross_att[i + 1](token)
            token_new = token[:, 0:1, :]
            bbox = torch.cat([token_new, bbox[:, 1:, :]], dim=1)
            vel = torch.cat([token_new, vel[:, 1:, :]], dim=1)
            bbox = self.bbox_ffn[i + 1](bbox)
            vel = self.vel_ffn[i + 1](vel)
            token = self.cross_ffn[i + 1](token)[:, 0:1, :]


        cls_out = torch.cat([bbox[:, 0:1, :], vel[:, 0:1, :]], dim=1) # 拼接bbox的token和vel的token
        cls_out_flatten = torch.flatten(cls_out, start_dim=1) # 展平
        end_point = self.endp(cls_out_flatten) # 全连接层预测endpoint

        bnk = self.relu(self.time_att(self.bottlenecks(bbox, vel))) # Bottlenecks
        tmp = self.last(bnk) # 全连接层预测穿越行为
        pred = self.sigmoid(tmp)
        return pred, end_point, self.sigma_cls, self.sigma_reg # 返回预测结果，endpoint预测结果，分类的sigma，回归的sigma

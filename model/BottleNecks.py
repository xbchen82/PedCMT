import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Bottlenecks(nn.Module): # 瓶颈结构
    def __init__(self, dims, args):
        super(Bottlenecks, self).__init__()
        self.dims = dims
        self.num_bnks = args.num_bnks # 单元数目
        self.num_layers = args.bnks_layers # 层数
        self.bbox = nn.ModuleList()
        self.vel = nn.ModuleList()

        self.bbox.append(nn.Linear(dims, dims + self.num_bnks, bias=True)) 
        self.vel.append(nn.Linear(dims, dims + self.num_bnks, bias=True))

        for _ in range(self.num_layers - 1):
            self.bbox.append(nn.Linear(dims + self.num_bnks, dims + self.num_bnks, bias=True))
            self.vel.append(nn.Linear(dims + self.num_bnks, dims + self.num_bnks, bias=True))
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def cut(self, x): # 切片
        return x[:, :, :self.dims], x[:, :, -self.num_bnks:] # 从第0个到第dims个，从倒数第num_bnks个到最后一个

    def forward(self, bbox, vel):
        bbox, bnk_bbox = self.cut(self.dropout(self.relu(self.bbox[0](bbox)))) # 生成下一层然后切片，得到bbox下一层和bnk_bbox
        vel, bnk_vel = self.cut(self.dropout(self.relu(self.vel[0](vel)))) # 生成下一层然后切片，得到vel下一层和bnk_vel
        bottlenecks = bnk_bbox + bnk_vel # 加和得到中间的交互单元

        for i in range(self.num_layers - 1):
            bbox = torch.cat((bbox, bottlenecks), dim=-1)
            bbox, bnk_bbox = self.cut(self.dropout(self.relu(self.bbox[i + 1](bbox))))
            vel, bnk_vel = self.cut(self.dropout(self.relu(self.vel[i + 1](torch.cat((vel, bottlenecks), dim=-1)))))
            bottlenecks = bnk_bbox + bnk_vel #+ bnk_token

        return bottlenecks
import torch
import torch.nn as nn
from einops import rearrange
from model.RevIN import RevIN
from tkinter import _flatten
import torch.nn.functional as F
import math
from model.kan import KAN
from model.embed import PositionalEmbedding
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一个全连接层
        self.fc2 = nn.Linear(hidden_size, output_size) # 第二个全连接层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU作为激活函数
        x = self.fc2(x)
        return x

def calculate_area(mu, sigma, x):
    # 计算正态分布的累积分布函数 CDF
    dist = torch.distributions.Normal(mu, sigma)
    cdf1 = dist.cdf(x[:,:,0])
    cdf2 = dist.cdf(x[:,:,-1])

    return cdf1 - cdf2

def calculate_area_1(mu, sigma, x):
    # 计算正态分布的累积分布函数 CDF
    dist = torch.distributions.Normal(mu, sigma)
    cdf = dist.cdf(x[:,:,-1])

    return -cdf
class GRUSTAD(nn.Module):
    def __init__(self, batch_size,win_size, enc_in, c_out, d_model=256, local_size=[3], global_size=[1], channel=55, dropout=0.05,mul_num=3, output_attention=True, ):
        super(GRUSTAD, self).__init__()
        self.output_attention = output_attention
        self.local_size = local_size
        self.global_size = global_size
        self.channel = channel
        self.win_size = win_size
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.batch_size = batch_size
        self.mul_num=mul_num
        self.kan_local_time = nn.ModuleList(
            KAN([localsize-1, d_model,1 ]) for index, localsize in enumerate(self.local_size))
        self.kan_global_time = nn.ModuleList(
            KAN([global_size[index]-localsize, d_model, 1]) for index, localsize in enumerate(self.local_size))
        self.kan_local_space = nn.ModuleList(
            KAN([channel, d_model, channel]) for index, localsize in enumerate(self.local_size))
        self.kan_global_space = nn.ModuleList(
            KAN([channel, d_model, channel]) for index, localsize in enumerate(self.local_size))
        # self.kan_local_time = nn.ModuleList(
        #     MLP(localsize - 1, d_model, 1) for index, localsize in enumerate(self.local_size))
        # self.kan_global_time = nn.ModuleList(
        #     MLP(global_size[index] - localsize, d_model, 1) for index, localsize in enumerate(self.local_size))
        # self.kan_local_space = nn.ModuleList(
        #     MLP(channel, d_model, channel) for index, localsize in enumerate(self.local_size))
        # self.kan_global_space = nn.ModuleList(
        #     MLP(channel, d_model, channel) for index, localsize in enumerate(self.local_size))
        self.position_embedding = PositionalEmbedding(d_model=d_model)
    def forward(self,x_in, in_size, in_num, op, it,in_x):
        local_out_time = []
        global_out_time = []
        local_out_space = []
        global_out_space = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        B, L, M = x_in.shape

        for index, localsize in enumerate(self.local_size):
            local_out_time.append(self.kan_local_time[index](in_size[index]).reshape(B,L,M))
            global_out_time.append(self.kan_global_time[index](in_num[index]).reshape(B,L,M))
            local_out_space.append(self.kan_local_space[index](x_in).reshape(B,L,M))
            global_out_space.append(torch.mean(
                self.kan_global_space[index](in_x[index].permute(0,1,3,2)).permute(0,1,3,2),dim=-1))

        local_out_time = list(_flatten(local_out_time))  # 3
        global_out_time = list(_flatten(global_out_time))  # 3
        local_out_space = list(_flatten(local_out_space))  # 3
        global_out_space = list(_flatten(global_out_space))  # 3

        if self.output_attention:
            return local_out_time, global_out_time, local_out_space, global_out_space
        else:
            return None


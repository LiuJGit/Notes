"""
定义神经网络模型
"""
from torch import nn
from lib import ws
import config
import torch.nn.functional as F
import torch

class ImdbModel(nn.Module):
    def __init__(self):
        super(ImdbModel,self).__init__()
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.embedding_dim = config.embedding_dim
        self.num_layer = config.num_layer
        self.bidirectional = config.bidirectional
        self.num_direc = config.num_direc
        self.dropout = config.dropout

        self.embedding = nn.Embedding(num_embeddings=len(ws), embedding_dim=config.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim,hidden_size=self.hidden_size,num_layers=self.num_layer,
                            bidirectional=self.bidirectional,batch_first=True,dropout=self.dropout)
        #两个全连接层，激活函数relu直接使用，无需定义
        self.fc1 = nn.Linear(self.hidden_size*self.num_direc, 20)
        self.fc2 = nn.Linear(20, 2)


    def forward(self,input):
        """
        :param input: [batch_size,max_sentence_len]
        :return:
        """
        x = self.embedding(input) #对input进行embedding操作，输出形状：[batch_size,max_sentence_len,embedding_dim]

        h_0 = torch.rand(self.num_direc * self.num_layer, self.batch_size, self.hidden_size).to(config.device)
        c_0 = torch.rand(self.num_direc * self.num_layer, self.batch_size, self.hidden_size).to(config.device)
        # 只选择最后一层的正反末尾的隐藏状态作为输出
        _,(h_n,c_n) = self.lstm(x,(h_0,c_0))
        out = torch.cat([h_n[-2,:,:],h_n[-1,:,:]],dim=-1) # [batch_size, num_direc * hidden_size]

        out = self.fc1(out) # [batch_size,20]
        out = F.relu(out) # [batch_size,20]
        out = self.fc2(out) # [batch_size,2]

        return F.log_softmax(out,dim=-1)


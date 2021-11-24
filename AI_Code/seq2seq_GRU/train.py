"""
模型的训练
"""

from seq2seq import Seq2Seq
import torch.nn as nn
from torch import optim
from word_sequence import word_sequence
from config import output_std_len, batch_size
from dataset import train_data_loader as train_data_loader
import torch
from tqdm import tqdm
import config
import os

# 模型
train_model = Seq2Seq().to(config.device)
# print(model)
# 优化器
optimizer = optim.Adam(train_model.parameters())

if os.path.exists(r".\model\seq2seq_model.pkl"):
    train_model.load_state_dict(torch.load(r".\model\seq2seq_model.pkl"))
    optimizer.load_state_dict(torch.load("./model/seq2seq_optimizer.pkl"))
    print('模型已加载')
else:
    #自定义初始化参数
    # print(model.named_parameters())
    for name, param in train_model.named_parameters():
        if 'bias' in name: #若参数是偏置，则初始化为0
           nn.init.constant_(param, 0.0)
        elif 'weight' in name: #若参数是权重，则使用xavier初始化
           nn.init.xavier_normal_(param)
    #     print("name:",name,'\n',"param:",param)
    #     print('-'*20)
    print('模型已初始化')


# 损失函数对象，之前都是用的函数直接计算，而不是对象
criterion = nn.NLLLoss(ignore_index=word_sequence.PAD,reduction="mean")

# 计算损失
def get_loss(decoder_outputs, target):
    # [batch_size,output_std_len]->[batch_size*output_std_len]
    target = target.view(-1)
    # [batch_size,output_std_len,vocab_size]->[batch_size*output_std_len,vocab_size]
    decoder_outputs = decoder_outputs.view((batch_size*output_std_len,-1))
    loss = criterion(decoder_outputs, target)
    return loss

def train(epoch):
    bar = tqdm(train_data_loader)
    for idx,(input,target,input_length,target_length) in enumerate(bar):
        optimizer.zero_grad()

        input = input.to(config.device)
        target = target.to(config.device)
        decoder_outputs, decoder_hidden = train_model(input, target, input_length) # 即 train_model.forward(input, target, input_length)
        loss = get_loss(decoder_outputs,target)
        loss.backward()
        optimizer.step()

        # print("Train Epoch:{} ".fromat(epoch, idx))
        bar.set_description('Train Epoch: {}, Loss: {:.6f}'.format(
            epoch, loss.item()))

    torch.save(train_model.state_dict(), "model/seq2seq_model.pkl")
    torch.save(optimizer.state_dict(), 'model/seq2seq_optimizer.pkl')

if __name__ == '__main__':
    for i in range(3):
        train(i)


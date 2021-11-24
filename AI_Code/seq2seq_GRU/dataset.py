"""
准备dataset、dataloader
"""
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
from word_sequence import word_sequence
import config,string

# 准备Dataset
class RandomDataset(Dataset):
    def __init__(self,data_size):
        super(RandomDataset,self).__init__()
        self.data_size = data_size
        self.max_seq_len = config.max_seq_len
        # np.random.seed()配合np.random.random()用于生产随机数
        # 每次random前将seed置为同一个置，则生成的随机数相同
        # np.random.seed(1)
        self.data = [] # 生成的输入字符串
        for i in range(self.data_size):
            str_len = np.random.randint(1, self.max_seq_len + 1)  # 随机生成的字符串长度
            str1 = ''.join([np.random.choice(list(string.ascii_lowercase)) for j in range(str_len)])
            self.data.append(str1)

    def __getitem__(self, item):
        """返回input，target，input_length,target_length(真实长度)"""
        input = self.data[item] # strs
        target = input + input[0] # strs
        input_length = len(input) # int
        target_length = len(target) # int
        # 这里还放回input及target的真实长度，是因为后面nn.utils.rnn.pack_padded_sequence需要使用
        return input,target,input_length,target_length

    def __len__(self):
        return self.data_size

# 准备DataLoader

def collate_fn(batch):
    # 1. 对batch进行排序，按照长度从长到短的顺序排序:
    # 后面使用nn.utils.rnn.pack_padded_sequence加速计算时需要按照句子的长度**降序排序**
    # batch的维度：[batch_size,4]
    # 每个数据点包含input(字符串),target(字符串),input_length,target_length，4项
    batch = sorted(batch,key=lambda x:x[3],reverse=True)
    # 拆包，将一个batch的数据的input、target等拆出来打包在一起
    input,target,input_length,target_length = zip(*batch)

    # 2. 将字符串的input,target转化为数字列表，并进行padding操作
    input = torch.LongTensor([word_sequence.transform(sentence,std_len=config.std_len,add_eos=True)
                              for sentence in input])
    target = torch.LongTensor([word_sequence.transform(sentence,std_len=config.output_std_len,add_eos=True)
                              for sentence in target])

    input_length = torch.LongTensor(input_length)
    target_length = torch.LongTensor(target_length)

    return input,target,input_length,target_length

# 实例化
train_data_set = RandomDataset(640000)
train_data_loader = DataLoader(dataset=train_data_set, batch_size=config.batch_size, collate_fn=collate_fn, drop_last=True)

eval_data_set = RandomDataset(64000)
eval_data_loader = DataLoader(dataset=eval_data_set, batch_size=config.batch_size, collate_fn=collate_fn, drop_last=True)



if __name__ == "__main__":
    for input, target, input_length, target_length in train_data_loader:
        print(input) # 输入tensor
        print(len(input)) # batch_size
        print(target) # 输出tensor
        print([word_sequence.inverse_transform(sentence) for sentence in input.numpy().tolist()])  # 输入字符串列表
        print([word_sequence.inverse_transform(sentence) for sentence in target.numpy().tolist()])
        break
# import numpy as np

# np.random.seed(10)
# print(np.random.randint(1,9,size=4))
#
# a = '123'
# b=list(a)
# b.extend([4])
#
# print(b)
#
# print([5]*6)

# a = np.random.randint(1,10,size=20)
#
# print(a)

# import  torch
# a = torch.LongTensor([[1,2,3]])
# print(a.shape)
# print(a.squeeze(0))
# print(a)

# eng = 'abcdefghijklmnopqrstuvwxyz'
# dict_char2num = {}
# for i in eng:
#     dict_char2num[i] = len(dict_char2num)
# print(dict_char2num)

# import string
# print(string.ascii_letters)
# print(string.ascii_lowercase)
# print(list('abcdef'))

import numpy as np
import string
# print(np.random.randint(1,11,size=100))
# print(np.random.choice(list(string.ascii_lowercase)))
# print(list(string.ascii_lowercase))

# inputs = []
# for i in range(2):
#     str_len = np.random.randint(1, 8 + 1)  # 随机生成的字符串长度
#     str1 = ''.join([np.random.choice(list(string.ascii_lowercase)) for j in range(str_len)])
#     inputs.append(str1)
# print(inputs)

# import torch
#
# print(torch.LongTensor([[1]]*5))


# from seq2seq import Seq2Seq
# import torch.nn as nn
#
# # 模型
# model = Seq2Seq()
# # print(model)
# #自定义初始化参数
# print(model.named_parameters())
# for name, param in model.named_parameters():
#     if 'bias' in name: #若参数是偏置，则初始化为0
#        nn.init.constant_(param, 0.0)
#     elif 'weight' in name: #若参数是权重，则使用xavier初始化
#        nn.init.xavier_normal_(param)
#     print("name:",name,'\n',"param:",param)
#     print('-'*20)
#     break

# import torch
# index =  torch.LongTensor([[19,1],[20,2],[21,3]])
# print(index)
# print(index.size(0))

# input = [2,3,5,7]
# print(max(input))

# input = 'abcd'
# print(input[0])

# import torch
# import config
#
# a = torch.LongTensor([[1,2],[3,4]])
# print(a.device)
# b = a.to(config.device)
# print(b.device)
# print(a.device)




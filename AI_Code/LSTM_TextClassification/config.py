import torch

# dataset 参数
batch_size = 128

# word to sequence 参数
min_word_count = 5 #最小词频
max_word_count = None #最大词频
max_features = None #除未知字符和填充字符外，词的最大个数

max_sentence_len = 20 #一个句子的最大长度
embedding_dim = 100 #embedding后每个单词的维度

# LSTM 网络参数
hidden_size = 64
num_layer = 2
bidirectional = True
if bidirectional:
    num_direc = 2
else:
    num_direc = 1
dropout = 0.5


# GPU训练参数
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")





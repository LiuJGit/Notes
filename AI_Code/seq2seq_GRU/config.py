"""
参数配置
"""
import  torch

max_seq_len = 8 # 生成训练数据时，句子的最大长度
std_len = 9 # 句子规整后的统一长度
output_std_len = 10 # 输出句子规整后的统一长度
batch_size = 64

# 网络相关设置

# encoder
en_embedding_dim = 100
en_dropout = 0.5
# en_bidirectional目前只支持 False：若改为True，encoder的代码不用改，反而是decoder的代码需要改动
# 因为decoder的初始hidden由encoder最后一步的hidden给出，decoder的hidden_size的设置依赖于encoder
en_bidirectional = False
en_hidden_size = 20
en_num_layers = 2

# decoder
de_embedding_dim = 100   # 可以和en_embedding_dim设置得不一样
de_dropout = 0.5    # 可以和en_dropout设置得不一样
de_bidirectional = False    # decoder的bidirectional貌似只能是False
# 因为decoder的初始hidden由encoder最后一步的hidden给出，
# 因此num_layers及hidden_size二者应该相同
de_hidden_size = en_hidden_size
de_num_layers = en_num_layers

# 解码时，引入teacher forcing 机制
teacher_forcing_rate = 0.5

# GPU 训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
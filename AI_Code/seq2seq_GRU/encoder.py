"""
编码器
"""

import torch.nn as nn
from word_sequence import word_sequence
import config
from dataset import train_data_loader

class NumEncoder(nn.Module):
    def __init__(self):
        super(NumEncoder, self).__init__()
        self.vocab_size = len(word_sequence)
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=config.en_embedding_dim)
        self.gru = nn.GRU(input_size=config.en_embedding_dim,hidden_size=config.en_hidden_size,
                          num_layers=config.en_num_layers,batch_first=True,dropout=config.en_dropout,
                          bidirectional=config.en_bidirectional)

    def forward(self,input,input_length):
        """
        :param input:[batch_size,std_len]
        :param input_length:[batch_size]
        :return:
        """
        embedded = self.embedding(input) #[batch_size,std_len,embedding_dim]

        # 对文本对齐（加上末尾标记，并padding）之后的句子进行打包，能够加速在LSTM or GRU中的计算过程，相应地，计算后的结果要进一步使用需要解包
        # 无论是否使用 GPU，'lengths' argument should be a 1D CPU int64 tensor
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths=input_length.cpu(), batch_first=True)
        # 这里没有提供初始隐变量，因此，初始隐变量会取默认值0
        output,hidden = self.gru(embedded)
        # 对前面打包后的结果再进行解包
        output, output_length = nn.utils.rnn.pad_packed_sequence(output,
                                                                 batch_first=True,padding_value=word_sequence.PAD)
        # output [batch_size,max_seq_len,num_directions*hidden_size], max_seq_len: the max seq_len, not include EOS, PAD
        # hidden: [num_layers * num_directions, batch, hidden_size]
        # output_length: [batch_size], the value gives the real length of each sentence,
        #                not include EOS, PAD, i.e. seq_len
        return output, hidden, output_length

if __name__ == "__main__":
    num_encoder = NumEncoder().to(config.device)
    for input, target, input_length, target_length in train_data_loader:
        print(input.shape,input_length)
        # 对于input_length，nn.utils.rnn.pack_padded_sequence需要的总是 CPU 类型的
        # 因此，在上述NumEncoder代码中，我们将参数的输入强制转化为CPU类型：lengths=input_length.cpu()。
        # 基于这个强制转换，下面的两行代码效果是相同的，input_length后有无.to(config.device)均可
        # output, hidden, output_length = num_encoder.forward(input.to(config.device),input_length)
        output, hidden, output_length = num_encoder.forward(input.to(config.device),input_length.to(config.device))
        # 无论是用 GPU 还是 CPU 计算，返回的 output_length 均是 CPU 类型的！

        print(output.shape,output_length)
        print(hidden.shape)
        print('-'*10)
        print(output.device) # cuda:0
        print(hidden.device) # cuda:0
        print(output_length.device) # cpu
        break






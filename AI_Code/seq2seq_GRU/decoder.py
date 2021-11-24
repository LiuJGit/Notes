"""
实现解码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from word_sequence import word_sequence
import random

class NumDecoder(nn.Module):
    def __init__(self):
        super(NumDecoder, self).__init__()
        self.vocab_size = len(word_sequence)

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=config.de_embedding_dim)
        self.gru = nn.GRU(input_size=config.de_embedding_dim, hidden_size=config.de_hidden_size,
                          num_layers=config.de_num_layers, batch_first=True, dropout=config.de_dropout,
                          bidirectional=False)
        self.fc = nn.Linear(config.de_hidden_size,self.vocab_size)

    def forward(self,encoder_hidden,target):
        """
        只能用于训练，无法用于预测、评估
        :param encoder_hidden: [batch_size,hidden_size]
        :param target: [batch_size,output_std_len]，用于teacher forcing
        :return decoder_output:[batch_size,output_std_len,vocab_size]
        :return decoder_hidden:[num_layers,batch_size,hidden_size]
        """
        # 初始化
        # 初始输入为SOS, [batch_size,1]: 值为对应的数字
        decoder_input = torch.LongTensor([[word_sequence.SOS]]*config.batch_size).to(config.device) # 还未embedding
        # 初始化隐状态, [num_layers,batch_size,hidden_size]
        decoder_hidden = encoder_hidden
        # 保存解码器各步的输出, [batch_size,output_std_len,vocab_size]
        # 因为word_sequence中将文本序列数字化了,每个单词对应一个类别，也就是0到vocab_size-1中的一个数字
        decoder_outputs = torch.zeros(config.batch_size,config.output_std_len,self.vocab_size).to(config.device) # soft one-hot编码

        # 与 encoder 不同，每个时间步的输入并不是事先就已知的，需要在计算过程中确定
        # 因此，无法像 encoder 中那样一次调用就计算完所有时间步，而需要借助for循环
        for t in range(config.output_std_len):
            # decoder_output_t:[batch_Size, vocab_size], decoder_hidden:[num_layers,bath_size,hidden_size]
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input,decoder_hidden)
            # 保存第t步的结果，soft one-hot 编码
            decoder_outputs[:,t,:] = decoder_output_t

            # 在训练的过程中，使用 teacher forcing，进行纠偏
            # decoder_input 由 decoder_output_t 给出
            use_teacher_forcing = random.random() < config.teacher_forcing_rate
            if use_teacher_forcing:
                # 下一步的输入使用真实值，也就是teacher forcing
                decoder_input = target[:,t] # [batch_size]
                decoder_input = decoder_input.unsqueeze(1) # [batch_size] --> [batch_size,1]
            else:
                # 下一步的输入使用预测值
                # index:[bath_size,1]
                value, index = torch.topk(decoder_output_t, k=1)  # 取topk的值和下标,k=1,也就是最大值
                decoder_input = index

        return decoder_outputs, decoder_hidden

    def forward_step(self,decoder_input,decoder_hidden):
        """
        :param decoder_input:[batch_size,1]，还未embedding
        :param decoder_hidden:[num_layers,bath_size,hidden_size]
        :return decoder_output_t:[batch_Size, vocab_size]
        :return decoder_hidden:[num_layers,batch_size,hidden_size]
        """
        # 对输入进行embedding
        # decoder_input:[batch_size,1]
        # embedded:[batch_size,1,embedding_dim]
        embedded = self.embedding(decoder_input)

        # GRU 单元计算
        # 只计算了一个时间步，因此 decoder_output_t.permute([1,0,2])==decoder_hidden: True
        decoder_output_t, decoder_hidden = self.gru(embedded, decoder_hidden)
        # print("decoder_output_t:",decoder_output_t.shape) # [batch_size,1,hidden_size]
        # print("decoder_hidden:",decoder_hidden.shape) # [1,batch_size,hidden_size]

        decoder_output_t = decoder_output_t.squeeze(1) # 张量降阶，去除第0阶:[batch_size,hidden_size]
        # print("decoder_output_t:", decoder_output_t.shape)
        decoder_output_t = self.fc(decoder_output_t) # [batch_Size, vocab_size]
        # print("decoder_output_t:", decoder_output_t.shape)
        decoder_output_t = F.log_softmax(decoder_output_t,dim=-1) # [batch_Size,vocab_size]
        # print("decoder_output_t:", decoder_output_t.shape)

        return decoder_output_t, decoder_hidden

    def forward_eval(self,encoder_hidden):
        """
        该方法相当于是模型训练完毕后，用于预测、评估的 forward 方法
        :param encoder_hidden: [batch_size,hidden_size]
        :return decoder_num_list:[ [num_1,num_2,...,num_output_std_len],...,
                                   [num_1,num_2,...,num_output_std_len] ] len=batch_size
        """
        # 初始化
        batch_size = encoder_hidden.size(1)  # 评估的时候和训练的batch_size不同，不适用config的配置
        # 初始输入为SOS, [batch_size,1]: 值为对应的数字
        decoder_input = torch.LongTensor([[word_sequence.SOS]] * batch_size).to(config.device)  # 还未embedding
        # 初始化隐状态, [num_layers,batch_size,hidden_size]
        decoder_hidden = encoder_hidden
        # 保存解码器各步的输出, [batch_size,output_std_len,vocab_size]
        # 因为word_sequence中将文本序列数字化了,每个单词对应一个类别，也就是0到vocab_size-1中的一个数字
        # 但这里得到的 decoder_outputs 只是 soft one-hot编码
        decoder_outputs = torch.zeros(batch_size, config.output_std_len, self.vocab_size).to(config.device)

        # 与 encoder 不同，每个时间步的输入并不是事先就已知的，需要在计算过程中确定
        # 因此，无法像 encoder 中那样一次调用就计算完所有时间步，而需要借助for循环
        for t in range(config.output_std_len):
            # decoder_output_t:[batch_Size, vocab_size], decoder_hidden:[num_layers,bath_size,hidden_size]
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            # 保存第t步的结果，soft one-hot 编码
            decoder_outputs[:, t, :] = decoder_output_t

            value, index = torch.topk(decoder_output_t, k=1)  # 取topk的值和下标,k=1,也就是最大值
            decoder_input = index

        # 将 soft one-hot 编码decoder_outputs的最大值的index提取出来
        # decoder_index:[batch_size,output_st_len]
        decoder_index = decoder_outputs.max(dim=-1)[1]

        return decoder_index

if __name__ == "__main__":
    from encoder import NumEncoder
    from dataset import train_data_loader

    num_encoder = NumEncoder().to(config.device)
    num_decoder = NumDecoder().to(config.device)
    for input, target, input_length, target_length in train_data_loader:
        output, hidden, output_length = num_encoder.forward(input.to(config.device), input_length)
        decoder_outputs, decoder_hidden = num_decoder.forward(encoder_hidden=hidden.to(config.device),
                                                              target=target.to(config.device))
        # print(decoder_outputs.shape,decoder_hidden.shape)
        print(decoder_outputs.shape)
        #将 soft one-hot 编码decoder_outputs的最大值的index提取出来，转化为列表，即数值序列
        out_put_num_list = decoder_outputs.max(dim=-1)[1].cpu().numpy().tolist()
        print(out_put_num_list)
        print(len(out_put_num_list))

        print('-'*10)
        print(decoder_outputs.device) # cuda:0
        print(decoder_hidden.device) # cuda:0

        break
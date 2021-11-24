"""
完成seq2seq模型
"""
import torch.nn as nn
from encoder import NumEncoder
from decoder import NumDecoder

encoder = NumEncoder()
decoder = NumDecoder()

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target, input_length):
        # 进行编码和解码 (训练用)
        output, hidden, output_length = self.encoder.forward(input, input_length)
        decoder_outputs, decoder_hidden = self.decoder.forward(hidden, target)
        return decoder_outputs, decoder_hidden

    def predict(self,input, input_length):
        # 进行编码和解码 (预测用)
        output, hidden, output_length = self.encoder.forward(input, input_length)
        decoder_index = self.decoder.forward_eval(hidden)
        return decoder_index
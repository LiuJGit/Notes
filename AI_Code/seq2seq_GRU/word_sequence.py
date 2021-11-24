"""
文本序列化：
1. 构建字典
2. 字符串-->数字序列
3. 数字序列-->字符串
"""

import string

class WordSequence:
    UNK_TAG = "<UNK>" #未知字符
    PAD_TAG = "<PAD>" #填充字符
    UNK = 0
    PAD = 1

    EOS_TAG = "EOS" # 句子结束字符
    SOS_TAG = "SOS" # 句子开始字符
    EOS = 2
    SOS = 3

    def __init__(self):
        # dict字典用于保存词语和对应的数字
        self.dict = {
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD,
            self.EOS_TAG:self.EOS,
            self.SOS_TAG:self.SOS
        }
        for i in string.ascii_lowercase:
            self.dict[i] = len(self.dict) # 26个小写英文字母字符串，对应数字 4-29

        # num2word_dict用于数字和对应的字符
        self.num2str_dict = dict(zip(self.dict.values(),self.dict.keys()))

    def __len__(self):
        # 返回词典的长度
        return len(self.dict)

    def transform(self,sentence,std_len=None,add_eos=True):
        """
        将句子转化为数字序列
        :param sentence: 句子，一个字符串
        :param std_len: 句子规整后的长度
        :param add_eos:是否添加结束符
        :return:
        """
        str_list = list(sentence) # [str1,str2,...]，相当于对句子进行分词

        seq_len = len(str_list) + 1 if add_eos else len(str_list)

        if std_len is not None:
            assert std_len >= seq_len, "规整的句子长度std_len:{}设置得小于seq_len:{}！".format(std_len,seq_len)

        num_list = [self.dict.get(i,self.UNK) for i in str_list]
        if add_eos:
            num_list.append(self.EOS)
        if std_len is not None:
            num_list.extend([self.PAD]*(std_len-seq_len))

        return num_list

    def inverse_transform(self,num_list):
        """数字序列-->字符串"""
        str_list = []
        for i in num_list:
            if i == self.EOS:
                break
            str_list.append(self.num2str_dict.get(i,self.UNK_TAG))

        return ''.join(str_list)

# 实例化类
word_sequence = WordSequence()

if __name__ == "__main__":
    print(word_sequence.dict)
    print("-"*10)

    print(word_sequence.num2str_dict)
    print("-" * 10)

    num_list = word_sequence.transform("abcd", add_eos=True)
    print(num_list)
    print(word_sequence.inverse_transform(num_list))


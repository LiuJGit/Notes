'''
文本序列化：
构建词典，句子-->数字序列，数字序列-->句子
'''

from config import min_word_count,max_word_count,max_features,max_sentence_len

class WordSequence:
    UNK_TAG = "<UNK>" #未知字符
    PAD_TAG = "<PAD>" #填充字符
    UNK = 0
    PAD = 1

    def __init__(self):
        # dict字典用于保存词语和对应的数字
        self.dict = {
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD
        }
        # count字典用于记录词频
        self.count = {}

    def fit(self,sentence):
        """
        接受一条句子，修正词频
        :param sentence:[word1,word2,...]
        :return:
        """
        for word in sentence:
            self.count[word] = self.count.get(word,0) + 1

    def build_vocab(self,min_count=min_word_count,max_count=max_word_count,max_features=max_features):
        """
        构造词典
        :param min_count:最小词频
        :param max_count: 最大词频
        :param max_features: 最大词语数
        :return:
        """
        if min_count is not None:
            self.count = {word:count for word,count in self.count.items() if count >= min_count}
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count <= max_count}
        if max_features is not None:
            self.count = dict(sorted(self.count.items(), key=lambda x:x[-1], reverse=True)[:max_features])

        # 构造词典：
        for word in self.count.keys():
            self.dict[word] = len(self.dict) #每个词和一个数字对应

        # 将词典键值对进行翻转：
        self.inverse_dict = dict(zip(self.dict.values(),self.dict.keys()))

    def transform(self,sentence,max_len=max_sentence_len):
        """
        规整句子长度，把一条句子转化为一个数字序列
        :param sentence:[str1,str2,...]
        :return: [int1,int2,...]
        """
        # 调整sentence的长度
        if len(sentence)>max_len:
            sentence = sentence[:max_len]
        else:
            sentence = sentence + [self.PAD_TAG]*(max_len-len(sentence))
        # 将sentence转化为num，并返回
        return [self.dict.get(i,self.UNK) for i in sentence]

    def inverse_transform(self,num_seq):
        """
        将一个数字序列转化为一条句子
        :param num: [int1,int2,...]
        :return: [str1,str2,...]
        """
        return [self.inverse_dict.get(i,self.UNK_TAG) for i in num_seq]

    def __len__(self):
        # 返回词典的长度
        return len(self.dict)

if __name__ == '__main__':
    sentences  = [["今天","天气","很","好"],
                  ["今天","去","吃","什么"]]
    ws = WordSequence()
    for sentence in sentences:
        ws.fit(sentence)
    ws.build_vocab(min_count=1)
    print(ws.dict)
    ret = ws.transform(["好","好","好","好","好","好","好","热","呀"],max_len=10)
    print(ret)
    ret = ws.inverse_transform(ret)
    print(ret)

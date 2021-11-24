"""
模型的评估
"""
from seq2seq import Seq2Seq
import torch
from dataset import eval_data_loader
from tqdm import tqdm

from word_sequence import word_sequence
import config

# 模型
eval_model = Seq2Seq().to(config.device)
eval_model.load_state_dict(torch.load("model/seq2seq_model.pkl"))
eval_model.eval()

def eval_all():
    # 评估模型的训练效果

    count = 0
    bar = tqdm(eval_data_loader)
    for idx, (input, target, input_length, target_length) in enumerate(bar):
        # 使用模型进行预测
        input = input.to(config.device)
        input_length = input_length.to(config.device)
        decoder_index = eval_model.predict(input, input_length)
        # print(decoder_index)

        # 比较预测结果
        target = target.to(config.device)
        for i in range(target.size(0)):
            if decoder_index[i,0:target_length[i]].equal(target[i,0:target_length[i]]):
                count += 1

    accuracy = count/len(eval_data_loader.dataset) # drop_last=True 时，可能评估的总样本数并没有分母那么多
    print('count:{},total:{},accuracy:{}'.format(count,len(eval_data_loader.dataset),accuracy))

def pre(sentences=['abc','abcd']):
    # 给定 sentences 用于预测输出

    # 转小写
    sentences = [sentence.lower() for sentence in sentences]
    # 提供给 encoder 的句子，排列顺序应按长度倒序
    sentences = sorted(sentences,key=lambda x:len(x),reverse=True)
    input_length = [len(sentence) for sentence in sentences]

    # -----比较如下两种写法的不同----
    # # 写法一：直接采用assert，若不满足条件input_length[0] <= 5，则程序终止运行并报错
    # assert input_length[0] <= 5, '句子长度应该小于{}'.format(5)
    # ---------------------------
    # # 写法二：将assert 与异常处理结合到一起，若不满足条件input_length[0] <= 5，
    # # 则会捕捉到 AssertionError: 句子长度应该小于5,
    # # 然后打印错误具体信息，**程序继续往下运行**.
    # # 事实上，word_sequence 的要求是句子的最大长度为 config.max_seq_len = 8
    # # 因此，若采用写法二，对于长度大于5，不超过8的句子，捕获异常并处理后，接下来的程序依旧会顺利、正确地处理
    # # 但若句子长度大于8，则会触发 word_sequence 中的assert，程序终止运行并报错
    # try:
    #     assert input_length[0] <= 5, '句子长度应该小于{}'.format(5)
    # except Exception as detail:
    #     print(detail)
    # ---------------end---------

    input = [word_sequence.transform(sentence=sentence,std_len=config.std_len,add_eos=True) for sentence in sentences]

    # 转化为 seq2seq 模型能处理的 tensor
    input_length = torch.LongTensor(input_length)
    input = torch.LongTensor(input).to(config.device)
    # 输入给模型计算得到输出 decoder_index
    decoder_index = eval_model.predict(input, input_length)
    # decoder_index 是一个tensor，将其转化为 list
    num_lists = decoder_index.cpu().numpy().tolist()
    # 将每个 num_list 转化为字符串
    output = [word_sequence.inverse_transform(num_list) for num_list in num_lists]
    # 输入、输出对比，并打印
    result =  list(zip(sentences, output))
    print(result)

if __name__ == '__main__':

    eval_all()

    sentences = ['zhou','wen','li','hello','Python','Liu','Jian','LOVE']
    pre(sentences)
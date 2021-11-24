"""
计算并保存WordSequence模型
"""
from word_sequence import WordSequence
from dataset import get_dataloader
import pickle
from tqdm import tqdm

if __name__ == "__main__":
    dl_train = get_dataloader(train=True)
    dl_test = get_dataloader(train=False)

    ws = WordSequence()

    # tqdm(range(100), ascii=False, total=100, desc="自定义的输出为")
    for content, label in tqdm(dl_train,ascii=False,desc="训练数据的处理:"):
        for sentence in content:
            ws.fit(sentence)
    for content, label in tqdm(dl_test,ascii=False,desc="测试数据的处理:"):
        for sentence in content:
            ws.fit(sentence)

    ws.build_vocab()
    # 保存ws模型
    pickle.dump(ws,open(r".\models\ws.pkl",'wb'))
    # 加载ws模型
    ws = pickle.load(open(r".\models\ws.pkl", "rb"))
    print(len(ws))






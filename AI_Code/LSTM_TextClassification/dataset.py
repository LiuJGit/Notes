"""
完成数据集的准备
"""

from torch.utils.data import Dataset, DataLoader
import os
from config import batch_size
from utils import tokenlize,collate_fn1,collate_fn2

# 数据集类
class ImdbDataset(Dataset):
    def __init__(self,train=True):

        # 得到所有文件的路径
        data_path = r".\aclImdb"
        data_path += r"\train" if train else r"\test"
        self.total_path = []
        for tem_path in [r"\pos",r"\neg"]:
            cur_path = data_path + tem_path
            files_path = [os.path.join(cur_path,i) for i in os.listdir(cur_path) if i.endswith(".txt")]
            self.total_path += files_path

    def __getitem__(self, index):
        file_path = self.total_path[index]
        # 获取label
        label = int(file_path.split("_")[-1].split('.')[0])
        label = 0 if label<5 else 1
        # 获取内容，并分词
        content = open(file_path,encoding='UTF-8').read()
        content = tokenlize(content)

        # 返回内容和标签
        return content, label

    def __len__(self):
        return len(self.total_path)

# 加载数据，collate_fn：collate_fn1时返回的是句子，collate_fn2时返回的句子对应的数字序列
def get_dataloader(train=True,collate_fn=collate_fn1):
    dataset = ImdbDataset(train)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn,drop_last=True)
    return data_loader

if __name__ == "__main__":
    for idx, (input, target) in enumerate(get_dataloader(train=True,collate_fn=collate_fn1)):
        print(idx)
        print(input)
        print(target)
        break

    for idx, (input, target) in enumerate(get_dataloader(train=True,collate_fn=collate_fn2)):
        print(idx)
        print(input)
        print(target)
        break
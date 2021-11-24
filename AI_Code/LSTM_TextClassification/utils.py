import re
from lib import ws
import torch

# 分词函数
def tokenlize(content):
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@'
        , '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]
    content = content.lower()  # 把大写转化为小写
    content = re.sub("<.*?>", " ", content) # "<.*?>" 和 "|".join(fileters) 均为正则表达式，"|" 表示或
    content = re.sub("|".join(fileters), " ", content)
    return [i.strip() for i in content.split()]

# 重写collate_fn函数，用于build_ws时构建dataloader
def collate_fn1(batch):
    """
    :param batch: [(content1,label1),(content2,label2),...]
    :return:
    """
    content, label = list(zip(*batch))

    return content, label

# 重写collate_fn函数，用于ws构建完后构建dataloader
def collate_fn2(batch):
    """
    :param batch: [(content1,label1),(content2,label2),...]
    :return:
    """
    content, label = list(zip(*batch))

    # 使用ws将文本序列化，这时经过DataLoader返回的就不是文本，而是数字列表组成的列表了
    content = [ws.transform(sentence) for sentence in content]
    # 还要将数字列表转化为longtensor
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)

    return content, label
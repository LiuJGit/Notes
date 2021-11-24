"""
模型测试
"""
import torch
from tqdm import tqdm
from dataset import get_dataloader
from torch.nn import functional as F
from model import ImdbModel
from utils import collate_fn2
from config import device
from train import train

def test():
    test_loss = 0
    correct = 0
    imdb_model = ImdbModel().to(device)
    imdb_model.load_state_dict(torch.load(r".\models\imdb_model.pkl"))
    imdb_model.eval()
    test_dataloader = get_dataloader(train=False,collate_fn=collate_fn2)
    with torch.no_grad():
        for input, target in tqdm(test_dataloader):
            # GPU转换
            input = input.to(device)
            target = target.to(device)

            output = imdb_model(input) #[batchsize,2]
            # 总损失
            test_loss += F.nll_loss(output, target)
            # 正确预测的总数
            pred = output.max(dim=1,keepdim=False)[1] # dim=0(按列取最大)，1/-1(按行取最大)；[1]返回的是最大的索引
            correct += pred.eq(target.data).sum()
            # print(pred)
            # print(target.data)
            # print(pred.eq(target.data))
            # print(pred.eq(target.data).sum())
            # break

        # 平均损失
        avg_loss = test_loss/len(test_dataloader.dataset)
        # 正确率
        accuracy = correct/len(test_dataloader.dataset)
        print("avg_loss:{}; accuracy:{}".format(avg_loss, accuracy))

        # print(correct)
        # print(len(test_dataloader.dataset))

if __name__ == "__main__":
    # test()
    for i in range(3):
        train(i)
        if (i+1)%2 != 0:
            print('-'*10,i,'-'*10)
            test()
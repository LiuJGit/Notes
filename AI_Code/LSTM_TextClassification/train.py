"""
训练模型
"""
from model import ImdbModel
from torch.optim import Adam
from dataset import get_dataloader
from tqdm import tqdm
from torch.nn import functional as F
import torch
from utils import collate_fn2
from config import device
import os

model = ImdbModel().to(device)
if os.path.exists(r".\models\imdb_model.pkl"):
    model.load_state_dict(torch.load(r".\models\imdb_model.pkl"))
optimizer = Adam(model.parameters())

def train(epoch):
    train_dataloader = get_dataloader(train=True,collate_fn=collate_fn2)
    bar = tqdm(train_dataloader, total=len(train_dataloader))
    for idx, (input, target) in enumerate(bar):
        # GPU转换
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        pred = output.max(dim=1, keepdim=False)[1]
        correct = 100. *pred.eq(target.data).cpu().numpy().mean()

        bar.set_description("epcoh:{} idx:{} loss:{:.6f} accuracy:{:.2f}%".format(
            epoch, idx, loss.item(),correct))

    torch.save(model.state_dict(), "models\imdb_model.pkl")
    torch.save(optimizer.state_dict(), "models\optimizer.pkl")


if __name__ == '__main__':
    for i in range(3):
        train(i)
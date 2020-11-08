import torch
import numpy as np
import torch.nn as nn
from nvdm_torch import *
from torch.utils.data import DataLoader
from dataset import FeatDataset


def test(dataloader, model):
    loss_sum = 0.0
    ppx_sum = 0.0
    kld_sum = 0.0
    word_count = 0
    doc_count = 0
    for data_batch, count_batch in dataloader:
        data_batch = data_batch.float()
        kld, recons_loss = model(data_batch)
        loss = kld + recons_loss
        loss_sum += torch.sum(loss)
        kld_sum += torch.mean(kld)
        word_count += torch.sum(count_batch)
        count_batch = torch.add(count_batch, 1e-12)
        ppx_sum += torch.sum(torch.div(loss, count_batch))
        doc_count += len(data_batch)

    print_ppx = torch.exp(loss_sum / word_count)
    print_ppx_perdoc = torch.exp(ppx_sum / doc_count)
    print_kld = kld_sum / len(dataloader)
    print('| Perplexity: {:.9f}'.format(print_ppx),
          '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
          '| KLD: {:.5}'.format(print_kld))


def train(dataloader, model, epoch_num):
    loss_sum = 0.0
    ppx_sum = 0.0
    kld_sum = 0.0
    word_count = 0
    doc_count = 0
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(epoch_num):
        for data_batch, count_batch in dataloader:
            data_batch = data_batch.float()
            kld, recons_loss = model(data_batch)
            loss = kld + recons_loss
            loss_sum += torch.sum(loss)
            kld_sum += torch.mean(kld)
            word_count += torch.sum(count_batch)
            count_batch = torch.add(count_batch, 1e-12)
            ppx_sum += torch.sum(torch.div(loss, count_batch))
            doc_count += len(data_batch)
            #
            optim.zero_grad()
            loss.mean().backward()
            optim.step()
        print_ppx = torch.exp(loss_sum / word_count)
        print_ppx_perdoc = torch.exp(ppx_sum / doc_count)
        print_kld = kld_sum / len(dataloader)
        print('| Epoch train: {:d} |'.format(epoch + 1),
              '| Perplexity: {:.9f}'.format(print_ppx),
              '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
              '| KLD: {:.5}'.format(print_kld))

# 超参数
vocab_size = 2000
batch_size = 64
n_hidden = 500
n_topic = 50
n_sample = 1
# 训练集与测试集加载
train_dataset = FeatDataset(r'data\20news\train.feat', vocab_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataset = FeatDataset(r'data\20news\test.feat', vocab_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
# 模型
model = NVDM(vocab_size, n_hidden, n_topic, n_sample)
# 训练
train(dataloader=train_dataloader, model=model, epoch_num=30)
# 测试
model.eval()
test(test_dataloader, model)

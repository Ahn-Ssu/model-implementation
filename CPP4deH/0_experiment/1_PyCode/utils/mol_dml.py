# Gyoung S. Na, Hyun Woo Kim, and Hyunju Chang,
# Machine-Guided Representation for Accurate Graph-Based Molecular Machine Learning,
# Phys. Chem. Chem. Phys., 2020, 22, 18526-18535

import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch


def get_pairs(batch):
    num_data = len(batch)
    pos_list = list()
    neg_list = list()

    for anc in batch:
        target = anc.y
        idx = random.sample(range(0, num_data), 2)

        if abs(target - batch[idx[0]].y) < abs(target - batch[idx[1]].y):
            pos_list.append(batch[idx[0]])
            neg_list.append(batch[idx[1]])
        else:
            pos_list.append(batch[idx[1]])
            neg_list.append(batch[idx[0]])

    return Batch.from_data_list(pos_list), Batch.from_data_list(neg_list)


def train(model, optimizer, data_loader):
    model.train()
    train_loss = 0

    for i, (batch) in enumerate(data_loader):
        batch_pos, batch_neg = get_pairs(batch.to_data_list())

        emb_anc = F.normalize(model(batch), p=2, dim=1)
        emb_pos = F.normalize(model(batch_pos), p=2, dim=1)
        emb_neg = F.normalize(model(batch_neg), p=2, dim=1)

        dist_ratio_x = torch.norm(emb_anc - emb_pos, dim=1) / (torch.norm(emb_anc - emb_neg, dim=1) + 1e-5)
        dist_ratio_x = -(torch.exp(-dist_ratio_x + 1) - 1)
        dist_ratio_y = torch.norm(batch.y - batch_pos.y, dim=1) / (torch.norm(batch.y - batch_neg.y, dim=1) + 1e-5)
        dist_ratio_y = -(torch.exp(-dist_ratio_y + 1) - 1)
        loss = torch.mean((dist_ratio_x - dist_ratio_y)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()

    return train_loss / len(data_loader)


def test(model, data_loader):
    model.eval()
    emb_result = list()

    with torch.no_grad():
        for batch in data_loader:
            emb = model(batch)
            emb_result.append(emb)

    return torch.cat(emb_result, dim=0)

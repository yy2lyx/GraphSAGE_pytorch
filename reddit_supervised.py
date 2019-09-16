import time

from sklearn.metrics import f1_score
import torch
from models.GraphSAGE import GraphSAGE
from models.data_utils import loadRedditFromNPZ
from models.neighbor_sampler import *


def build_reverse_index(indexes):
    max_index = max(indexes)
    reverse_map = np.arange(max_index + 1)
    for i, each in enumerate(indexes):
        reverse_map[each] = i
    return reverse_map


def evaluate(output, labels):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


# lil_matrix使用两个列表保存非零元素。data保存每行中的非零元素，rows保存非零元素所在的列。这种格式也很适合逐个添加元素，并且能快速获取行相关的数据。
# b = sparse.lil_matrix((10, 5))
# b[2, 3] = 1.0
# b[3, 4] = 2.0
# b[3, 2] = 3.0
# print b.data
# print b.rows
# [[] [] [1.0] [3.0, 2.0] [] [] [] [] [] []]
# [[] [] [3] [2, 4] [] [] [] [] [] []]


# 这里构建train函数：让adj的所有索引中不包含test和val的索引，因此得到的train_adj是完全和test和val分开的数据
# 返回的是adj.rows(还是232965行的非test和val索引的索引list)
def build_inductive_train(adj, val_index, test_index):
    val_set = set(val_index)
    test_set = set(test_index)
    rows = adj.rows
    for i, each in enumerate(rows):
        rows[i] = list(set(each).difference(val_set).difference(test_set))
    return rows


dataset_dir = "data"
adj, feats, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ(dataset_dir, add_self_loop=False)
device = torch.device("cpu")
train_adj = build_inductive_train(adj, val_index, test_index)
print(f'feats_shape:{feats.shape}\ntrain_size:{y_train.size}\ntest_size:{y_test.size}\nval_size:{y_val.size}')


def batch_gnn():
    global feats, y_train, y_val, y_test, device
    sampler_list = [10, 25]
    sampler = AdjacencySamplerFaster(train_adj, batch_size=512, num_sample_list=sampler_list)
    test_sampler = AdjacencySamplerFaster(adj, batch_size=512, num_sample_list=sampler_list)
    # sampler = AdjacencySampler(adj, batch_size=1024, num_sample_list=sampler_list)
    gnn = GraphSAGE(602, 128, concat=False, agg_type="mean")

    features = torch.Tensor(feats).to(device)
    feats = (features - features[train_index].mean(dim=0)) / features[train_index].std(dim=0)
    y_train = torch.LongTensor(y_train).to(device)
    y_val = torch.LongTensor(y_val).to(device)
    y_test = torch.LongTensor(y_test).to(device)
    # y_train = torch.cat([y_train, y_val, y_test, y_train])
    train_reverse_index = build_reverse_index(train_index)
    test_reverse_index = build_reverse_index(test_index)
    print(y_train.shape)
    print(adj.shape)
    gnn.to(device)

    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01, weight_decay=0)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.09)
    loss_fn = torch.nn.NLLLoss()

    for epoch in range(10):
        # sampler.resample()
        start_time = time.time()
        sampler.set_start_nodes(train_index)
        for i,(start_nodes, adj_list) in enumerate(sampler):
            gnn.train()
            # scheduler.step()
            optimizer.zero_grad()
            a = [feats, start_nodes, adj_list, sampler_list]
            output = gnn(feats, start_nodes, adj_list, sampler_list)
            output = torch.log_softmax(output, dim=-1)
            loss = loss_fn(output, y_train[train_reverse_index[start_nodes]])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), 25)
            optimizer.step()
            if i % 10 == 0:
                test_sampler.set_start_nodes(test_index)
                train_model(epoch, feats, gnn, sampler_list, start_time, test_reverse_index, test_sampler, y_test)


def train_model(epoch, feats, gnn, sampler_list, start_time, test_reverse_index, test_sampler, y_test):
    gnn.eval()
    predictions = None
    labels = None
    for start_nodes, adj_list in test_sampler:
        output = gnn(feats, start_nodes, adj_list, sampler_list)
        output = torch.log_softmax(output, dim=-1)
        _, indices = torch.max(output, dim=1)
        if predictions is None:
            predictions = indices
            labels = y_test[test_reverse_index[start_nodes]]
        else:
            predictions = torch.cat([predictions, indices], dim=-1)
            labels = torch.cat([labels, y_test[test_reverse_index[start_nodes]]], dim=-1)
    end_time = time.time()
    print(f"Epoch:{epoch},time:{end_time - start_time},test_acc",
          f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average="micro"))


if __name__ == "__main__":
    # test_mlp()
    batch_gnn()

    # test_feats()

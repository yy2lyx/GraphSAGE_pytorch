import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros

# 模型中聚合器类型的定义
def aggregator_map(agg_type="mean"):
    if agg_type == "mean":
        return SAGELayer
    elif agg_type == "gcn":
        return SAGEGCNLayer
    elif agg_type == "lstm":
        return SAGESeqLayer

# 整体模型框架GraphSAGE
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, out_dim, num_class=41, concat=True, agg_type="mean"):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(aggregator_map(agg_type)(in_dim, 128, act=F.relu, concat=concat))
        if concat:
            self.layers.append(aggregator_map(agg_type)(128 * 2, out_dim, concat=concat))
            self.pred_layer = nn.Linear(out_dim * 2, num_class)
        else:
            self.layers.append(aggregator_map(agg_type)(128, out_dim, concat=concat))
            self.pred_layer = nn.Linear(out_dim, num_class)

    def forward(self, feats, start_nodes, adj_list, num_sample_list):
        # feats ==> （232965，602）是所有节点的特征表示
        # start_nodes ==> 开始的节点，一共有batch_size个开始节点(512)
        # adj_list ==>带自身(第一层邻居个数，第二层邻居个数)==>(5632,146432)

        num_start_node = len(start_nodes) # 这里每次更新，后面需要用到（第二次采样就变了）
        index_neighbor_self = [] # 这里是每一次采样自身节点的邻居节点数量[140800,5120] 140800 = 512*(10+1)*25
        for each in num_sample_list:
            index_neighbor_self.insert(0, each * num_start_node)
            num_start_node = each * num_start_node + num_start_node

        output = feats[adj_list[-1]] # 这里拿到所有邻居节点＋自身的所有特征：size==>(146432,602)
        num_sample_list.reverse()

        for i, num_sample in enumerate(num_sample_list):
            # neighbor_feats ==> 关于所有邻居节点的feature（这里由于在前面的邻接矩阵采样的时候，定义了前面的索引是自身节点，而后面的索引是其邻居节点）
            neighbor_feats, self_feats = output[-index_neighbor_self[i]:], output[:-index_neighbor_self[i]]
            output = self.layers[i](neighbor_feats, self_feats, num_sample)
            print(output.detach().numpy())

        output = F.normalize(output, p=2, dim=-1)
        return self.pred_layer(output)


# 求mean模式聚合器
class SAGELayer(nn.Module):
    def __init__(self, in_channels, out_channels, act=lambda x: x, concat=True, bias=False):
        super(SAGELayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act

        # torch.nn.Parameter是Variable的一个变种， 如果没有Parameter这个类的话，那么这些临时变量也会注册成为模型变量
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.self_weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            if concat:
                self.bias = Parameter(torch.Tensor(out_channels * 2))
            else:
                self.bias = Parameter(torch.Tensor(out_channels))
        self.concat = concat
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.self_weight)
        if hasattr(self, "bias"):
            zeros(self.bias)

    def forward(self, neighbor_feats, self_feats, num_sample):
        """"""
        # shape: [num_nodes, 1, num_heads, out_dim]
        self_feats = self_feats.view(-1, self.in_channels)
        neighbor_feats = neighbor_feats.view(-1, num_sample, self.in_channels)
        # shape: [num_nodes, num_neighbor, num_heads, out_dim]

        # 这里是聚类器最关键的地方：对于每一个start节点而言，需要把需要对其邻居节点的特征求平均
        neighbor_feats = neighbor_feats.mean(dim=-2)
        out = torch.mm(neighbor_feats, self.weight).view(-1, self.out_channels)
        # out = x_j.mean(dim=-2)

        x_i = torch.mm(self_feats, self.self_weight).view(-1, self.out_channels)
        if self.concat:
            out = torch.cat([out, x_i], dim=-1)
        else:
            out = out + x_i

        if hasattr(self, "bias"):
            out = out + self.bias
        return self.act(out)


class SAGEGCNLayer(SAGELayer):
    def __init__(self, in_channels, out_channels, act=lambda x: x, concat=True, bias=False):
        super(SAGEGCNLayer, self).__init__(in_channels, out_channels, act, concat, bias)

    def forward(self, neighbor_feats, self_feats, num_sample):
        """"""
        # shape: [num_nodes, 1, num_heads, out_dim]
        self_feats = self_feats.view(-1, 1, self.in_channels)
        neighbor_feats = neighbor_feats.view(-1, num_sample, self.in_channels)
        neighbor_feats = torch.cat([neighbor_feats, self_feats], dim=-2)
        neighbor_feats = neighbor_feats.mean(dim=-2)
        out = torch.mm(neighbor_feats, self.weight).view(-1, self.out_channels)
        # out = x_j.mean(dim=-2)

        if self.concat:
            out = torch.cat([out, out], dim=-1)
        else:
            out = out

        if hasattr(self, "bias"):
            out = out + self.bias

        return self.act(out)


class SAGESeqLayer(nn.Module):
    def __init__(self, in_channels, out_channels, act=lambda x: x, concat=True, bias=False, num_hidden_layers=1, model_size="small"):
        super(SAGESeqLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.num_hidden_layers = num_hidden_layers
        if model_size == "small":
            self.hidden_dim = hidden_dim = 128
        elif model_size == "big":
            self.hidden_dim = hidden_dim = 256

        self.neigh_weight = Parameter(torch.Tensor(hidden_dim, out_channels))
        self.self_weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.lstm = nn.LSTM(in_channels, out_channels, num_hidden_layers, bias=bias, batch_first=True)
        if bias:
            if concat:
                self.bias = Parameter(torch.Tensor(out_channels * 2))
            else:
                self.bias = Parameter(torch.Tensor(out_channels))
        self.concat = concat
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.neigh_weight)
        glorot(self.self_weight)
        if hasattr(self, "bias"):
            zeros(self.bias)

    def forward(self, neighbor_feats, self_feats, num_sample):
        neighbor_feats = neighbor_feats.view(-1, num_sample, self.in_channels)

        batch_size = neighbor_feats.size(0)
        h0 = torch.zeros(self.num_hidden_layers, batch_size, self.hidden_dim).to(neighbor_feats.device)
        c0 = torch.zeros(self.num_hidden_layers, batch_size, self.hidden_dim).to(neighbor_feats.device)

        output, _ = self.lstm(neighbor_feats, (h0, c0))
        out = output[:, -1, :]
        out = out.view(-1, self.hidden_dim)
        out = torch.mm(out, self.neigh_weight).view(-1, self.out_channels)

        # shape: [num_nodes, 1, num_heads, out_dim]
        self_feats = self_feats.view(-1, self.in_channels)
        x_i = torch.mm(self_feats, self.self_weight).view(-1, self.out_channels)

        if self.concat:
            out = torch.cat([out, x_i], dim=-1)
        else:
            out = out + x_i

        if hasattr(self, "bias"):
            out = out + self.bias

        return self.act(out)

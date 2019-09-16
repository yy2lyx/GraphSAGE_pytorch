import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


class GraphSampler(object):
    def __init__(self,
                 g,
                 batch_size=512,
                 num_sample_list=[10, 25],
                 shuffle=True,
                 replace=True):
        if not isinstance(g, nx.Graph):
            raise RuntimeError("Wrong graph type")

        self.g = g
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replace = replace
        self.num_sample_list = num_sample_list
        self._num_nodes = g.number_of_nodes()
        self._num_batch = int(np.ceil(self._num_nodes / self.batch_size))

        self.adj = nx.to_scipy_sparse_matrix(self.g, format="lil").rows

    def __iter__(self):
        if self.shuffle:
            nodes = list(self.g.nodes)
            np.random.shuffle(nodes)
        for index in range(self._num_batch):
            print(index)
            start_nodes = nodes[index * self.batch_size:(index + 1) * self.batch_size]
            adj_list = []
            for num_sample in self.num_sample_list:
                start_nodes, adjs = self.build_batch(start_nodes, num_sample)
                adj_list.append(adjs)
            start_nodes = nodes[index * self.batch_size:(index + 1) * self.batch_size]
            yield start_nodes, adj_list

    def build_batch(self, start_nodes, num_sample):
        nodes = None
        adjs = torch.LongTensor(np.random.choice(self.adj[start_nodes[0]], num_sample, replace=self.replace))
        for i in range(1, len(start_nodes)):
            tmp_neoghbor = torch.LongTensor(
                np.random.choice(self.adj[start_nodes[i]], num_sample, replace=self.replace))
            nodes = torch.cat([adjs, tmp_neoghbor], dim=-1)
            adjs = torch.cat([adjs, tmp_neoghbor], dim=-2)
        return nodes, adjs

# 基类：邻接矩阵采样器
class AdjacencySampler(object):
    def __init__(self,
                 adj,
                 batch_size=512,
                 num_sample_list=[10, 25],
                 shuffle=True,
                 replace=True):
        if isinstance(adj, sp.spmatrix):
            if not isinstance(adj, sp.lil_matrix):
                adj = adj.tolil()
            self.adj = adj.rows
        else:
            self.adj = adj
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replace = replace
        self.num_sample_list = num_sample_list
        self._num_nodes = self.adj.shape[0]
        self._num_batch = int(np.ceil(self._num_nodes / self.batch_size)) # np.ceil 向上取整
        self.nodes = np.arange(self._num_nodes)


    # 这里重置，由于输入的是nodes = train_index(train 的索引)
    def set_start_nodes(self, nodes, batch_size=512):
        self.nodes = nodes
        self._num_nodes = len(nodes)
        self.batch_size = batch_size
        self._num_batch = int(np.ceil(self._num_nodes / self.batch_size))
        print(f'Total train batch num：{self._num_batch}')

    def __iter__(self):
        if self.nodes is None:
            raise RuntimeError("No start nodes, run set_start_nodes first!")
        if self.shuffle:
            nodes = self.nodes
            np.random.shuffle(nodes)
        for index in range(self._num_batch):

            start_nodes = nodes[index * self.batch_size:(index + 1) * self.batch_size]
            adj_list = []
            for num_sample in self.num_sample_list:
                start_nodes, adjs = self.build_batch(start_nodes, num_sample)
                adj_list.append(adjs)
            start_nodes = nodes[index * self.batch_size:(index + 1) * self.batch_size]
            yield start_nodes, adj_list
            # yield ==> 返回的是一个生成器函数
            # def square():
            #     for x in range(4):
            #         yield x ** 2
            #
            # square_gen = square()
            # for x in square_gen:
            #     print(x)

    def build_batch(self, start_nodes, num_sample):

        tmp_neoghbor = self.adj[start_nodes[0]]
        adjs = torch.LongTensor(np.random.choice(tmp_neoghbor, num_sample, replace=self.replace))
        for i in range(1, len(start_nodes)):
            tmp_neoghbor = torch.LongTensor(
                np.random.choice(self.adj[start_nodes[i]], num_sample, replace=self.replace))
            adjs = torch.cat([adjs, tmp_neoghbor])
        return adjs.numpy(), adjs.view([-1, num_sample])

    # sample when build batch, do not need re_sample
    def resample(self):
        pass


class AdjacencySamplerOnce(AdjacencySampler):
    def __init__(self,
                 adj,
                 batch_size=512,
                 num_sample_list=[10, 25],
                 shuffle=True,
                 replace=True):

        super(AdjacencySamplerOnce, self).__init__(adj, batch_size, num_sample_list, shuffle, replace)
        self.resample()

    def resample(self):
        self.adj = self.adj.tolist()
        for i, tmp_adj in enumerate(self.adj):
            self.adj[i] = np.random.choice(tmp_adj, sum(self.num_sample_list), replace=True).tolist()
        self.adj = torch.LongTensor(self.adj)

    def __iter__(self):
        if self.shuffle:
            nodes = self.nodes
            np.random.shuffle(nodes)
        for index in range(self._num_batch):
            start_nodes = nodes[index * self.batch_size:(index + 1) * self.batch_size]
            adj_list = []
            start_sample_index = 0
            for num_sample in self.num_sample_list:
                end_sample_index = start_sample_index + num_sample
                start_nodes, adjs = self.build_batch(start_nodes, start_sample_index, end_sample_index)
                adj_list.append(adjs)
            start_nodes = nodes[index * self.batch_size:(index + 1) * self.batch_size]
            yield start_nodes, adj_list

    def build_batch(self, start_nodes, start_sample_index, end_sample_index):
        tmp_neoghbor = self.adj[start_nodes, start_sample_index:end_sample_index]
        return tmp_neoghbor.view(-1), tmp_neoghbor


class AdjacencySamplerOnceForSage(AdjacencySampler):
    def __init__(self,
                 adj,
                 batch_size=512,
                 num_sample_list=[10, 25],
                 shuffle=True,
                 replace=True,
                 with_self_loop=True):

        super(AdjacencySamplerOnceForSage, self).__init__(adj, batch_size, num_sample_list, shuffle, replace)
        self.resample()
        self.with_self_loop = with_self_loop

    def resample(self):
        self.adj = self.adj.tolist()
        for i, tmp_adj in enumerate(self.adj):
            if len(tmp_adj) == 0:
                tmp_adj = [-1]
            self.adj[i] = np.random.choice(tmp_adj, sum(self.num_sample_list), replace=True).tolist()
        self.adj = torch.LongTensor(self.adj)
        print(self.adj)

    def __iter__(self):
        if self.shuffle:
            nodes = self.nodes
            np.random.shuffle(nodes)
        for index in range(self._num_batch):
            start_nodes = nodes[index * self.batch_size:(index + 1) * self.batch_size]
            adj_list = []
            start_sample_index = 0
            for num_sample in self.num_sample_list:
                end_sample_index = start_sample_index + num_sample
                tmp_nodes = self.build_batch(start_nodes, start_sample_index, end_sample_index)
                # start_nodes = torch.cat([torch.LongTensor(tmp_nodes), torch.LongTensor(start_nodes)])
                start_nodes = torch.cat([torch.LongTensor(start_nodes), torch.LongTensor(tmp_nodes)])
                adj_list.append(start_nodes)
            start_nodes = nodes[index * self.batch_size:(index + 1) * self.batch_size]
            yield start_nodes, adj_list

    def build_batch(self, start_nodes, start_sample_index, end_sample_index):
        tmp_neoghbor = self.adj[start_nodes, start_sample_index:end_sample_index]
        return tmp_neoghbor.view(-1)

    def build_edge_index(self):
        start_node_length = len(self.nodes)
        edge_index_list = []
        for sample_size in self.num_sample_list:
            rows = torch.arange(start=0, end=start_node_length).view(-1, 1)
            rows = rows + torch.zeros([1, sample_size], dtype=torch.int64)
            if self.with_self_loop:
                cols = torch.arange(start=0, end=start_node_length * (sample_size + 1))
                prefix = torch.arange(start=0, end=start_node_length)
                rows = torch.cat([prefix, rows.view(-1)], dim=-1)
            else:
                cols = torch.arange(start=0, end=start_node_length * (sample_size + 1))
                rows = rows.view(-1)

            edge_index_list.append([rows, cols])
            start_node_length *= sample_size
        return edge_index_list

"""利用邻接矩阵采集器采集节点邻居的时候，顺序是：先是10，再是以10为基础*25
而模型则完全相反的过程：先是将所有10*25的节点进行第一次聚合==>10，然后让这10个节点第二次聚合为1"""
class AdjacencySamplerFaster(AdjacencySampler):
    def __init__(self,
                 adj,
                 batch_size=512,
                 num_sample_list=[10, 25], # 第二次重采样，找到每个节点对应的第一层邻居10和第二层邻居25
                 max_degree=128, # 第一次重采样，找到每个节点的最大深度的邻居节点128（因为每个节点对应的邻居个数不一致，不好处理）
                 shuffle=True,
                 replace=True,
                 with_self_loop=False):

        super(AdjacencySamplerFaster, self).__init__(adj, batch_size, num_sample_list, shuffle, replace)
        self.with_self_loop = with_self_loop
        self.max_degree = max_degree
        self.resample()

    def resample(self):
        self.adj = self.adj.tolist() # listoflist[[],[]...],len=232965,内部的list的长度都是不等的:a
        a = [len(i) for i in self.adj]
        for i, tmp_adj in enumerate(self.adj):
            # 如果第i行的值不在adj的list当中，看下是否加入本身i
            if self.with_self_loop:
                if i not in tmp_adj:
                    tmp_adj.append(i)
            if len(tmp_adj) == 0:
                tmp_adj = [i]
            # 随机抽取self.max_degree个（128）
            self.adj[i] = np.random.choice(tmp_adj, self.max_degree, replace=True).tolist()
        self.adj = torch.LongTensor(self.adj)
        print(f'After resampling....adj.shape:{self.adj.shape}')

    def __iter__(self):
        if self.shuffle:
            nodes = self.nodes
            np.random.shuffle(nodes)
        for index in range(self._num_batch):
            print(f'Current Batch:{index}')
            # 这里每一个batch都有一个batchsize（512）个大小的节点输入到模型
            start_nodes = nodes[index * self.batch_size:(index + 1) * self.batch_size]
            adj_list = []
            for num_sample in self.num_sample_list:
                tmp_nodes = self.build_batch(start_nodes, num_sample) # size:5120=batch*10
                # 这里更新startnodes节点（5120+512）放到新一轮的25中
                start_nodes = torch.cat([torch.LongTensor(start_nodes), torch.LongTensor(tmp_nodes)])
                adj_list.append(start_nodes)
            start_nodes = nodes[index * self.batch_size:(index + 1) * self.batch_size]
            # 这里返回的adj_list(邻接矩阵索引列表，包含自身)：<1>(1+10)*512 = 5632 <2>(25+1)*5632 = 146432个节点索引
            yield start_nodes, adj_list

    def build_batch(self, start_nodes, num_sample):
        # 抽取所有包含start_nodes的节点
        tmp_neoghbor = self.adj[start_nodes][:, torch.randint(high=self.max_degree, size=(num_sample,))] #返回均匀分布的[low,high]之间的整数随机值
        x = tmp_neoghbor.view(-1) # 将tmp_neoghbor(512,10)转成[5120]
        return x

    def build_edge_index(self):
        start_node_length = len(self.nodes)
        edge_index_list = []
        for sample_size in self.num_sample_list:
            rows = torch.arange(start=0, end=start_node_length).view(-1, 1)
            rows = rows + torch.zeros([1, sample_size], dtype=torch.int64)
            if self.with_self_loop:
                cols = torch.arange(start=0, end=start_node_length * (sample_size + 1))
                prefix = torch.arange(start=0, end=start_node_length)
                rows = torch.cat([prefix, rows.view(-1)], dim=-1)
            else:
                cols = torch.arange(start=0, end=start_node_length * (sample_size + 1))
                rows = rows.view(-1)

            edge_index_list.append([rows, cols])
            start_node_length *= sample_size
        return edge_index_list

class AdjacencySamplerOnceWithEdgeIndex(AdjacencySampler):
    def __init__(self,
                 adj,
                 batch_size=512,
                 num_sample_list=[10, 25],
                 shuffle=True,
                 replace=True,
                 with_self_loop=True):

        super(AdjacencySamplerOnceWithEdgeIndex, self).__init__(adj, batch_size, num_sample_list, shuffle, replace)
        self.resample()
        self.with_self_loop = with_self_loop

    def resample(self):
        self.adj = self.adj.tolist()
        for i, tmp_adj in enumerate(self.adj):
            self.adj[i] = np.random.choice(tmp_adj, sum(self.num_sample_list), replace=True).tolist()
        self.adj = torch.LongTensor(self.adj)

    def __iter__(self):
        if self.shuffle:
            nodes = self.nodes
            np.random.shuffle(nodes)
        for index in range(self._num_batch):
            start_nodes = nodes[index * self.batch_size:(index + 1) * self.batch_size]
            adj_list = []
            start_sample_index = 0
            for num_sample in self.num_sample_list:
                end_sample_index = start_sample_index + num_sample
                tmp_nodes = self.build_batch(start_nodes, start_sample_index, end_sample_index)
                # start_nodes = torch.cat([torch.LongTensor(tmp_nodes), torch.LongTensor(start_nodes)])
                start_nodes = torch.cat([torch.LongTensor(start_nodes), torch.LongTensor(tmp_nodes)])
                adj_list.append(start_nodes)
            start_nodes = nodes[index * self.batch_size:(index + 1) * self.batch_size]
            yield start_nodes, adj_list, self.build_edge_index(len(start_nodes))

    def build_batch(self, start_nodes, start_sample_index, end_sample_index):
        tmp_neoghbor = self.adj[start_nodes, start_sample_index:end_sample_index]
        return tmp_neoghbor.view(-1)

    def build_edge_index(self, start_node_length):
        edge_index_list = []
        for sample_size in self.num_sample_list:
            rows = torch.arange(start=0, end=start_node_length).view(-1, 1)
            rows = rows + torch.zeros([1, sample_size], dtype=torch.int64)
            if self.with_self_loop:
                cols = torch.arange(start=0, end=start_node_length * (sample_size + 1))
                prefix = torch.arange(start=0, end=start_node_length)
                rows = torch.cat([prefix, rows.view(-1)], dim=-1)
            else:
                cols = torch.arange(start=0, end=start_node_length * (sample_size))
                cols = cols + start_node_length
                rows = rows.view(-1)

            edge_index_list.append(torch.stack([cols, rows]))
            start_node_length *= (sample_size+1)
        return edge_index_list

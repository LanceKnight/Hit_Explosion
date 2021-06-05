from torch_geometric.nn import GINEConv
import torch
from torch.nn import Embedding
import torch.nn.functional as F

num_atom_features = 2
num_elem_types = 118
num_chirality_tags = 4

num_bond_features = 2
num_bond_types = 4
num_bond_dir = 3


class GraphNet(torch.nn.Module):
    def __init__(self, num_layers, emb_dim, drop_ratio=0):
        super(GraphNet, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio

        self.mlp = torch.nn.Sequential(torch.nn.Linear(
            emb_dim, 2 * emb_dim), torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.conv = GINEConv(nn=self.mlp)

        self.x_elem_emb = Embedding(num_elem_types, emb_dim)
        self.x_chirality_emb = Embedding(num_chirality_tags, emb_dim)

        self.edge_attr_bond_type_emb = Embedding(num_bond_types, emb_dim)
        self.edge_attr_bond_dir_emb = Embedding(num_bond_dir, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_elem_emb.weight.data)
        torch.nn.init.xavier_uniform_(self.x_chirality_emb.weight.data)

        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.gnns.append(self.conv)

        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_elem_emb(x[:, 0]) + self.x_chirality_emb(x[:, 1])
        edge_attr = self.edge_attr_bond_type_emb(
            edge_attr[:, 0]) + self.edge_attr_bond_dir_emb(edge_attr[:, 1])

        h_list = [x]
        for layer in range(self.num_layers):
            # print(f'h:{h_list[layer]}, index:{edge_index}, attr:{edge_attr}')
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio,
                              training=self.training)
            h_list.append(h)

        node_representation = h_list[-1]
        return node_representation


if __name__ == '__main__':
    print('this is model1.py')
    model = GraphNet(num_layers=5, emb_dim=300)

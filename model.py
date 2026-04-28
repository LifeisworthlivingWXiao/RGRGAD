import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act_func, bias_flag=True):
        super(GCN, self).__init__()
        self.linear_map = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = nn.PReLU() if act_func == 'prelu' else act_func

        if bias_flag:
            self.bias_param = nn.Parameter(torch.FloatTensor(out_dim))
            self.bias_param.data.fill_(0.0)
        else:
            self.register_parameter('bias_param', None)

        for mod in self.modules():
            self.init_weights(mod)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, input_seq, adjacency, sparse_flag=False):
        mapped_fts = self.linear_map(input_seq)
        if sparse_flag:
            output_fts = torch.unsqueeze(torch.spmm(adjacency, torch.squeeze(mapped_fts, 0)), 0)
        else:
            output_fts = torch.bmm(adjacency, mapped_fts)
        if self.bias_param is not None:
            output_fts += self.bias_param

        return self.activation(output_fts)


class AveragePoolingReadout(nn.Module):
    def __init__(self):
        super(AveragePoolingReadout, self).__init__()

    def forward(self, input_seq):
        return torch.mean(input_seq, 1)


class MaxPoolingReadout(nn.Module):
    def __init__(self):
        super(MaxPoolingReadout, self).__init__()

    def forward(self, input_seq):
        return torch.max(input_seq, 1).values


class MinPoolingReadout(nn.Module):
    def __init__(self):
        super(MinPoolingReadout, self).__init__()

    def forward(self, input_seq):
        return torch.min(input_seq, 1).values


class WeightedSumReadout(nn.Module):
    def __init__(self):
        super(WeightedSumReadout, self).__init__()

    def forward(self, input_seq, attention_query):
        query_t = attention_query.permute(0, 2, 1)
        similarity = torch.matmul(input_seq, query_t)
        similarity = F.softmax(similarity, dim=1)
        similarity = similarity.repeat(1, 1, input_seq.size(-1))
        weighted_out = torch.mul(input_seq, similarity)
        weighted_out = torch.sum(weighted_out, 1)
        return weighted_out


class NodeDiscriminator(nn.Module):
    def __init__(self, hidden_dim, neg_samples):
        super(NodeDiscriminator, self).__init__()
        self.bilinear_layer = nn.Bilinear(hidden_dim, hidden_dim, 1)

        for mod in self.modules():
            self.init_weights(mod)

        self.negsamp_num = neg_samples

    def init_weights(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, context_embed, node_embed):
        scores_list = []
        scores_list.append(self.bilinear_layer(node_embed, context_embed))

        context_shifted = context_embed
        for _ in range(self.negsamp_num):
            context_shifted = torch.cat((context_shifted[-2:-1, :], context_shifted[:-1, :]), 0)
            scores_list.append(self.bilinear_layer(node_embed, context_shifted))

        output_logits = torch.cat(tuple(scores_list))
        return output_logits


class Model(nn.Module):
    def __init__(self, in_dim, hid_dim, act_func, neg_samples, pooling_type):
        super(Model, self).__init__()
        self.readout_mode = pooling_type
        self.gcn_layer = GCN(in_dim, hid_dim, act_func)

        if pooling_type == 'max':
            self.readout_op = MaxPoolingReadout()
        elif pooling_type == 'min':
            self.readout_op = MinPoolingReadout()
        elif pooling_type == 'avg':
            self.readout_op = AveragePoolingReadout()
        elif pooling_type == 'weighted_sum':
            self.readout_op = WeightedSumReadout()

        self.disc_model = NodeDiscriminator(hid_dim, neg_samples)

    def forward(self, feature_seq, adj_matrix, sparse_flag=False):
        h_all = self.gcn_layer(feature_seq, adj_matrix, sparse_flag)

        node_interest_h = h_all[:, -1, :]

        if self.readout_mode != 'weighted_sum':
            context_c = self.readout_op(h_all[:, : -1, :])
        else:
            context_c = self.readout_op(h_all[:, : -1, :], h_all[:, -2: -1, :])

        subgraph_embed = torch.mean(h_all[:, :-1, :], dim=1)

        logits_out = self.disc_model(context_c, node_interest_h)

        return logits_out, node_interest_h, context_c, subgraph_embed


class RoutingGate(nn.Module):
    def __init__(self, hidden_dim=16):
        super(RoutingGate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward_nodes(self, anomaly_score, degree_score):
        gate_input = torch.stack([anomaly_score, degree_score], dim=1)
        return self.mlp(gate_input).squeeze(-1)

    def forward(self, anomaly_score, degree_score):
        node_gate = self.forward_nodes(anomaly_score, degree_score)
        routing_weight = 0.5 * (node_gate.unsqueeze(1) + node_gate.unsqueeze(0))
        routing_weight = routing_weight - torch.diag_embed(torch.diag(routing_weight))
        routing_weight = torch.clamp(routing_weight, min=0.0, max=1.0)
        return node_gate, routing_weight

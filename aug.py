import torch as th
import torch.nn.functional as F
from torch_scatter import scatter_add
from utils import normalize_adj
import numpy as np
import scipy.sparse as sp
import dgl



def drop_node_features(feature_tensor, drop_prob):
    drop_mask = th.empty((feature_tensor.size(1),), dtype=th.float32, device=feature_tensor.device).uniform_(0, 1) < drop_prob
    new_feat = feature_tensor.clone()
    new_feat[:, drop_mask] = 0
    return new_feat


def degree_based_edge_masking(adj_tensor, node_indices, sim_matrix, max_degree_val, node_degrees, mask_probability):
    if node_indices.numel() == 0 or len(node_degrees) == 0:
        empty_idx = th.empty(0, dtype=th.long, device=adj_tensor.device)
        return empty_idx, empty_idx

    aug_degree = (node_degrees.float() * (1 - mask_probability)).long()
    aug_degree = th.clamp(aug_degree, min=0)

    if aug_degree.sum() == 0:
        empty_idx = th.empty(0, dtype=th.long, device=adj_tensor.device)
        return empty_idx, empty_idx

    sim_dist = sim_matrix[node_indices] * adj_tensor[node_indices]
    sim_dist = th.nan_to_num(sim_dist, nan=0.0, posinf=0.0, neginf=0.0)

    if sim_dist.size(1) == 0:
        empty_idx = th.empty(0, dtype=th.long, device=adj_tensor.device)
        return empty_idx, empty_idx

    row_sum = sim_dist.sum(dim=1, keepdim=True)
    zero_mask = (row_sum <= 0)
    if zero_mask.any():
        sim_dist = sim_dist.clone()
        sim_dist[zero_mask.squeeze(1)] = 1.0
        row_sum = sim_dist.sum(dim=1, keepdim=True)

    sim_dist = sim_dist / row_sum
    new_targets = th.multinomial(sim_dist, int(max_degree_val), replacement=True)

    target_indices = th.arange(max_degree_val, device=adj_tensor.device).unsqueeze(dim=0)
    valid_mask = (target_indices - aug_degree.unsqueeze(dim=1) < 0)

    new_col_indices = new_targets[valid_mask]
    new_row_indices = node_indices.repeat_interleave(aug_degree)

    return new_row_indices, new_col_indices


def degree_masking_with_threshold(adj_tensor, node_indices, sim_matrix, max_degree_val, node_degrees, fix_threshold):
    if node_indices.numel() == 0 or len(node_degrees) == 0:
        empty_idx = th.empty(0, dtype=th.long, device=adj_tensor.device)
        return empty_idx, empty_idx

    fixed_aug_degree = th.full_like(node_degrees, fill_value=fix_threshold)
    fixed_aug_degree = th.clamp(fixed_aug_degree, min=0)

    if fixed_aug_degree.sum() == 0:
        empty_idx = th.empty(0, dtype=th.long, device=adj_tensor.device)
        return empty_idx, empty_idx

    sim_dist = sim_matrix[node_indices] * adj_tensor[node_indices]
    sim_dist = th.nan_to_num(sim_dist, nan=0.0, posinf=0.0, neginf=0.0)

    if sim_dist.size(1) == 0:
        empty_idx = th.empty(0, dtype=th.long, device=adj_tensor.device)
        return empty_idx, empty_idx

    row_sum = sim_dist.sum(dim=1, keepdim=True)
    zero_mask = (row_sum <= 0)
    if zero_mask.any():
        sim_dist = sim_dist.clone()
        sim_dist[zero_mask.squeeze(1)] = 1.0
        row_sum = sim_dist.sum(dim=1, keepdim=True)

    sim_dist = sim_dist / row_sum
    new_targets = th.multinomial(sim_dist, int(max_degree_val), replacement=True)

    target_indices = th.arange(max_degree_val, device=adj_tensor.device).unsqueeze(dim=0)
    valid_mask = (target_indices - fixed_aug_degree.unsqueeze(dim=1) < 0)

    new_col_indices = new_targets[valid_mask]
    new_row_indices = node_indices.repeat_interleave(fixed_aug_degree)

    return new_row_indices, new_col_indices



def get_routing_weight(anomaly_score):
    """
    anomaly_score: [N], where larger values indicate more anomalous nodes.
    return: [N, N] routing weight matrix.
    """
    score_col = anomaly_score.view(-1, 1)
    score_row = anomaly_score.view(1, -1)
    routing = 0.5 * (score_col + score_row)
    routing = routing - th.diag_embed(th.diag(routing))
    routing = th.clamp(routing, min=0.0)
    return routing


def neighbor_mixup(source_indices, dest_indices, adj_dist, sim_matrix,
                   max_degree_val, augmentation_degree, device_id):
    if source_indices.numel() == 0 or dest_indices.numel() == 0 or augmentation_degree.numel() == 0:
        empty_idx = th.empty(0, dtype=th.long, device=device_id)
        return empty_idx, empty_idx

    augmentation_degree = augmentation_degree.long()
    augmentation_degree = th.clamp(augmentation_degree, min=0)

    if augmentation_degree.sum() == 0:
        empty_idx = th.empty(0, dtype=th.long, device=device_id)
        return empty_idx, empty_idx

    phi_coeff = sim_matrix[source_indices, dest_indices].unsqueeze(dim=1).to(device_id)
    phi_coeff = th.clamp(phi_coeff, 0, 0.5)

    mix_dist = sim_matrix[dest_indices] * adj_dist[dest_indices] * phi_coeff + \
               sim_matrix[source_indices] * adj_dist[source_indices] * (1 - phi_coeff)

    mix_dist = th.nan_to_num(mix_dist, nan=0.0, posinf=0.0, neginf=0.0)

    if mix_dist.size(1) == 0:
        empty_idx = th.empty(0, dtype=th.long, device=device_id)
        return empty_idx, empty_idx

    row_sum = mix_dist.sum(dim=1, keepdim=True)
    zero_mask = (row_sum <= 0)
    if zero_mask.any():
        mix_dist = mix_dist.clone()
        mix_dist[zero_mask.squeeze(1)] = 1.0
        row_sum = mix_dist.sum(dim=1, keepdim=True)

    mix_dist = mix_dist / row_sum
    new_targets = th.multinomial(mix_dist, int(max_degree_val), replacement=True)

    target_indices = th.arange(max_degree_val, device=device_id).unsqueeze(dim=0)
    valid_mask = (target_indices - augmentation_degree.unsqueeze(dim=1) < 0)

    new_col_indices = new_targets[valid_mask]
    new_row_indices = source_indices.repeat_interleave(augmentation_degree)

    return new_row_indices, new_col_indices


def redundancy_pruning(dgl_graph, adj_dist, sim_matrix, features_input, node_degrees, feat_drop_prob_1,
                       feat_drop_prob_2, degree_threshold):
    feat_v1 = drop_node_features(features_input, feat_drop_prob_1)
    feat_v2 = drop_node_features(features_input, feat_drop_prob_2)

    max_degree_val = int(np.max(node_degrees)) if len(node_degrees) > 0 else 1
    device = adj_dist.device

    low_deg_idx = th.LongTensor(np.argwhere(node_degrees < degree_threshold).flatten()).to(device)
    high_deg_idx = th.LongTensor(np.argwhere(node_degrees >= degree_threshold).flatten()).to(device)

    mix_node_degree = node_degrees[(node_degrees <= degree_threshold) & (node_degrees > 2)]
    mask_node_degree = node_degrees[node_degrees >= degree_threshold]

    mix_node_degree = th.LongTensor(mix_node_degree).to(device)
    mask_node_degree = th.LongTensor(mask_node_degree).to(device)

    # Keep the original edges when no low-degree node candidates are available.
    src_nodes, dst_nodes = dgl_graph.edges()
    src_nodes = src_nodes.to(device)
    dst_nodes = dst_nodes.to(device)

    if low_deg_idx.numel() == 0 or mix_node_degree.numel() == 0:
        mix_rows = src_nodes
        mix_cols = dst_nodes
    else:
        degree_dist_counts = scatter_add(
            th.ones(mix_node_degree.size(0), device=device),
            mix_node_degree
        )

        if degree_dist_counts.numel() == 0:
            aug_deg_mix = th.empty(0, dtype=th.long, device=device)
        else:
            deg_prob = degree_dist_counts.float()
            deg_prob = th.nan_to_num(deg_prob, nan=0.0, posinf=0.0, neginf=0.0)

            if deg_prob.sum() <= 0:
                deg_prob = th.ones_like(deg_prob) / deg_prob.numel()
            else:
                deg_prob = deg_prob / deg_prob.sum()

            aug_deg_mix = th.multinomial(
                deg_prob.unsqueeze(0).repeat(low_deg_idx.size(0), 1),
                1,
                replacement=True
            ).flatten()

        src_edge_indices = np.where(np.isin(src_nodes.cpu().numpy(), low_deg_idx.cpu().numpy()))[0]
        dst_edge_indices = np.where(np.isin(dst_nodes.cpu().numpy(), low_deg_idx.cpu().numpy()))[0]
        combined_indices = np.unique(np.concatenate((src_edge_indices, dst_edge_indices))) if (
            len(src_edge_indices) + len(dst_edge_indices) > 0
        ) else np.array([], dtype=np.int64)

        if combined_indices.size == 0:
            mix_rows = src_nodes
            mix_cols = dst_nodes
        else:
            mix_rows, mix_cols = src_nodes[combined_indices], dst_nodes[combined_indices]

    if high_deg_idx.numel() == 0 or mask_node_degree.numel() == 0:
        mask_rows = th.empty(0, dtype=th.long, device=device)
        mask_cols = th.empty(0, dtype=th.long, device=device)
    else:
        mask_rows, mask_cols = degree_masking_with_threshold(
            adj_dist, high_deg_idx, sim_matrix, max_degree_val,
            mask_node_degree, degree_threshold
        )

    new_src_nodes = th.cat((mix_rows, mask_rows)).cpu()
    new_dst_nodes = th.cat((mix_cols, mask_cols)).cpu()

    new_dgl_graph = dgl.graph((new_src_nodes, new_dst_nodes), num_nodes=dgl_graph.number_of_nodes())
    new_dgl_graph = dgl.remove_self_loop(new_dgl_graph)
    new_dgl_graph = dgl.add_self_loop(new_dgl_graph)

    new_adj_dense = new_dgl_graph.adjacency_matrix().to_dense()
    new_adj_sparse = sp.csr_matrix(new_adj_dense.cpu().numpy())
    new_adj_norm = normalize_adj(new_adj_sparse)
    new_adj_matrix = (new_adj_norm + sp.eye(new_adj_norm.shape[0])).todense()
    new_adj_tensor = th.FloatTensor(np.asarray(new_adj_matrix)[np.newaxis])

    return new_dgl_graph, feat_v1.unsqueeze(0), feat_v2.unsqueeze(0), new_adj_tensor


def neighbor_completion(dgl_graph, adj_dist, sim_matrix, ano_sim_matrix, anomaly_score, features_input, node_degrees,
                        feat_drop_prob_1, edge_mask_rate_1, feat_drop_prob_2, edge_mask_rate_2,
                        degree_threshold, device_id, routing_mode='full', routing_weight_override=None):
    feat_v1 = drop_node_features(features_input, feat_drop_prob_1)
    feat_v2 = drop_node_features(features_input, feat_drop_prob_2)

    max_degree_val = int(np.max(node_degrees)) if len(node_degrees) > 0 else 1
    low_deg_idx = th.LongTensor(np.argwhere(node_degrees < degree_threshold).flatten()).to(device_id)
    high_deg_idx = th.LongTensor(np.argwhere(node_degrees >= degree_threshold).flatten()).to(device_id)
    mix_node_degree = node_degrees[(node_degrees <= degree_threshold) & (node_degrees > 2)]
    mask_node_degree = node_degrees[node_degrees >= degree_threshold]

    feat_sim = sim_matrix

    if routing_weight_override is None:
        routing_weight = get_routing_weight(anomaly_score).to(device_id)
    else:
        routing_weight = routing_weight_override.to(device_id)

    if routing_mode == 'feat_only':
        sim_matrix = feat_sim.clone()
    elif routing_mode == 'feat_ano':
        sim_matrix = feat_sim * ano_sim_matrix
    elif routing_mode == 'feat_routing':
        sim_matrix = feat_sim * routing_weight
    elif routing_mode == 'full':
        sim_matrix = feat_sim * ano_sim_matrix * routing_weight
    else:
        sim_matrix = feat_sim.clone()

    sim_matrix = th.clamp(sim_matrix, 0, 1)

    mix_node_degree = th.LongTensor(mix_node_degree).to(device_id)
    mask_node_degree = th.LongTensor(mask_node_degree).to(device_id)

    if low_deg_idx.numel() == 0:
        dest_mix_idx = th.empty(0, dtype=th.long, device=device_id)
        aug_deg_mix = th.empty(0, dtype=th.long, device=device_id)
    else:
        low_deg_sim = sim_matrix[low_deg_idx]
        low_deg_sim = th.nan_to_num(low_deg_sim, nan=0.0, posinf=0.0, neginf=0.0)

        row_sum = low_deg_sim.sum(dim=1, keepdim=True)
        zero_mask = (row_sum <= 0)
        if zero_mask.any():
            low_deg_sim = low_deg_sim.clone()
            low_deg_sim[zero_mask.squeeze(1)] = 1.0
            row_sum = low_deg_sim.sum(dim=1, keepdim=True)
        low_deg_sim = low_deg_sim / row_sum

        dest_mix_idx = th.multinomial(low_deg_sim, 1, replacement=True).flatten()

        if mix_node_degree.numel() == 0:
            aug_deg_mix = th.zeros(low_deg_idx.size(0), dtype=th.long, device=device_id)
        else:
            degree_dist_counts = scatter_add(
                th.ones(mix_node_degree.size(0), device=device_id),
                mix_node_degree
            )

            if degree_dist_counts.numel() == 0:
                aug_deg_mix = th.zeros(low_deg_idx.size(0), dtype=th.long, device=device_id)
            else:
                deg_prob = degree_dist_counts.float()
                deg_prob = th.nan_to_num(deg_prob, nan=0.0, posinf=0.0, neginf=0.0)
                if deg_prob.sum() <= 0:
                    deg_prob = th.ones_like(deg_prob) / deg_prob.numel()
                else:
                    deg_prob = deg_prob / deg_prob.sum()

                aug_deg_mix = th.multinomial(
                    deg_prob.unsqueeze(0).repeat(low_deg_idx.size(0), 1),
                    1,
                    replacement=True
                ).flatten()

    mix_rows_1, mix_cols_1 = neighbor_mixup(
        low_deg_idx, dest_mix_idx, adj_dist, sim_matrix,
        max_degree_val, aug_deg_mix, device_id
    )
    mask_rows_1, mask_cols_1 = degree_based_edge_masking(
        adj_dist, high_deg_idx, sim_matrix, max_degree_val,
        mask_node_degree, edge_mask_rate_1
    )

    src_nodes_1 = th.cat((mix_rows_1, mask_rows_1)).cpu()
    dst_nodes_1 = th.cat((mix_cols_1, mask_cols_1)).cpu()

    if src_nodes_1.numel() == 0 or dst_nodes_1.numel() == 0:
        src0, dst0 = dgl_graph.edges()
        src_nodes_1, dst_nodes_1 = src0.cpu(), dst0.cpu()

    new_dgl_graph_1 = dgl.graph((src_nodes_1, dst_nodes_1), num_nodes=dgl_graph.number_of_nodes())
    new_dgl_graph_1 = dgl.remove_self_loop(new_dgl_graph_1)
    new_dgl_graph_1 = dgl.add_self_loop(new_dgl_graph_1)

    mix_rows_2, mix_cols_2 = neighbor_mixup(
        low_deg_idx, dest_mix_idx, adj_dist, sim_matrix,
        max_degree_val, aug_deg_mix, device_id
    )
    mask_rows_2, mask_cols_2 = degree_based_edge_masking(
        adj_dist, high_deg_idx, sim_matrix, max_degree_val,
        mask_node_degree, edge_mask_rate_2
    )

    src_nodes_2 = th.cat((mix_rows_2, mask_rows_2)).cpu()
    dst_nodes_2 = th.cat((mix_cols_2, mask_cols_2)).cpu()

    if src_nodes_2.numel() == 0 or dst_nodes_2.numel() == 0:
        src0, dst0 = dgl_graph.edges()
        src_nodes_2, dst_nodes_2 = src0.cpu(), dst0.cpu()

    new_dgl_graph_2 = dgl.graph((src_nodes_2, dst_nodes_2), num_nodes=dgl_graph.number_of_nodes())
    new_dgl_graph_2 = dgl.remove_self_loop(new_dgl_graph_2)
    new_dgl_graph_2 = dgl.add_self_loop(new_dgl_graph_2)

    adj_dense_1 = new_dgl_graph_1.adjacency_matrix().to_dense()
    adj_sparse_1 = sp.csr_matrix(adj_dense_1.cpu().numpy())
    adj_norm_1 = normalize_adj(adj_sparse_1)
    adj_matrix_1 = (adj_norm_1 + sp.eye(adj_norm_1.shape[0])).todense()
    adj_tensor_1 = th.FloatTensor(np.asarray(adj_matrix_1)[np.newaxis])

    adj_dense_2 = new_dgl_graph_2.adjacency_matrix().to_dense()
    adj_sparse_2 = sp.csr_matrix(adj_dense_2.cpu().numpy())
    adj_norm_2 = normalize_adj(adj_sparse_2)
    adj_matrix_2 = (adj_norm_2 + sp.eye(adj_norm_2.shape[0])).todense()
    adj_tensor_2 = th.FloatTensor(np.asarray(adj_matrix_2)[np.newaxis])

    return new_dgl_graph_1, new_dgl_graph_2, feat_v1.unsqueeze(0), feat_v2.unsqueeze(0), adj_tensor_1, adj_tensor_2

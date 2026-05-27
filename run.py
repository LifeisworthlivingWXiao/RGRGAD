import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import dgl
import scipy.sparse as sp

from model import Model, RoutingGate
from utils import load_mat, preprocess_features, adj_to_dgl_graph, normalize_adj, generate_rwr_subgraph
from aug import redundancy_pruning, neighbor_completion

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    parser = argparse.ArgumentParser(description='RGRGAD: Routing-Guided Refinement for Graph Anomaly Detection')

    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--data_dir', type=str, default='./Data', help='Dataset directory')
    parser.add_argument('--exp_name', type=str, default='full_model', help='Experiment name')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--train_epoch', type=int, default=100)
    parser.add_argument('--test_rounds', type=int, default=196)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--fast_cuda', action='store_true')
    parser.add_argument('--amp', action='store_true')

    parser.add_argument('--subgraph_size', type=int, default=2)
    parser.add_argument('--readout', type=str, default='avg')
    parser.add_argument('--neg_sam_rat', type=int, default=1)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--threshold', type=int, default=8)
    parser.add_argument('--routing_mode', type=str, default='full',
                        choices=['feat_only', 'feat_ano', 'feat_routing', 'full'])
    parser.add_argument('--train_stage', type=str, default='staged',
                        choices=['pruning_only', 'completion_only', 'staged'])
    parser.add_argument('--routing_warmup', type=int, default=20)
    parser.add_argument('--ema_momentum', type=float, default=0.9)
    parser.add_argument('--curriculum_power', type=float, default=1.0)

    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--lambda_gate', type=float, default=0.05)
    parser.add_argument('--lambda_decouple', type=float, default=0.05)
    parser.add_argument('--decouple_margin', type=float, default=1.0)
    parser.add_argument('--conf_low_q', type=float, default=0.30)
    parser.add_argument('--conf_high_q', type=float, default=0.70)
    parser.add_argument('--lambda_proto', type=float, default=0.02)
    parser.add_argument('--lambda_boundary', type=float, default=0.02)
    parser.add_argument('--proto_momentum', type=float, default=0.95)

    return parser.parse_args()


def set_seed(seed, deterministic=True):
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = (not deterministic)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def select_augmentation_stage(epoch, total_epochs, stage_mode, curriculum_power):
    if stage_mode == 'pruning_only':
        return True
    elif stage_mode == 'completion_only':
        return False
    else:
        completion_ratio = ((epoch + 1) / float(total_epochs)) ** curriculum_power
        return random.random() > completion_ratio


def build_subgraph_batch(indices, subg_set, adj_view, feat_view, subgraph_size, feature_size, device):
    batch_adj_list, batch_feat_list = [], []
    current_batch_size = len(indices)

    pad_adj_row_b = torch.zeros((current_batch_size, 1, subgraph_size), device=device)
    pad_adj_col_b = torch.zeros((current_batch_size, subgraph_size + 1, 1), device=device)
    pad_adj_col_b[:, -1, :] = 1.
    pad_feat_row_b = torch.zeros((current_batch_size, 1, feature_size), device=device)

    for i in indices:
        batch_adj_list.append(adj_view[:, subg_set[i], :][:, :, subg_set[i]])
        batch_feat_list.append(feat_view[:, subg_set[i], :])

    b_adj = torch.cat(batch_adj_list).to(device)
    b_adj = torch.cat((b_adj, pad_adj_row_b), dim=1)
    b_adj = torch.cat((b_adj, pad_adj_col_b), dim=2)

    b_feat = torch.cat(batch_feat_list).to(device)
    b_feat = torch.cat((b_feat[:, :-1, :], pad_feat_row_b, b_feat[:, -1:, :]), dim=1)

    return b_adj, b_feat


def get_confidence_thresholds(pseudo_ano_score, low_q, high_q):
    low_thr = torch.quantile(pseudo_ano_score.detach(), low_q)
    high_thr = torch.quantile(pseudo_ano_score.detach(), high_q)
    if high_thr < low_thr:
        high_thr = low_thr
    return low_thr, high_thr


def compute_total_loss(h_v1, h_v2, g_v1, g_v2, logits_v1, logits_v2, batch_labels,
                       pseudo_ano_score, ano_sim_matrix, node_gate, indices, args, device,
                       conf_low_thr, conf_high_thr, normal_prototype, b_xent, cross_ent_loss):
    loss_dict = {}

    loss_anomaly = b_xent(logits_v1, batch_labels) + b_xent(logits_v2, batch_labels)
    loss_dict["bce_raw"] = loss_anomaly
    loss_dict["bce"] = torch.mean(loss_anomaly)

    pred_h_12, pred_h_21 = torch.mm(h_v1, h_v2.T), torch.mm(h_v2, h_v1.T)
    pred_g_12, pred_g_21 = torch.mm(g_v1, g_v2.T), torch.mm(g_v2, g_v1.T)

    cl_labels = torch.arange(pred_h_12.shape[0], device=device)
    loss_dict["contrastive"] = (
                                       cross_ent_loss(pred_h_12 / args.tau, cl_labels) +
                                       cross_ent_loss(pred_h_21 / args.tau, cl_labels) +
                                       cross_ent_loss(pred_g_12 / args.tau, cl_labels) +
                                       cross_ent_loss(pred_g_21 / args.tau, cl_labels)
                               ) / 4

    pseudo_batch = pseudo_ano_score[indices].detach()
    conf_normal_mask = (pseudo_batch <= conf_low_thr).float()
    conf_anomaly_mask = (pseudo_batch >= conf_high_thr).float()
    conf_mask = ((conf_normal_mask + conf_anomaly_mask) > 0).float()

    mean_h, mean_g = 0.5 * (h_v1 + h_v2), 0.5 * (g_v1 + g_v2)
    sg_dist = torch.norm(mean_h - mean_g, p=2, dim=1)

    normal_w = conf_normal_mask * (1.0 - pseudo_batch)
    anomaly_w = conf_anomaly_mask * pseudo_batch

    loss_align_normal = torch.sum(normal_w * (sg_dist ** 2)) / (torch.sum(normal_w) + 1e-12)
    loss_separate_anomaly = torch.sum(anomaly_w * (F.relu(args.decouple_margin - sg_dist) ** 2)) / (
            torch.sum(anomaly_w) + 1e-12)
    loss_dict["decouple"] = loss_align_normal + loss_separate_anomaly

    gate_batch = node_gate[indices]
    gate_pair = 0.5 * (gate_batch.unsqueeze(1) + gate_batch.unsqueeze(0))
    gate_target = ano_sim_matrix[indices][:, indices].detach()
    pair_mask = conf_mask.unsqueeze(1) * conf_mask.unsqueeze(0)
    gate_diff_sq = (gate_pair - gate_target) ** 2
    loss_dict["routing_gate"] = torch.sum(pair_mask * gate_diff_sq) / (torch.sum(pair_mask) + 1e-12)

    z_batch = 0.5 * (mean_h + mean_g)
    if torch.sum(conf_normal_mask) > 0:
        batch_proto = torch.sum(z_batch * conf_normal_mask.unsqueeze(1), dim=0) / (torch.sum(conf_normal_mask) + 1e-12)
        updated_prototype = args.proto_momentum * normal_prototype + (1.0 - args.proto_momentum) * batch_proto.detach()
    else:
        updated_prototype = normal_prototype

    proto_ref = updated_prototype.detach()
    proto_dist = torch.norm(z_batch - proto_ref.unsqueeze(0), p=2, dim=1)
    loss_dict["proto"] = torch.sum(conf_normal_mask * (proto_dist ** 2)) / (torch.sum(conf_normal_mask) + 1e-12)
    loss_dict["boundary"] = torch.sum(conf_anomaly_mask * (F.relu(args.decouple_margin - proto_dist) ** 2)) / (
            torch.sum(conf_anomaly_mask) + 1e-12)

    loss_dict["total"] = (
            loss_dict["bce"]
            + args.alpha * loss_dict["contrastive"]
            + args.lambda_decouple * loss_dict["decouple"]
            + args.lambda_gate * loss_dict["routing_gate"]
            + args.lambda_proto * loss_dict["proto"]
            + args.lambda_boundary * loss_dict["boundary"]
    )
    loss_dict["updated_prototype"] = updated_prototype
    return loss_dict


def clone_state_dict_to_cpu(module):
    """Keep the best model in memory without writing checkpoints to disk."""
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=(not args.fast_cuda))

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'[RGRGAD] dataset={args.dataset} exp={args.exp_name} seed={args.seed}')
    print(f'[RGRGAD] device={device} mode={args.routing_mode} stage={args.train_stage}')

    adj_matrix, feat_data, anomaly_labels = load_mat(args.dataset, args.data_dir)
    feat_data, _ = preprocess_features(feat_data)

    dgl_g = adj_to_dgl_graph(adj_matrix).to(device)

    num_nodes = feat_data.shape[0]
    feature_size = feat_data.shape[1]

    adj_tens = dgl_g.adjacency_matrix().to_dense().clone().detach().to(device)
    node_degrees = adj_tens.sum(0).detach().cpu().numpy().squeeze()
    degree_tensor = torch.tensor(node_degrees, dtype=torch.float32, device=device)
    degree_tensor = (degree_tensor - degree_tensor.min()) / (degree_tensor.max() - degree_tensor.min() + 1e-12)

    adj_tens = adj_tens + torch.eye(adj_tens.size(0), device=device)
    adj_matrix_norm = normalize_adj(adj_matrix)
    adj_matrix_norm = (adj_matrix_norm + sp.eye(adj_matrix_norm.shape[0])).todense()

    feat_data_tensor = torch.FloatTensor(np.asarray(feat_data)[np.newaxis]).to(device)
    adj_matrix_tensor = torch.FloatTensor(np.asarray(adj_matrix_norm)[np.newaxis]).to(device)

    gcl_model = Model(feature_size, args.embedding_dim, 'prelu', args.neg_sam_rat, args.readout).to(device)
    routing_gate = RoutingGate(hidden_dim=16).to(device)

    optimizer = torch.optim.Adam(
        list(gcl_model.parameters()) + list(routing_gate.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    amp_enabled = args.amp and torch.cuda.is_available()
    scaler = GradScaler(enabled=amp_enabled)

    b_xent = nn.BCEWithLogitsLoss(
        reduction='none',
        pos_weight=torch.tensor([args.neg_sam_rat], device=device)
    )
    cross_ent_loss = nn.CrossEntropyLoss().to(device)

    best_loss = 1e9
    best_ep = 0
    best_gcl_state = None
    num_batches = num_nodes // args.batch_size + 1

    ano_sim_matrix = torch.ones((num_nodes, num_nodes), device=device)
    ema_pseudo_score = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    normal_prototype = torch.zeros(args.embedding_dim, dtype=torch.float32, device=device)
    epoch_node_loss_buffer = torch.zeros((num_nodes, 1), device=device)
    loss_hist_list = []

    print("Precomputing global feature similarity matrix...")
    with torch.no_grad():
        norm_raw_features = F.normalize(feat_data_tensor.squeeze(), p=2, dim=1)
        norm_raw_features_cpu = norm_raw_features.cpu()
        sim_matrix_cpu = torch.mm(norm_raw_features_cpu, norm_raw_features_cpu.t())
        sim_matrix = sim_matrix_cpu.to(norm_raw_features.device)

        del norm_raw_features_cpu
        torch.cuda.empty_cache()
        sim_matrix = torch.clamp(sim_matrix, 0, 1)
        sim_matrix = sim_matrix - torch.diag_embed(torch.diag(sim_matrix))
        sim_matrix = sim_matrix.detach().to(device)

    with tqdm(total=args.train_epoch, desc='Training', ncols=100) as prog_bar:
        for epoch in range(args.train_epoch):
            raw_pseudo_score = epoch_node_loss_buffer.squeeze().detach()
            raw_pseudo_score = (raw_pseudo_score - raw_pseudo_score.min()) / (
                    raw_pseudo_score.max() - raw_pseudo_score.min() + 1e-12
            )

            ema_pseudo_score = raw_pseudo_score if epoch == 0 else (
                    args.ema_momentum * ema_pseudo_score + (1.0 - args.ema_momentum) * raw_pseudo_score
            )

            pseudo_ano_score = torch.full_like(ema_pseudo_score,
                                               0.5) if epoch < args.routing_warmup else ema_pseudo_score
            conf_low_thr, conf_high_thr = get_confidence_thresholds(
                pseudo_ano_score, args.conf_low_q, args.conf_high_q
            )

            with torch.no_grad():
                _, learned_routing_weight = routing_gate(pseudo_ano_score, degree_tensor)
                node_gate_epoch = routing_gate.forward_nodes(pseudo_ano_score, degree_tensor)

            epoch_node_loss_buffer = torch.zeros((num_nodes, 1), device=device)
            gcl_model.train()

            all_indices = list(range(num_nodes))
            random.shuffle(all_indices)

            total_epoch_bce_loss = 0.
            cur_loss_matrix = torch.zeros((num_nodes, 2), device=device)

            do_pruning = select_augmentation_stage(
                epoch, args.train_epoch, args.train_stage, args.curriculum_power
            )

            if do_pruning:
                graph_v1, adj_v1 = dgl_g, adj_matrix_tensor
                graph_v2, feature_v1, feature_v2, adj_v2 = redundancy_pruning(
                    dgl_g, adj_tens, sim_matrix, feat_data_tensor.squeeze(),
                    node_degrees, 0.2, 0.2, args.threshold
                )
            else:
                graph_v1, graph_v2, feature_v1, feature_v2, adj_v1, adj_v2 = neighbor_completion(
                    dgl_g, adj_tens, sim_matrix, ano_sim_matrix, pseudo_ano_score,
                    feat_data_tensor.squeeze(), node_degrees,
                    0.2, 0.2, 0.2, 0.2, args.threshold, device,
                    routing_mode=args.routing_mode,
                    routing_weight_override=learned_routing_weight
                )

            subg_set1 = generate_rwr_subgraph(graph_v1, args.subgraph_size)
            subg_set2 = generate_rwr_subgraph(graph_v2, args.subgraph_size)

            for batch_num_idx in range(num_batches):
                optimizer.zero_grad(set_to_none=True)

                is_last_batch = (batch_num_idx == (num_batches - 1))
                start_idx = batch_num_idx * args.batch_size
                end_idx = None if is_last_batch else (batch_num_idx + 1) * args.batch_size
                indices = all_indices[start_idx:end_idx]

                if len(indices) == 0:
                    continue

                current_batch_size = len(indices)
                batch_labels = torch.cat((
                    torch.ones(current_batch_size),
                    torch.zeros(current_batch_size * args.neg_sam_rat)
                )).unsqueeze(1).to(device)

                b_adj_v1, b_feat_v1 = build_subgraph_batch(
                    indices, subg_set1, adj_v1, feature_v1,
                    args.subgraph_size, feature_size, device
                )
                b_adj_v2, b_feat_v2 = build_subgraph_batch(
                    indices, subg_set2, adj_v2, feature_v2,
                    args.subgraph_size, feature_size, device
                )

                with autocast(enabled=amp_enabled):
                    logits_v1, h_v1, _, g_v1 = gcl_model(b_feat_v1, b_adj_v1)
                    logits_v2, h_v2, _, g_v2 = gcl_model(b_feat_v2, b_adj_v2)

                    loss_dict = compute_total_loss(
                        h_v1, h_v2, g_v1, g_v2, logits_v1, logits_v2, batch_labels,
                        pseudo_ano_score, ano_sim_matrix, node_gate_epoch, indices,
                        args, device, conf_low_thr, conf_high_thr, normal_prototype,
                        b_xent, cross_ent_loss
                    )

                normal_prototype = loss_dict["updated_prototype"].detach()

                scaler.scale(loss_dict["total"]).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_bce_loss_val = loss_dict["bce"].detach().item()
                epoch_node_loss_buffer[indices] = loss_dict["bce_raw"][:current_batch_size].detach()

                cur_loss_matrix[indices] = torch.cat((
                    loss_dict["bce_raw"][:current_batch_size].detach(),
                    loss_dict["bce_raw"][current_batch_size:].detach()
                ), dim=1)

                if not is_last_batch:
                    total_epoch_bce_loss += batch_bce_loss_val

            mean_epoch_bce = (
                                     total_epoch_bce_loss * args.batch_size + batch_bce_loss_val * current_batch_size
                             ) / num_nodes

            loss_hist_list.append(cur_loss_matrix)
            window_size = 5
            if len(loss_hist_list) >= window_size:
                sim_calc_matrix = torch.cat(loss_hist_list[-window_size:], dim=1)
                mean_pos = torch.mean(sim_calc_matrix[:, 0::2], dim=1)
                var_pos = torch.var(sim_calc_matrix[:, 0::2], dim=1)
                mean_neg = torch.mean(sim_calc_matrix[:, 1::2], dim=1)
                var_neg = torch.var(sim_calc_matrix[:, 1::2], dim=1)
                sim_calc_matrix = torch.cat([
                    sim_calc_matrix,
                    mean_pos.unsqueeze(1),
                    var_pos.unsqueeze(1),
                    mean_neg.unsqueeze(1),
                    var_neg.unsqueeze(1)
                ], dim=1)
                ano_sim_matrix = torch.sigmoid(torch.mm(sim_calc_matrix, sim_calc_matrix.t()) * 0.07)
                loss_hist_list = loss_hist_list[-window_size:]

            if mean_epoch_bce < best_loss:
                best_loss = mean_epoch_bce
                best_ep = epoch
                best_gcl_state = clone_state_dict_to_cpu(gcl_model)

            prog_bar.set_postfix(bce_loss=f"{mean_epoch_bce:.4f}")
            prog_bar.update(1)

            if 'graph_v1' in locals():
                del graph_v1, graph_v2, feature_v1, feature_v2, adj_v1, adj_v2
            torch.cuda.empty_cache()

    print(f'Loading best epoch in memory: {best_ep}')
    if best_gcl_state is not None:
        gcl_model.load_state_dict(best_gcl_state)
    gcl_model.eval()

    multi_round_scores = np.zeros((args.test_rounds, num_nodes), dtype=np.float32)

    with tqdm(total=args.test_rounds, desc='Testing', ncols=100) as prog_bar_test:
        for round_idx in range(args.test_rounds):
            all_indices = list(range(num_nodes))
            random.shuffle(all_indices)
            subg_set = generate_rwr_subgraph(dgl_g, args.subgraph_size)

            for batch_num_idx in range(num_batches):
                is_last_batch = (batch_num_idx == (num_batches - 1))
                start_idx = batch_num_idx * args.batch_size
                end_idx = None if is_last_batch else (batch_num_idx + 1) * args.batch_size
                indices = all_indices[start_idx:end_idx]

                if len(indices) == 0:
                    continue

                current_batch_size = len(indices)
                b_adj, b_feat = build_subgraph_batch(
                    indices, subg_set, adj_matrix_tensor, feat_data_tensor,
                    args.subgraph_size, feature_size, device
                )

                with torch.no_grad():
                    with autocast(enabled=amp_enabled):
                        logits_out = gcl_model(b_feat, b_adj)[0]
                        logits_out = torch.sigmoid(torch.squeeze(logits_out))

                score_batch = - (
                        logits_out[:current_batch_size] - logits_out[current_batch_size:]).detach().cpu().numpy()
                multi_round_scores[round_idx, indices] = score_batch

            prog_bar_test.update(1)

    final_anomaly_scores = np.mean(multi_round_scores, axis=0)
    auc_result = roc_auc_score(anomaly_labels, final_anomaly_scores)

    print('====================================')
    print(f'[RGRGAD] FINAL RESULT')
    print(f'Dataset    : {args.dataset}')
    print(f'Seed       : {args.seed}')
    print(f'Best epoch : {best_ep}')
    print(f'Best loss  : {best_loss:.6f}')
    print(f'ROC-AUC    : {auc_result:.4f}')
    print('====================================')


if __name__ == '__main__':
    main()

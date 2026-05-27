import os
import random
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import torch as th
import dgl


def sparse_to_tuple_rep(sparse_matrix, add_batch_dim=False):
    def matrix_to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if add_batch_dim:
            coordinates = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values_data = mx.data
            matrix_shape = (1,) + mx.shape
        else:
            coordinates = np.vstack((mx.row, mx.col)).transpose()
            values_data = mx.data
            matrix_shape = mx.shape
        return coordinates, values_data, matrix_shape

    if isinstance(sparse_matrix, list):
        for i in range(len(sparse_matrix)):
            sparse_matrix[i] = matrix_to_tuple(sparse_matrix[i])
    else:
        sparse_matrix = matrix_to_tuple(sparse_matrix)

    return sparse_matrix


def preprocess_features(input_features):
    if sp.issparse(input_features):
        row_sum = np.array(input_features.sum(1))
        with np.errstate(divide='ignore'):
            r_inv = np.power(row_sum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        norm_features = r_mat_inv.dot(input_features)
        return np.asarray(norm_features.todense()), sparse_to_tuple_rep(norm_features)
    else:
        input_features = np.asarray(input_features, dtype=np.float32)
        row_sum = input_features.sum(1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        norm_features = input_features / row_sum
        return norm_features, None


def normalize_adj(adj_matrix):
    adj_matrix = sp.coo_matrix(adj_matrix)
    row_sum = np.array(adj_matrix.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()



def _resolve_data_dir(data_dir=None):
    candidate_dirs = []

    if data_dir is not None:
        candidate_dirs.append(data_dir)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs.append(os.path.join(current_dir, "Data"))
    candidate_dirs.append(os.path.join(os.getcwd(), "Data"))

    unique_dirs = []
    for d in candidate_dirs:
        if d not in unique_dirs:
            unique_dirs.append(d)

    for d in unique_dirs:
        if os.path.isdir(d):
            return d

    raise FileNotFoundError(f"Dataset directory not found. Tried: {unique_dirs}")


def _resolve_mat_path(base_dir, dataset_name):
    candidate_files = [
        f"{dataset_name}.mat",
        f"{dataset_name.lower()}.mat",
        f"{dataset_name.upper()}.mat",
        f"{dataset_name.capitalize()}.mat",
    ]

    for fname in candidate_files:
        path = os.path.join(base_dir, fname)
        if os.path.isfile(path):
            return path

    target_lower = f"{dataset_name.lower()}.mat"
    for real_name in os.listdir(base_dir):
        if real_name.lower() == target_lower:
            return os.path.join(base_dir, real_name)

    raise FileNotFoundError(f"Dataset file {dataset_name}.mat was not found under directory: {base_dir}")


def load_mat(dataset_name, data_dir=None, train_rate=0.3, val_rate=0.1):
    base_dir = _resolve_data_dir(data_dir)
    file_path = _resolve_mat_path(base_dir, dataset_name)

    print(f"[load_mat] Loading: {file_path}")
    data_dict = sio.loadmat(file_path)

    if 'Label' in data_dict:
        class_labels = data_dict['Label']
    elif 'gnd' in data_dict:
        class_labels = data_dict['gnd']
    elif 'label' in data_dict:
        class_labels = data_dict['label']
    else:
        raise KeyError(f"No label field was found in {file_path}. Supported fields: Label / gnd / label")

    if 'Attributes' in data_dict:
        features_attr = data_dict['Attributes']
    elif 'X' in data_dict:
        features_attr = data_dict['X']
    elif 'attr' in data_dict:
        features_attr = data_dict['attr']
    else:
        raise KeyError(f"No attribute field was found in {file_path}. Supported fields: Attributes / X / attr")

    if 'Network' in data_dict:
        network_data = data_dict['Network']
    elif 'A' in data_dict:
        network_data = data_dict['A']
    elif 'adj' in data_dict:
        network_data = data_dict['adj']
    else:
        raise KeyError(f"No adjacency field was found in {file_path}. Supported fields: Network / A / adj")

    adj_mat = sp.csr_matrix(network_data)

    if sp.issparse(features_attr):
        feat_mat = features_attr.tolil()
    else:
        feat_mat = sp.lil_matrix(np.asarray(features_attr, dtype=np.float32))

    anomaly_status = np.squeeze(np.array(class_labels))

    return adj_mat, feat_mat, anomaly_status


def adj_to_dgl_graph(adj_matrix):
    if not sp.issparse(adj_matrix):
        adj_matrix = sp.csr_matrix(adj_matrix)

    src, dst = adj_matrix.nonzero()
    src = th.from_numpy(src.astype(np.int64))
    dst = th.from_numpy(dst.astype(np.int64))

    dgl_g = dgl.graph((src, dst), num_nodes=adj_matrix.shape[0])
    return dgl_g


def generate_rwr_subgraph(dgl_graph, required_size):

    try:
        num_nodes = dgl_graph.number_of_nodes()
    except Exception:
        num_nodes = dgl_graph.num_nodes()

    all_nodes = list(range(num_nodes))
    reduced_size = required_size - 1

    subgraph_nodes_list = []

    for center in range(num_nodes):
        try:
            neighbors = dgl_graph.successors(center).cpu().numpy().tolist()
        except Exception:
            try:
                neighbors = dgl_graph.successors(th.tensor(center)).cpu().numpy().tolist()
            except Exception:
                neighbors = []

        neighbors = [int(n) for n in neighbors if int(n) != center]
        neighbors = list(dict.fromkeys(neighbors))

        if len(neighbors) >= reduced_size:
            chosen = random.sample(neighbors, reduced_size)
        else:
            chosen = neighbors[:]
            needed = reduced_size - len(chosen)

            candidate_pool = [n for n in all_nodes if n != center and n not in chosen]
            if len(candidate_pool) >= needed:
                chosen.extend(random.sample(candidate_pool, needed))
            else:
                chosen.extend(candidate_pool)
                while len(chosen) < reduced_size:
                    chosen.append(center)

        chosen = chosen[:reduced_size]
        chosen.append(center)
        subgraph_nodes_list.append(chosen)

    return subgraph_nodes_list

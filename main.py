"""基于 MovieLens-100K 数据的GraphAutoEncoder"""
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F

from dataset import MovielensDataset
from autoencoder import StackGCNEncoder, FullyConnected, Decoder


######hyper
DEVICE = torch.device('cpu')
LEARNING_RATE = 0.015
EPOCHS = 1000
NODE_INPUT_DIM = 2625
SIDE_FEATURE_DIM = 41
GCN_HIDDEN_DIM = 500
SIDE_HIDDEN_DIM = 10
ENCODE_HIDDEN_DIM = 75
NUM_BASIS = 4
DROPOUT_RATIO = 0.55
WEIGHT_DACAY = 0.
MASK_RATE = 0.3
REPLACE_RATE = 0.1
MASK_TOKEN_RATE = 1 - REPLACE_RATE
######hyper


SCORES = torch.tensor([[1, 2, 3, 4, 5]]).to(DEVICE)


def to_torch_sparse_tensor(x, device):
    if not sp.isspmatrix_coo(x):
        x = sp.coo_matrix(x)
    row, col = x.row, x.col
    data = x.data

    indices = torch.from_numpy(np.asarray([row, col]).astype('int64')).long()
    values = torch.from_numpy(x.data.astype(np.float32))
    th_sparse_tensor = torch.sparse_coo_tensor(indices, values,
                                                x.shape).to(device)

    return th_sparse_tensor


def tensor_from_numpy(x, device):

    return torch.from_numpy(x).to(device)


class GraphMatrixCompletion(nn.Module):
    def __init__(self, input_dim, side_feat_dim,
                 gcn_hidden_dim, side_hidden_dim,
                 encode_hidden_dim,
                 num_support=5, num_classes=5, num_basis=3):
        super(GraphMatrixCompletion, self).__init__()
        self.encoder = StackGCNEncoder(input_dim, gcn_hidden_dim, num_support, DROPOUT_RATIO)
        self.dense1 = FullyConnected(side_feat_dim, side_hidden_dim, dropout=0.,
                                     use_bias=True)
        self.dense2 = FullyConnected(gcn_hidden_dim + side_hidden_dim, encode_hidden_dim,
                                     dropout=DROPOUT_RATIO, activation=lambda x: x)
        self.decoder = Decoder(encode_hidden_dim, num_basis, num_classes,
                               dropout=DROPOUT_RATIO, activation=lambda x: x)
    
    def encoding_mask_noise(self, user2movie_adjacencies, movie2user_adjacencies, 
                            user_indices, movie_indices, labels):
        num_ratings = len(labels)
        perm = torch.randperm(num_ratings, device=DEVICE)
        #num_mask_nodes = int(mask_rate * num_ratings)

        # random masking
        num_mask_nodes = int(MASK_RATE * num_ratings)
        mask_nodes = perm[: num_mask_nodes]
        # keep_nodes = perm[num_mask_nodes: ]
        mask_user_nodes = list(set(user_indices[mask_nodes]))
        mask_movie_nodes = list(set(movie_indices[mask_nodes]))

        if REPLACE_RATE > 0:
            num_noise_nodes = int(REPLACE_RATE * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=DEVICE)
            token_nodes = mask_nodes[perm_mask[: int(MASK_TOKEN_RATE * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(REPLACE_RATE * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_ratings, device=DEVICE)[:num_noise_nodes]

            out_labels = labels.clone()
            out_user_indices = user_indices.clone()
            out_movie_indices = movie_indices.clone()
            out_user2movie_adjacencies = user2movie_adjacencies.copy()
            out_movie2user_adjacencies = movie2user_adjacencies.copy()
            
            out_labels[token_nodes] = -1
            out_labels[noise_nodes] = labels[noise_to_be_chosen]
            out_user_indices[token_nodes] = -1
            out_user_indices[noise_nodes] = user_indices[noise_to_be_chosen]
            out_movie_indices[token_nodes] = -1
            out_movie_indices[noise_nodes] = movie_indices[noise_to_be_chosen]
            
            for user_index, movie_index, rating_index in zip(user_indices[token_nodes], movie_indices[token_nodes], labels[token_nodes]):
                out_user2movie_adjacencies[rating_index][user_index][movie_index] = 0.0 + nn.Parameter(torch.zeros(1))
                out_movie2user_adjacencies[rating_index][movie_index][user_index] = 0.0 + nn.Parameter(torch.zeros(1))
            for user_index, movie_index, rating_index, user_noise, movie_noise, rating_noise in zip(user_indices[noise_nodes], movie_indices[noise_nodes], labels[noise_nodes], user_indices[noise_to_be_chosen], movie_indices[noise_to_be_chosen], labels[noise_to_be_chosen]):
                out_user2movie_adjacencies[rating_index][user_index][movie_index] = user2movie_adjacencies[rating_noise][user_noise][movie_noise]
                out_movie2user_adjacencies[rating_index][movie_index][user_index] = movie2user_adjacencies[rating_noise][movie_noise][user_noise]
        else:
            token_nodes = mask_nodes
            
            out_labels = labels.clone()
            out_user_indices = user_indices.clone()
            out_movie_indices = movie_indices.clone()
            out_user2movie_adjacencies = user2movie_adjacencies.copy()
            out_movie2user_adjacencies = movie2user_adjacencies.copy()
            
            out_labels[mask_nodes] = -1
            out_user_indices[mask_nodes] = -1
            out_movie_indices[mask_nodes] = -1
            
            out_user2movie_adjacencies[labels[mask_nodes]][user_indices[mask_nodes]][movie_indices[mask_nodes]] = 0.0 + nn.Parameter(torch.zeros(1))
            out_user2movie_adjacencies[labels[mask_nodes]][movie_indices[mask_nodes]][user_indices[mask_nodes]] = 0.0 + nn.Parameter(torch.zeros(1))

        return out_labels, out_user_indices, out_movie_indices, out_user2movie_adjacencies, out_movie2user_adjacencies, (mask_user_nodes, mask_movie_nodes)

    def forward(self, user_supports, item_supports,
                user_inputs, item_inputs,
                user_side_inputs, item_side_inputs,
                user_edge_idx, item_edge_idx, labels):
        dense_user_supports = []
        dense_item_supports = []
        for i in range(len(user_supports)):
            tmp = user_supports[i].to_dense()
            dense_user_supports.append(tmp)
        for i in range(len(item_supports)):
            tmp = item_supports[i].to_dense()
            dense_item_supports.append(tmp)
        
        _, pre_user_edge_idx, pre_item_edge_idx, pre_user_supports, pre_item_supports, (mask_user_nodes, mask_movie_nodes) = self.encoding_mask_noise(dense_user_supports, dense_item_supports, user_edge_idx, item_edge_idx, labels)
        
        for i in range(len(pre_user_supports)):
            pre_user_supports[i] = pre_user_supports[i].to_sparse()
        for i in range(len(pre_item_supports)):
            pre_item_supports[i] = pre_item_supports[i].to_sparse()
        
        user_gcn, movie_gcn = self.encoder(pre_user_supports, pre_item_supports, user_inputs, item_inputs)
        user_side_feat, movie_side_feat = self.dense1(user_side_inputs, item_side_inputs)

        user_feat = torch.cat((user_gcn, user_side_feat), dim=1)
        movie_feat = torch.cat((movie_gcn, movie_side_feat), dim=1)

        user_embed, movie_embed = self.dense2(user_feat, movie_feat)
        # print(user_gcn.shape)
        # print(movie_gcn.shape)
        # remask
        # for i, j in zip(mask_user_nodes, mask_movie_nodes):
        #     user_embed[mask_user_nodes] = 0.0
        #     movie_embed[mask_movie_nodes] = 0.0

        edge_logits = self.decoder(user_embed, movie_embed, pre_user_edge_idx, pre_item_edge_idx)
        print(edge_logits.shape)

        return edge_logits

data = MovielensDataset()
user2movie_adjacencies, movie2user_adjacencies, \
    user_side_feature, movie_side_feature, \
    user_identity_feature, movie_identity_feature, \
    user_indices, movie_indices, labels, train_mask = data.build_graph(
        *data.read_data())

user2movie_adjacencies = [to_torch_sparse_tensor(adj, DEVICE) for adj in user2movie_adjacencies]
movie2user_adjacencies = [to_torch_sparse_tensor(adj, DEVICE) for adj in movie2user_adjacencies]
user_side_feature = tensor_from_numpy(user_side_feature, DEVICE).float()
movie_side_feature = tensor_from_numpy(movie_side_feature, DEVICE).float()
user_identity_feature = tensor_from_numpy(user_identity_feature, DEVICE).float()
movie_identity_feature = tensor_from_numpy(movie_identity_feature, DEVICE).float()
user_indices = tensor_from_numpy(user_indices, DEVICE).long()
movie_indices = tensor_from_numpy(movie_indices, DEVICE).long()
labels = tensor_from_numpy(labels, DEVICE)
train_mask = tensor_from_numpy(train_mask, DEVICE)


model = GraphMatrixCompletion(NODE_INPUT_DIM, SIDE_FEATURE_DIM, GCN_HIDDEN_DIM,
                              SIDE_HIDDEN_DIM, ENCODE_HIDDEN_DIM, num_basis=NUM_BASIS).to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DACAY)
model_inputs = (user2movie_adjacencies, movie2user_adjacencies,
                user_identity_feature, movie_identity_feature,
                user_side_feature, movie_side_feature, user_indices, movie_indices, labels)

def train():
    test_result = []
    model.train()
    for e in range(EPOCHS):
        logits = model(*model_inputs)
        index = torch.argmax(logits, dim=1)
        percentage = sum(1 for i, l in zip(index[train_mask], labels[train_mask]) if i == l) / len(index[train_mask])
        loss = criterion(logits[train_mask], labels[train_mask])
        rmse = expected_rmse(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新

        tr = test()
        test_result.append(tr)
        model.train()
        print(f"Epoch {e:04d}: TrainLoss: {loss.item():.4f}, TrainRMSE: {rmse.item():.4f}, TrainAccuracy: {percentage:.4f}, "
              f"TestRMSE: {tr[0]:.4f}, TestLoss: {tr[1]:.4f}, TestAccuracy: {tr[2]:.4f}")

    test_result = np.asarray(test_result)
    idx = test_result[:, 0].argmin()
    print(f'test min rmse {test_result[idx]} on epoch {idx}')
    tr = test()
    return tr[3]


@torch.no_grad()
def test():
    model.eval()
    logits = model(*model_inputs)
    test_mask = ~train_mask
    index = torch.argmax(logits, dim=1)
    percentage = sum(1 for i, l in zip(index[test_mask], labels[test_mask]) if i == l) / len(index[test_mask])
    loss = criterion(logits[test_mask], labels[test_mask])
    rmse = expected_rmse(logits[test_mask], labels[test_mask])
    return rmse.item(), loss.item(), percentage, logits[test_mask]

@torch.no_grad()
@torch.no_grad()
def eval():
    model.load_state_dict(torch.load('model.pth', weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    logits = model(*model_inputs)
    test_mask = ~train_mask
    results = torch.argmax(logits[test_mask], dim=1) + 1
    return results.tolist()

def expected_rmse(logits, label):
    true_y = label + 1  # 原来的评分为1~5，作为label时为0~4
    prob = F.softmax(logits, dim=1)
    pred_y = torch.sum(prob * SCORES, dim=1)
    
    diff = torch.pow(true_y - pred_y, 2)
    
    return torch.sqrt(diff.mean())


if __name__ == "__main__":
    train()
    # print(user2movie_adjacencies[0])

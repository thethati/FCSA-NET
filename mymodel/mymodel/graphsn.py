import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch
from torch.nn import Sequential, Linear, ReLU, Parameter, Sequential, BatchNorm1d, Dropout
import torch.nn.functional as F
import math
import torch_geometric
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class GraphSN(nn.Module):
    def __init__(self, input_dim, hidden_dim, batchnorm_dim, dropout):
        super().__init__()
        
        self.mlp = Sequential(Linear(input_dim, hidden_dim), Dropout(dropout), 
                              ReLU(),
                              Linear(hidden_dim, hidden_dim), Dropout(dropout), 
                              ReLU())

        # self.mlp1 = Sequential(Linear(input_dim, hidden_dim), Dropout(dropout), ReLU())

        # self.mlp2 = Sequential(Linear(input_dim, hidden_dim), Dropout(dropout), ReLU())
                           
        
        self.linear = Linear(hidden_dim, hidden_dim)
        
        self.eps = Parameter(torch.FloatTensor(1))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv_eps = 0.1 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)

    def forward(self, A, X):
        """
        Params
        ------
        A [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix
        
        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        
        batch, N = A.shape[:2]
        mask = torch.eye(N).unsqueeze(0).to('cuda')
        # # print("&"*36)
        # # print(batch) # 1
        # print(mask.shape) # 1 10 10
        # print(A.shape)   # 1 10  10
        batch_diagonal = torch.diagonal(A, 0, 1, 2)
        #print(batch_diagonal.shape) # 1 10 
        batch_diagonal = self.eps * batch_diagonal
        # print(batch_diagonal)
        # print(torch.diag_embed(batch_diagonal).shape)  # 1 10 10
        # print(1. - mask) # 非对角矩阵1
        A = mask*torch.diag_embed(batch_diagonal) + (1. - mask)*A
        # print(A)
        X = self.mlp(A @ X)
        X = self.linear(X)
        X = F.relu(X)
      
        #print(X.shape) # ([1, 22, 64])
        return X


class GNN(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, batchnorm_dim, dropout_1, dropout_2):
        super().__init__()
        
        self.dropout = dropout_1
        
        self.convs = nn.ModuleList()
        
        self.convs.append(GraphSN(input_dim, hidden_dim, batchnorm_dim, dropout_2))
        
        for _ in range(n_layers-1):
            self.convs.append(GraphSN(hidden_dim, hidden_dim, batchnorm_dim, dropout_2))
        
        #self.out_proj = nn.Linear(input_dim+hidden_dim*(n_layers), output_dim)
        self.out_proj = nn.Linear((input_dim+hidden_dim*(n_layers)), output_dim)

    def forward(self, data):
        # X, A = data[:2]

        A = data.edge_index

        X =  data.x

        edge_weight = data.edge_weight

        # print(type(A)) # <class 'torch.Tensor'>
        # A = torch_geometric.utils.to_scipy_sparse_matrix(A)
        # print(type(A)) # <class 'scipy.sparse._coo.coo_matrix'>
        # A = A.toarray() 
        # print(type(A)) # <class 'numpy.ndarray'>  
      
        # print(edge_weight)

        # A, _ = add_self_loops(A)
       
        # A = torch_geometric.utils.to_dense_adj(edge_index=A, batch=None, edge_attr=edge_weight).squeeze(0).to('cuda')
        A = torch_geometric.utils.to_dense_adj(edge_index=A, batch=None, edge_attr=edge_weight).to('cuda')
       
        mask = torch.eye(A.shape[-1]).unsqueeze(0).to('cuda')
       
        A = A + mask
     
        # print(A)
        #A = torch.squeeze(A)

        # print(A)
        
        # print(A.shape)
        # exit()
        hidden_states = []
        
        X = torch.unsqueeze(X, 0)

        hidden_states.append(X)

        # print(X.shape)  # torch.Size([8, 64])

        # print(len(hidden_states))  # 1
        for layer in self.convs:
            X = F.dropout(layer(A, X), self.dropout)
            hidden_states.append(X)
        
        # print(len(hidden_states)) # 3
        # print("*"*36)
        # for index in hidden_states:
        #     print(index.shape)
        # print("*"*36)
        #X = torch.cat(hidden_states, dim=-1).sum(dim=1) # 1 24 192
        X = torch.cat(hidden_states, dim=-1)  # 1 24 192
        #print(X.shape)

        
        X = self.out_proj(X)
    

        X = torch.squeeze(X)

        return X




        #  model = GNN(input_dim=loaders[0].dataset.features_dim,
        #         hidden_dim=args.hidden_dim, 64
        #         output_dim=loaders[0].dataset.n_classes,
        #         n_layers=args.n_layers, 2
        #         batchnorm_dim=args.batchnorm_dim, 28
        #         dropout_1=args.dropout_1, 0.5
        #         dropout_2=args.dropout_2  0.6
        #         ).to(args.device)

            

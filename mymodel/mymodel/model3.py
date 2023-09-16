import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import GATv2Conv,  global_add_pool, GCNConv
from utils.tools import model_size
from .resnet_18 import resnet18
from .transformer_module import CTransformerEncoder
from .graphsn import GNN

class fmodel(nn.Module):
    def __init__(self, num_node_features: int, num_branch: int = 5):

        super(fmodel, self).__init__()
        # iamge embedding dim = 64
        self.num_node_features = num_node_features

        self.num_negative = 1
        # the depth of gnn
        self.gnn_depth = 3

        self.dropout = 0.3

        self.embedding = resnet18(pretrained=True, embedding_size=self.num_node_features)

        self.ln = nn.LayerNorm(512)

        self.lin1 = nn.Linear(256,1)
        #self.lin2 = nn.Linear(64,1)
        
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model = 128,
            nhead = 8
        )
        
        # self.gcn1 = GCNConv(in_channels = 64, out_channels= 64, add_self_loops =False)
        # self.gcn2 = GCNConv(in_channels = 64, out_channels=64, add_self_loops =False)
        # self.gcn3 = GCNConv(in_channels = 64, out_channels=64)
        # self.gcn4 = GCNConv(in_channels = 64, out_channels=64)
        # self.gcn5 = GCNConv(in_channels = 64, out_channels=64)

        # self.gnn1 = GNN(input_dim=64,hidden_dim=64, output_dim=64,n_layers=2,batchnorm_dim=10,dropout_1=0.5,dropout_2=0.6)

        # best aim
        self.gnn1 = GNN(input_dim=64,hidden_dim=64, output_dim=64,n_layers=2,batchnorm_dim=10,dropout_1=0,dropout_2=0)

        # self.gnn1 = GNN(nfeat=64,nhid=64, nclass=64,dropout=0)
        # self.gnn1 = GNN(in_channels = 64, out_channels= 64)
        # self.lrelu1 = torch.nn.LeakyReLU(0.01)
        # self.dp1 = torch.nn.Dropout(p=self.dropout)
        # self.gnn2 = GNN(in_channels = 64, out_channels= 64)
        # self.lrelu2 = torch.nn.LeakyReLU(0.01)
        # # self.dp2 = torch.nn.Dropout(p=self.dropout)
        # self.gnn3 = GNN(in_channels = 64, out_channels= 64)
        # # self.lrelu3 = torc081d372f89eah.nn.LeakyReLU(0.01)

      

        self.trans1 = torch.nn.TransformerEncoder(
            encoder_layer = self.encoder_layer,
            num_layers = 4,
            norm = None
        )

        self.encoder_layer2 = torch.nn.TransformerEncoderLayer(
            d_model = 128,
            nhead = 8
        )

        self.reset_parameters()

    # def prepareclip(self):

    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     #  model, preprocess = clip.load("ViT-B/32", device=device)
    #     model, process  = clip.load('ViT-B/32', device)
    #     return model,process


    @staticmethod
    def _init_sequential(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, 0.001)
            nn.init.constant_(m.bias, 0)

    def reset_parameters(self):
        self.lin1.apply(self._init_sequential)
        #self.lin2.apply(self._init_sequential)

    @staticmethod
    def _calc_src_key_padding_mask(graphs, is_bool=True):

        max_len = max([s.size(0) for s in graphs])
        padding_mask = torch.ones(len(graphs), max_len)
        for i, graph in enumerate(graphs):        
            index = torch.tensor([max_len - ti for ti in range(1, max_len - graph.size(0) + 1)])     
            if len(index):
                padding_mask[i].index_fill_(0, index, 0)
        if is_bool:
            return (1 - padding_mask).bool()
        else:
            return padding_mask

    def forward(self, data):

        # print(data.x.shape) # N 3 H W torch.Size([720, 3, 112, 112])
        # print(data.num_graphs) # batch_size * 2
       
        data.x = self.embedding(data.x) # N 64

        # data.x = self.gcn1(x=data.x, edge_index=data.edge_index, edge_weight =data.edge_weight)# N 64
        # data.x = self.gcn2(x=data.x, edge_index=data.edge_index, edge_weight =data.edge_weight)# N 64

        before_data = data.x.clone()

        data.x = self.gnn1(data)  # N 64

        data.x = torch.cat([before_data, data.x ], dim = -1)

        #beforedata = data.x.clone()

        #print(data.x.shape) # 680 64
        #data.x = self.gnn1(data)

        #print(data.edge_index.shape)

        #print(data.edge_weight.shape) # torch.Size([4072, 1])
     
        #print(data.edge_weight.shape)
        #data.x = data.x + beforedata

        #beforedata = data.x.clone()

        #data.x = self.dp1(data.x)
        #data.x = self.gnn2(data)

    
        #data.x = data.x + beforedata

        # data.x = self.gnn3(data)
        # data.x = self.gcn3(data.x, data.edge_index)

        # data.x = data.x + beforedata

        # data.x = self.gcn4(data.x, data.edge_index)

        # data.x = data.x + beforedata
        
        # data.x = self.gcn5(data.x,data.edge_index)

        # data.x = data.x + beforedata

        graphs = []
        
        for shift in range(data.num_graphs):

            tmp_slices = data.slices_indicator[shift:shift + 2]

            graphs.append(data.x[tmp_slices[0].item():tmp_slices[1].item()])
                                                                                                                                                                                                                                                                                                            
        graph_attention = []
        
        for graph in graphs:
            # GR-BLOCK
            graph_attention.append(global_add_pool(graph, torch.tensor(0).to("cuda")))
        #print(len(graphs))
        batch_data_attention  = torch.cat(graph_attention,dim = 0) # N E 128 64       

        #print(batch_data_attention.shape)

        batch_data_attention = self.encoder_layer2(batch_data_attention)
        
        batch_data = pad_sequence(graphs)   # S N E   11  128  64

        # print(batch_data.shape)

        padding_mask = self._calc_src_key_padding_mask(graphs).to("cuda")

        batch_data = self.trans1(batch_data, src_key_padding_mask = padding_mask)

        batch_data.transpose_(0, 1) # N S E    128, 8, 64

        # batch_data = torch.einsum('bij,bi->bij', [batch_data, final_att])  # (N,S,E)

        batch_data = batch_data.sum(1)  # (N,E) 128 64

        batch_data = torch.cat([batch_data_attention,batch_data], dim = -1) # 128 128

        batch_data = self.lin1(batch_data) # 线性层
        #batch_data = self.lin2(batch_data)

        fine_score = batch_data.sum(dim=1)

        return fine_score

    def bpr_loss(self, output):

        output = output.view(-1, (self.num_negative + 1))  # each row: (pos, neg, neg, neg, ..., neg) 128 

        # the first score (pos scores) minus each remainder scores (neg scores)
        output = output[:, 0].unsqueeze(-1).expand_as(output[:, 1:]) - output[:, 1:]

        # 输出大于0的个数
        batch_acc = (output > 0).sum().item() * 1.0 / output.nelement()

        return -F.logsigmoid(output).mean(), batch_acc


    @torch.no_grad()
    def test_auc(self, batch):
        self.eval()
        output = self(batch)
        return output.view(-1)


    @torch.no_grad()
    def test_fitb(self, batch):
        self.eval()
        output = self(batch) 
        output = output.view(-1, 4)  # each row: (pos, neg, neg, neg)
        _, max_idx = output.max(dim=-1)
        return (max_idx == 0).sum().item()

    
    @torch.no_grad()
    def test_retrieval(self, batch, ranking_neg_num):
        self.eval()
        output = self(batch)
        return output.view(-1, ranking_neg_num + 1)




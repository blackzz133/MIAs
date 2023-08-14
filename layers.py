import torch
import torch.nn as nn
import torch.nn.functional as F
import copy as cp
import math
from torch_geometric.utils import softmax
#from torch.nn.functional import softmax
from torch_scatter import scatter
import torch_geometric



class StructuralAttentionLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_heads,
                 attn_drop,
                 ffd_drop,
                 residual):
        super(StructuralAttentionLayer, self).__init__()
        self.input_dim = input_dim

        self.out_dim = output_dim // n_heads
        self.n_heads = n_heads
        self.act = nn.ELU()
        self.beta = 0.5

        self.lin = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        self.att_r = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        self.sim_weight_linear = torch.nn.Parameter(
            torch.Tensor(n_heads, self.input_dim, self.out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.attn_drop = nn.Dropout(attn_drop)
        self.ffd_drop = nn.Dropout(ffd_drop)

        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)

        #self.coefficient = None

        self.xavier_init()

    def forward(self, graph):
        graph = torch_geometric.data.data.Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=graph.y)
        #graph = copy.deepcopy(graph)
        edge_index = graph.edge_index
        #x_origin = cp.deepcopy(x)
        edge_weight = graph.edge_attr.reshape(-1, 1)
        H, C = self.n_heads, self.out_dim
        x = self.lin(graph.x).view(-1, H, C)  # [N, heads, out_dim]
        # attention
        alpha_l = (x * self.att_l).sum(dim=-1).squeeze()  # [N, heads]
        alpha_r = (x * self.att_r).sum(dim=-1).squeeze()
        alpha_l = alpha_l[edge_index[0]]  # [num_edges, heads]
        alpha_r = alpha_r[edge_index[1]]
        alpha = alpha_r + alpha_l
        alpha = edge_weight * alpha
        alpha = self.leaky_relu(alpha)
        coefficients = softmax(alpha, edge_index[1])  # [num_edges, heads]
        # dropout
        if self.training:
            coefficients = self.selective_sampling(coefficients, graph)  # 4912,8
            coefficients = self.attn_drop(coefficients)
            x = self.ffd_drop(x)

        self.coefficient = coefficients #############################这个在不同时间会被替换
        x_j = x[edge_index[0]]  # [num_edges, heads, out_dim]
        # output
        out = self.act(scatter(x_j * coefficients[:, :, None], edge_index[1], dim=0, reduce="sum"))
        out = out.reshape(-1, self.n_heads * self.out_dim)  # [num_nodes, output_dim]
        if self.residual:
            edge_set = list(set(edge_index[1].tolist()))
            out = out + self.lin_residual(graph.x)
        graph.x = out
        return graph, coefficients

    def xavier_init(self):
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        nn.init.xavier_uniform_(self.sim_weight_linear)

    def selective_sampling(self, coefficients, graph): #选择性采样  sim
        sqrt_pi = math.sqrt(math.pi)
        coefficients = coefficients * sqrt_pi ** (self.beta * self.similarity(graph))
        coefficients = softmax(coefficients, graph.edge_index[1])
        return coefficients

    def similarity(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        #x_l = torch.matmul(self.model_free(x[edge_index[0]], edge_index, edge_attr).detach(), self.sim_weight_linear)
        #x_r = torch.matmul(self.model_free(x[edge_index[1]], edge_index, edge_attr).detach(), self.sim_weight_linear)
        x_l = torch.matmul(x[edge_index[0]], self.sim_weight_linear)
        x_r = torch.matmul(x[edge_index[1]], self.sim_weight_linear)
        cos = nn.CosineSimilarity(dim=2)
        return cos(x_l, x_r).transpose(0,1)

class TemporalAttentionLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 n_heads,
                 num_time_steps, #已经修改为time_span
                 attn_drop,
                 residual):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual
        # define weights
        self.temp_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.temp1_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.temp2_weights = nn.Parameter(torch.Tensor(self.num_time_steps, input_dim, input_dim))
        self.drop = nn.Dropout(0.8)
        #self.position_embeddings = nn.Parameter(torch.Tensor(self.num_time_steps, input_dim))
        #self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        #self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        #self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.type = type

        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        #N节点，T时间，F特征
        # 1: Add position embeddings to input
        # [N,T]
        #assert inputs.shape[1] >= self.num_time_steps
        #H_temp1 = []
        #H_temp2 = []
        time = inputs.shape[1] #(N.T,D)
        #inputs = inputs.transpose(0,1) #(N.T,D)
        H_temp1 = torch.matmul(inputs, self.temp1_weights)
        H_temp2 = inputs - inputs
        #H_temp2 = H_temp2 - H_temp2

        for t in range(0,time):
            for t2 in range(max(0,t-self.num_time_steps+1),t+1):
                H_temp2[:, t, :] += torch.matmul(inputs[:,t2,:],self.temp2_weights[t2])
            H_temp2[:, t, :] /= min(t+1, self.num_time_steps)
        alph = torch.bmm(H_temp1, H_temp2.transpose(1,2))
        H_temp = torch.matmul(inputs,self.temp_weights)
        outputs = torch.matmul(alph, H_temp)
        if self.training:
            outputs = self.drop(outputs)
        return outputs+inputs  # if node
    def xavier_init(self):
        #nn.init.xavier_uniform_(self.position_embeddings)
        #nn.init.xavier_uniform_(self.Q_embedding_weights)
        #nn.init.xavier_uniform_(self.K_embedding_weights)
        #nn.init.xavier_uniform_(self.V_embedding_weights)
        nn.init.xavier_uniform_(self.temp1_weights)
        nn.init.xavier_uniform_(self.temp2_weights)
        nn.init.xavier_uniform_(self.temp_weights)

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs+inputs

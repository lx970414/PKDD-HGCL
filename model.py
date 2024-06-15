"""NN modules"""
import torch as th
import torch.nn as nn
from torch.nn import init
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import time
from utils import get_activation, to_etype_name
from info_nce import InfoNCE

th.set_printoptions(profile="full")
class GCMCGraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=True,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        
        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        #self.att = nn.Parameter(th.Tensor(1 , basis_units))
        self.prob_score = nn.Linear(64, 1, bias=False)
        #self.relation_weight = nn.Linear(244, 64)
        # self.pro = nn.Sequential(
        #     nn.Linear(64, 16, bias=False),  
        #     nn.LeakyReLU(),
        #     nn.Linear(16, 1, bias=False)
        # )
        # self.review_w = nn.Linear(64, 90)
        # self.review_score = nn.Linear(64, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.prob_score.weight)
        # init.xavier_uniform_(self.review_w.weight)
        # init.xavier_uniform_(self.review_score.weight)
        #init.xavier_uniform_(self.att)

    def forward(self, graph, feat, weight=None, Two_Stage = False):

        with graph.local_scope():
            if isinstance(feat, tuple):
                feat, ifeat = feat      # dst feature not used
            cj = graph.srcdata['cj']
            ci = graph.dstdata['ci']
            if self.device is not None:
                cj = cj.to(self.device)
                ci = ci.to(self.device)
            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight
            #weight = th.matmul(self.att, weight.view(weight.shape[0], -1)).view(weight.shape[1], -1)
            if weight is not None:
                feat = dot_or_identity(feat, weight, self.device)
                ifeat = dot_or_identity(ifeat, weight, self.device)

            feat = feat * self.dropout(cj)
            graph.srcdata['h'] = feat
            graph.dstdata['h'] = ifeat
            #print(feat.shape)
            #print(ifeat.shape)
            review_feat = graph.edata['review_feat']
            #print(review_feat.shape)
            #print(pa.shape)
            #graph.edata['rf'] = self.review_w(review_feat) * th.sigmoid(self.review_score(review_feat))
            #rf = self.review_w(review_feat) * th.sigmoid(self.review_score(review_feat))
            #print(rf.shape)
            #print(review_feat.shape)

            # rating_feat = th.mean(review_feat,dim=0)
            # print(rating_feat.shape)
            # graph.apply_edges(udf_v_add_r)
            # m = graph.edata['m'] 
            # score = th.cat([m, review_feat], 1)
            # num = score.shape[0]
            #relation = self.relation_weight(score)  # src + dst + review 
            #print(relation.shape)
            # rating = th.mean(relation,dim=0)
            # rating_emb = rating.repeat(num,1)  # 2221,
            # print("m")
            # print(score.shape)

            graph.edata['pa'] = th.sigmoid(self.prob_score(review_feat))
            graph.update_all(lambda edges: {'m': (edges.data['pa'] * edges.src['h'])
                                                 },
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            rst = rst * ci 
            ######################
            # feat = feat * self.dropout(cj)
            # graph.srcdata['h'] = feat
            # graph.update_all(fn.copy_u('h', 'm'),
            #                  fn.sum(msg='m', out='h'))
            # rst = graph.dstdata['h']
            # rst = rst * ci 
            # #######################
        return rst



class HGNNLayer(nn.Module):
    def __init__(self):
        super(HGNNLayer, self).__init__()
        self.act = nn.ReLU()
        self.softmax0 = nn.Softmax(dim=0)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)

    def forward(self, adj, embeds):  # embed:5, 5541, 90  adj:5, 5541, 16
        # print("hgnn")
        # print(adj.shape, embeds.shape)
        HT = adj.permute(0, 2, 1)
        lat = self.softmax0(HT @ embeds)  #5 ,16, 90
        ret = self.softmax1(adj @ lat)
        #print(ret.shape)  # 5, 5541, 90
        return ret  


class HGCNLayer(nn.Module):
    def __init__(self):
        super(HGCNLayer, self).__init__()
        self.act = nn.ReLU()
        self.softmax0 = nn.Softmax(dim=0)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)

    def forward(self, adj, embeds):  # adj:5, 5541, 5541  embeds:5, 5541, 90
        #print("hgnn")
        # print(adj.shape, embeds.shape)
        #print(adj[0][0], adj[0][0].shape)
        lat = self.softmax1(adj @ embeds)
        # print(lat.shape)  # 5, 5541, 90
        return lat  


class GCMCLayer(nn.Module):

    def __init__(self,
                 hyperedge_number,
                 rating_vals,
                 user_in_units,
                 movie_in_units,
                 msg_units,
                 out_units,
                 dropout_rate=0.0,
                 agg='stack',  
                 dataset=None,
                 agg_act=None,
                 out_act=None,
                 share_user_item_param=False,
                 ini = True,
                 basis_units=5,
                 device=None,
                 user_num=0,
                 num=0):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.agg = agg
        self.hyper = hyperedge_number
        self.heads = 8
        self.share_user_item_param = share_user_item_param
        self.ufc = nn.Linear(msg_units, out_units)
        self.ifc = self.ufc
        self.ufc2 = nn.Linear(msg_units, out_units)                   
        self.ifc2 = self.ufc2
        self.user_in_units = user_in_units
        self.msg_units = msg_units
        self.num = user_num
        self.msg = msg_units
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        if ini:
            msg_units = msg_units  // 3
        self.ini = ini
        self.msg_units = msg_units
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(0.3)
        #self.W_r = nn.ParameterDict()
        self.W_r = {}
        subConv = {}
        subConv2 = {}
        self.basis_units = basis_units
        self.att = nn.init.xavier_uniform_(nn.Parameter(th.randn(len(self.rating_vals), basis_units)))
        self.basis = nn.init.xavier_uniform_((nn.Parameter(th.randn(basis_units, user_in_units, msg_units))))
        self.crossrate = Attentioncross(self.msg, device, len(self.rating_vals))
        for i, rating in enumerate(rating_vals):
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            if share_user_item_param and user_in_units == movie_in_units:

                subConv[rating] = GCMCGraphConv(user_in_units,
                                                msg_units,
                                                weight=False,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(user_in_units,
                                                    msg_units,
                                                    weight=False,
                                                    device=device,
                                                    dropout_rate=dropout_rate)


        self.conv = dglnn.HeteroGraphConv(subConv, aggregate=agg)
        self.hgnnLayer = HGNNLayer()
        #self.hgcnLayer = HGCNLayer()
        self.agg_act = get_activation(agg_act)
        self.out_act = get_activation(out_act)
        self.uHyper = nn.Parameter((th.randn(len(self.rating_vals), self.msg, self.hyper)))# 10, 30, 30
        self.iHyper = nn.Parameter((th.randn(len(self.rating_vals), self.msg, self.hyper)))
        self.hypergraph_u = dataset.hypergraph_u
        self.hypergraph_v = dataset.hypergraph_v
        # self.node_review_u = dataset.node_review_u
        # self.node_review_v = dataset.node_review_v
        #self.review_w = nn.Linear(64, 90)
        self.device = device
        self.reset_parameters()

    def partial_to(self, device):

        assert device == self.device
        if device is not None:
            self.ufc.cuda(device)
            if self.share_user_item_param is False:
                self.ifc.cuda(device)
            self.dropout.cuda(device)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.basis)


    def forward(self, graph, ufeat=None, ifeat=None, uhfeati=None, ihfeati=None, Two_Stage = False):
        CL = InfoNCE()
        in_feats = {'user' : ufeat, 'movie' : ifeat}
        mod_args = {}
        self.W = th.matmul(self.att, self.basis.view(self.basis_units, -1))
        self.W = self.W.view(-1, self.user_in_units, self.msg_units)
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            mod_args[rating] = (self.W[i,:,:] if self.W_r is not None else None, Two_Stage)
            mod_args[rev_rating] = (self.W[i,:,:] if self.W_r is not None else None, Two_Stage)
        out_feats = self.conv(graph, in_feats, mod_args=mod_args)
        ufeat = out_feats['user']
        ifeat = out_feats['movie']
        if(ufeat.shape[1] > 5):
            n = int(ufeat.shape[1] / 5 + 1)
            rcloss = 0
            rcloss = CL(ufeat[:, 3, :], ufeat[:, 8, :]) + 1 * CL(ifeat[:, 3, :], ifeat[:, 8, :])
            rcloss = rcloss + CL(ufeat[:, 12, :], ufeat[:, 18, :]) + 1 * CL(ifeat[:, 12, :], ifeat[:, 18, :])
            rcloss = rcloss + CL(ufeat[:, 24, :], ufeat[:, 30, :]) + 1 * CL(ifeat[:, 24, :], ifeat[:, 30, :])
            rcloss = rcloss + CL(ufeat[:, 39, :], ufeat[:, 48, :]) + 1 * CL(ifeat[:, 39, :], ifeat[:, 48, :])
            rcloss = rcloss + CL(ufeat[:, 58, :], ufeat[:, 67, :]) + 1 * CL(ifeat[:, 58, :], ifeat[:, 67, :])
        else:
            
            rcloss = 0
            rcloss = CL(ufeat[:, 0, :], ufeat[:, 1, :]) + 1 * CL(ifeat[:, 0, :], ifeat[:, 1, :])
            rcloss = rcloss + CL(ufeat[:, 2, :], ufeat[:, 3, :]) + 1 * CL(ifeat[:, 2, :], ifeat[:, 3, :])
            rcloss = rcloss + CL(ufeat[:, 3, :], ufeat[:, 4, :]) + 1 * CL(ifeat[:, 3, :], ifeat[:, 4, :])

        hyperedge_u = ufeat.permute(1, 0, 2) # 2300,10,30 -> 5, 5541, 90
        hyperedge_v = ifeat.permute(1, 0, 2)
        hu = hyperedge_u 
        hv = hyperedge_v
        # node_u = self.review_w(self.node_review_u)
        # node_v = self.review_w(self.node_review_v)
        ufeat = self.crossrate(ufeat)
        ifeat = self.crossrate(ifeat)
        
        ufeat = th.sum(ufeat, dim=1)
        ifeat = th.sum(ifeat, dim=1)

        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        uuHyper = self.relu(hyperedge_u @ self.uHyper) # 10, 2300, 32
        iiHyper = self.relu(hyperedge_v @ self.iHyper)

        # print("uuhyper")
        # print(uuHyper[0], uuHyper[0].shape)

        hyperULat = self.hgnnLayer(uuHyper, hu)
        hyperILat = self.hgnnLayer(iiHyper, hv)



        uhfeat = hyperULat.permute(1, 0, 2)  # 10, 2300, 30
        ihfeat = hyperILat.permute(1, 0, 2)
        uhfeat = self.crossrate(uhfeat)
        ihfeat = self.crossrate(ihfeat)



        uhfeat = th.sum(uhfeat, dim=1)# 2300, 10, 30
        ihfeat = th.sum(ihfeat, dim=1)


        uhfeat = self.dropout(uhfeat)
        ihfeat = self.dropout(ihfeat)
        uhfeat = self.ufc2(uhfeat)
        ihfeat = self.ifc2(ihfeat)

        uhfeat = self.agg_act(uhfeat)
        ihfeat = self.agg_act(ihfeat)


        return ufeat, ifeat, uhfeat, ihfeat, rcloss

def udf_u_mul_e_norm(edges):
    return {'reg' : edges.src['reg'] * edges.dst['ci']}
    #out_feats = edges.src['reg'].shape[1] // 3
    #return {'reg' : th.cat([edges.src['reg'][:, :out_feats] * edges.dst['ci'], edges.src['reg'][:, out_feats:out_feats*2], edges.src['reg'][:, out_feats*2:]], 1)}

def udf_u_add_e(edges):
    return {'m' : th.cat([edges.src['r'], edges.dst['r']], 1)}

def udf_uv_add_r(edges):
    return {'m' : th.cat([edges.src['h'], edges.dst['h']], 1)}

def udf_v_add_r(edges):
    return {'m' : edges.src['h']}

class MLPDecoder(nn.Module):

    def __init__(self,
                 in_units,
                 num_classes,
                 neg,
                 neg_test,
                 neg_valid,
                 num_basis=2,
                 dropout_rate=0.0):
        super(MLPDecoder, self).__init__()
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Sequential(
            nn.Linear(in_units * 2, in_units, bias=False),
            nn.ReLU(),
            nn.Linear(in_units, 64, bias=False),
        )
        self.neg = neg
        self.neg_test = neg_test
        self.neg_valid = neg_valid
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, graph, ufeat, ifeat):

        with graph.local_scope():
            CL = InfoNCE(negative_mode='paired')
            #CL = InfoNCE()
            graph.nodes['movie'].data['r'] = ifeat
            graph.nodes['user'].data['r'] = ufeat
            review_feat = graph.edata['review_feat']
            graph.apply_edges(udf_u_add_e)
            out = graph.edata['m']
            if out.shape[0] > 10000:
                neg_sample = self.neg
            elif out.shape[0] == 6470:
                neg_sample = self.neg_test
                return 0
            else:
                neg_sample = self.neg_valid
                return 0
            out = self.linear(out)
            #print(out.shape, review_feat.shape, neg_sample.shape)
            cl_score = CL(out, review_feat, neg_sample)
            #cl_score = CL(out, review_feat)
        return cl_score

class BiDecoder(nn.Module):
    def __init__(self,
                 in_units,
                 num_classes,
                 num_basis=2,
                 dropout_rate=0.0):
        super(BiDecoder, self).__init__()
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.Ps = nn.ParameterList(
            nn.Parameter(th.randn(in_units, in_units))
            for _ in range(num_basis))
        self.combine_basis = nn.Linear(self._num_basis, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat, ifeat):
        with graph.local_scope():
            ufeat = self.dropout(ufeat)
            ifeat = self.dropout(ifeat)
            graph.nodes['movie'].data['h'] = ifeat
            basis_out = []
            for i in range(self._num_basis):
                graph.nodes['user'].data['h'] = ufeat @ self.Ps[i]
                graph.apply_edges(fn.u_dot_v('h', 'h', 'sr'))
                basis_out.append(graph.edata['sr'])
            out = th.cat(basis_out, dim=1)
            out = self.combine_basis(out)
        return out




def dot_or_identity(A, B, device=None):
    # if A is None, treat as identity matrix
    if A is None:
        return B
    elif A.shape[1] == 3:
        if device is None:
            return th.cat([B[A[:, 0].long()], B[A[:, 1].long()], B[A[:, 2].long()]], 1)
        else:
            #return th.cat([B[A[:, 0].long()], B[A[:, 1].long()]], 1).to(device)
            return th.cat([B[A[:, 0].long()], B[A[:, 1].long()] , B[A[:, 2].long()]], 1).to(device)
    else:
        return A

class Attentioncross(nn.Module):
    def __init__(self, in_size=None, device=None, rate=None):
        super(Attentioncross, self).__init__()
        self.device = device
        self.project = nn.Sequential(
            nn.Linear(in_size, 16, bias=False),  
            nn.LeakyReLU(),
            nn.Linear(16, 1, bias=False)
        )
        # print("att")
        # print(in_size)
        self.pro = nn.ModuleList()
        for i in range(rate):
            self.pro.append(self.project)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z): # z 5541, 5, 90  review 5, 5541, 90
        R = z.shape[1] # 5
        N = z.shape[0] # 5541
        D = z.shape[2] # 90
        Z1 = z.permute(1,0,2) # 5, 5541, 90
        Z3 = th.zeros(R, N, D).to(self.device) # 5, 5541, 90
        for i in range(R):
            Z2 = th.zeros(3, N, D).to(self.device) # 3, 5541, 90
            #Zr = th.zeros(3, N, 90).to(self.device)
            if i == 0:
                Z2[0] = Z1[i]
                Z2[1] = Z1[i+1]
                Z2[2] = Z1[i + 2]
                # Zr[0] = review[i]
                # Zr[1] = review[i+1]
                # Zr[2] = review[i + 2]
            elif i == R-1:
                Z2[0] = Z1[i]
                Z2[1] = Z1[i - 1]
                Z2[2] = Z1[i - 2]
                # Zr[0] = review[i]
                # Zr[1] = review[i - 1]
                # Zr[2] = review[i - 2]
            else:
                Z2[0] = Z1[i]
                Z2[1] = Z1[i + 1]
                Z2[2] = Z1[i - 1]
                # Zr[0] = review[i]
                # Zr[1] = review[i + 1]
                # Zr[2] = review[i - 1]
            Z2 = Z2.permute(1, 0, 2) # 5541, 3, 90
            #Zr = Zr.permute(1, 0, 2) # 5541, 3, 90
            #input = th.cat((Z2, Zr), dim=2)
            input = Z2
            w = self.pro[i](input)  
            beta = th.softmax(w, dim=1)
            beta = beta[:, 1:,:]
            z2 = Z2[:, 1:,:]
            o = (beta * z2).sum(1)  
            Z3[i] = o + Z1[i]
        out = Z3.permute(1,0,2)
        return out

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, 1, bias=False), 
            nn.LeakyReLU(),
        )
        self.reset_parameters()


    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z):
        w = self.project(z) 
        beta = th.softmax(w, dim=1)
        out = (beta * z).sum(1)
        return out


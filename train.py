"""Training GCMC model on the MovieLens data set.

The script loads the full graph to the training device.
"""
import os, time
import argparse
import logging
import random
import string
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from data import DataSetLoader
#from data_custom import DataSetLoader
from model import BiDecoder, GCMCLayer, MLPDecoder
from utils import get_activation, get_optimizer, torch_total_param_num, torch_net_info, MetricLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from utils import to_etype_name
from info_nce import InfoNCE

class Net(nn.Module):
    def __init__(self, args, dataset):
        super(Net, self).__init__()
        self._act = get_activation(args.model_activation)
        self.encoder = nn.ModuleList()

        self.encoder.append(GCMCLayer(args.hyperedge_num,
                                 args.rating_vals,
                                 args.src_in_units,
                                 args.dst_in_units,
                                 args.gcn_agg_units,
                                 args.gcn_out_units,
                                 args.gcn_dropout,
                                 args.gcn_agg_accum,
                                 dataset,
                                 agg_act=self._act,
                                 share_user_item_param=args.share_param,
                                 device=args.device,
                                 user_num=args.u_num,
                                 num=args.src_in_units))
        self.gcn_agg_accum = args.gcn_agg_accum
        self.rating_vals = args.rating_vals
        self.device = args.device
        self.data = dataset
        self.gcn_agg_units = args.gcn_agg_units
        self.src_in_units = args.src_in_units
        for i in  range(1, args.layers):
            gcn_out_units = args.gcn_out_units
            self.encoder.append(GCMCLayer(args.hyperedge_num,
                                        args.rating_vals,
                                        args.gcn_out_units,
                                        args.gcn_out_units,
                                        gcn_out_units,
                                        args.gcn_out_units,
                                        args.gcn_dropout - i*0.1,
                                        args.gcn_agg_accum,
                                        dataset,
                                        agg_act=self._act,
                                        share_user_item_param=args.share_param,
                                        ini = False,
                                        device=args.device,
                                        user_num=args.u_num,
                                        num=args.src_in_units))

        if args.decoder == "Bi":
            self.decoder = BiDecoder(in_units= args.gcn_out_units, #* args.layers,
                                     num_classes=len(args.rating_vals),
                                     num_basis=args.gen_r_num_basis_func)


        elif args.decoder == "MLP":
            if args.loss_func == "CE":
                num_classes = len(args.rating_vals)
            else:
                num_classes = 1
            self.decoder = MLPDecoder(in_units= args.gcn_out_units,
                                     num_classes=num_classes,
                                     num_basis=args.gen_r_num_basis_func)

        num_classes = len(args.rating_vals)
        self.decoderCL = MLPDecoder(in_units= args.gcn_out_units,
                                     num_classes=num_classes,
                                     neg=args.neg_sample,
                                    neg_test=args.neg_sample_test,
                                    neg_valid=args.neg_sample_valid,
                                     num_basis=args.gen_r_num_basis_func)


        self.rating_vals = args.rating_vals
          
    def forward(self, enc_graph, dec_graph, ufeat, ifeat, uhfeat, ihfeat, Two_Stage = False):
        #user_out = []
        #movie_out = []
        closs = 0.0
        rcloss = 0.0
        CL = InfoNCE(negative_mode='paired')
        for i in range(0, args.layers):
            user_o, movie_o, user_h, movie_h, rloss = self.encoder[i](
                enc_graph,
                ufeat,
                ifeat,
                uhfeat,
                ihfeat,
                Two_Stage)

            ufeat = user_o + user_h
            ifeat = movie_o + movie_h
            if i == 0:
                user_local = user_o
                movie_local = movie_o
                user_hyper = user_h
                movie_hyper = movie_h
                user_out = ufeat
                movie_out = ifeat
            else:
                user_local = user_local + user_o / float(i + 1)
                movie_local = movie_local + movie_o / float(i + 1)
                user_hyper = user_hyper + user_h / float(i + 1)
                movie_hyper = movie_hyper + movie_h / float(i + 1)
                user_out = user_out + ufeat / float(i + 1)
                movie_out = movie_out + ifeat / float(i + 1)
            

            closs = closs + CL(user_local, user_hyper) + CL(movie_local, movie_hyper)
            rcloss = rcloss + rloss


        pred_ratings = self.decoder(dec_graph, user_out, movie_out)
        predcl = self.decoderCL(dec_graph, user_out, movie_out)
        W_r_last = None
        reg_loss = 0.0
        W = th.matmul(self.encoder[0].att, self.encoder[0].basis.view(self.encoder[0].basis_units, -1))
        W = W.view(len(self.rating_vals), self.src_in_units, -1)
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            if i != 0:
                reg_loss += -th.sum(th.cosine_similarity(W[i,:,:], W[i-1,:,:], dim=1))

        return pred_ratings, reg_loss, closs, rcloss, predcl

def evaluate(args, net, dataset, segment='valid', debug = False, idx = 0):
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = th.FloatTensor(possible_rating_values).to(args.device)
    if segment == "valid":
        rating_values = dataset.valid_truths
        enc_graph = dataset.valid_enc_graph
        dec_graph = dataset.valid_dec_graph
    elif segment == "test":
        rating_values = dataset.test_truths
        enc_graph = dataset.test_enc_graph
        dec_graph = dataset.test_dec_graph
        user_map = dataset.global_user_id_map
        movie_map = dataset.global_movie_id_map
    else:
        raise NotImplementedError

    # Evaluate RMSE
    net.eval()
    with th.no_grad():
        pred_ratings, reg_loss, _, _, _ = net(enc_graph, dec_graph,
                           dataset.user_feature, dataset.movie_feature, dataset.user_feature, dataset.movie_feature)
    if args.loss_func == "CE":
        test_gt_labels = dataset.test_labels
        pre = th.softmax(pred_ratings, dim=1)
        max_rating, max_indices = th.max(pre, dim=1)
        pred = nd_possible_rating_values[max_indices]
        real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                            nd_possible_rating_values.view(1, -1)).sum(dim=1)

        test_user = dataset.test_rating_pairs[0]
        test_movie = dataset.test_rating_pairs[1]




    elif args.loss_func == "MSE":
        real_pred_ratings = pred_ratings[:, 0]
    mse = ((real_pred_ratings - rating_values) ** 2.).mean().item()
    mae = (abs(real_pred_ratings - rating_values)).mean().item()
    #mape = (abs((real_pred_ratings - rating_values)/rating_values)*100).mean().item()
    mse = mse
    if segment == "test":
        print("MSE", mse)
        print("MAE", mae)
    return mse


def CrossEntropyLoss(output, target):
    #print(target, target.shape)
    #print(output[0])
    ind = target.view(-1, 1)
    #print(ind, ind.shape)
    in1 = th.zeros(ind.shape[0], 2, dtype=th.int64)
    for i in range(ind.shape[0]):
        if ind[i] == 0:
            in1[i][0] = 1
            in1[i][1] = 2
        elif ind[i] == 9:
            in1[i][0] = 8
            in1[i][1] = 7
        else:
            in1[i][0] = ind[i] - 1
            in1[i][1] = ind[i] + 1
    #print(in1)
    res1 = -output.gather(dim=1, index=in1)
    res1 += th.log(th.exp(output).sum(dim=1).view(-1, 1))
    res1 = res1.mean()

    res = -output.gather(dim=1, index=ind)
    res += th.log(th.exp(output).sum(dim=1).view(-1, 1))
    res = res.mean()
    return res + 0.2 * res1



def calcRegLoss(model):
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    # ret += (model.usrStruct + model.itmStruct)
    return ret

def contrastLoss(embeds1, embeds2, temp):

    pckEmbeds1 = embeds1
    pckEmbeds2 = embeds2

    nume = th.exp(th.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
    deno = th.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
    return -th.log(nume / deno).mean()

def train(args):
    print(args)
    dataset = DataSetLoader(args.data_name, args.device,
                use_one_hot_fea=args.use_one_hot_fea,
                symm=args.gcn_agg_norm_symm,
                test_ratio=args.data_test_ratio,
                valid_ratio=args.data_valid_ratio)
    alpha = args.alpha
    beta = args.beta
    #dataset = MovieLens(args.data_name, args.device, use_one_hot_fea=args.use_one_hot_fea, symm=args.gcn_agg_norm_symm,
    #                    test_ratio=args.data_test_ratio, valid_ratio=args.data_valid_ratio, sparse_ratio = args.sparse_ratio)
    print("Loading data finished ...\n")

    args.src_in_units = dataset.user_feature_shape[1]
    args.dst_in_units = dataset.movie_feature_shape[1]
    args.rating_vals = dataset.possible_rating_values
    args.u_num = dataset.user_feature.shape[0]
    print(args.u_num)
    args.neg_sample = dataset.neg_sample
    args.neg_sample_test = dataset.neg_sample_test
    args.neg_sample_valid = dataset.neg_sample_valid


    ### build the net
    net = Net(args=args, dataset=dataset)
    args.decoder = "MLP"
    net = net.to(args.device)
    nd_possible_rating_values = th.FloatTensor(dataset.possible_rating_values).to(args.device)
    rating_loss_net = nn.CrossEntropyLoss()
    learning_rate = args.train_lr
    optimizer = get_optimizer(args.train_optimizer)(net.parameters(), lr=learning_rate)
    print("Loading network finished ...\n")

    ### perpare training data
    train_gt_labels = dataset.train_labels
    

    if args.data_name in ['ml-100k', 'amazon', 'douban2', 'toys_and_games', 'CDs_and_Vinyl', 'Clothing', 'yelp', 'toys']:
        train_gt_labels_left = th.zeros(train_gt_labels.shape[0], dtype=th.long).to(args.device)
        train_gt_labels_right = th.zeros(train_gt_labels.shape[0], dtype=th.long).to(args.device)
        for i, rate in enumerate(train_gt_labels):
            if rate == 0:
                train_gt_labels_left[i] = 0
                train_gt_labels_right[i] = 1
            elif rate == 4:
                train_gt_labels_left[i] = 3
                train_gt_labels_right[i] = 4
            else:
                train_gt_labels_left[i] = rate - 1
                train_gt_labels_right[i] = rate + 1
    elif args.data_name == 'ml-1m':
        train_gt_labels_left = th.zeros(train_gt_labels.shape[0], dtype=th.long).to(args.device)
        train_gt_labels_right = th.zeros(train_gt_labels.shape[0], dtype=th.long).to(args.device)
        for i, rate in enumerate(train_gt_labels):
            if rate == 0:
                train_gt_labels_left[i] = 0
                train_gt_labels_right[i] = 1
            elif rate == 4:
                train_gt_labels_left[i] = 3
                train_gt_labels_right[i] = 4
            else:
                train_gt_labels_left[i] = rate - 1
                train_gt_labels_right[i] = rate + 1
    elif args.data_name == 'flixster':
        train_gt_labels_left = th.zeros(train_gt_labels.shape[0], dtype=th.long).to(args.device)
        train_gt_labels_right = th.zeros(train_gt_labels.shape[0], dtype=th.long).to(args.device)
        train_gt_labels_left2 = th.zeros(train_gt_labels.shape[0], dtype=th.long).to(args.device)
        train_gt_labels_right2 = th.zeros(train_gt_labels.shape[0], dtype=th.long).to(args.device)
        for i, rate in enumerate(train_gt_labels):
            if rate == 0:
                train_gt_labels_left[i] = 1
                train_gt_labels_right[i] = 2
                train_gt_labels_left2[i] = 3
                train_gt_labels_right2[i] = 4
            elif rate == 1:
                train_gt_labels_left[i] = 0
                train_gt_labels_right[i] = 2
                train_gt_labels_left2[i] = 0
                train_gt_labels_right2[i] = 3
            elif rate == 9:
                train_gt_labels_left[i] = 8
                train_gt_labels_right[i] = 7
                train_gt_labels_left2[i] = 6
                train_gt_labels_right2[i] = 5
            elif rate == 8:
                train_gt_labels_left[i] = 7
                train_gt_labels_right[i] = 9
                train_gt_labels_left2[i] = 6
                train_gt_labels_right2[i] = 9
            else:
                train_gt_labels_left[i] = rate - 1
                train_gt_labels_right[i] = rate + 1
                train_gt_labels_left2[i] = rate - 2
                train_gt_labels_right2[i] = rate + 2
    elif args.data_name == 'yahoo_music':
        train_gt_labels_left = th.zeros(train_gt_labels.shape[0], dtype=th.long).to(args.device)
        train_gt_labels_right = th.zeros(train_gt_labels.shape[0], dtype=th.long).to(args.device)
        train_gt_labels_left2 = th.zeros(train_gt_labels.shape[0], dtype=th.long).to(args.device)
        train_gt_labels_right2 = th.zeros(train_gt_labels.shape[0], dtype=th.long).to(args.device)
        for i, rate in enumerate(train_gt_labels):
            if rate == 0:
                train_gt_labels_left[i] = 1
                train_gt_labels_right[i] = 2
                train_gt_labels_left2[i] = 3
                train_gt_labels_right2[i] = 4
            elif rate == 1:
                train_gt_labels_left[i] = 0
                train_gt_labels_right[i] = 2
                train_gt_labels_left2[i] = 0
                train_gt_labels_right2[i] = 3
            elif rate == 67:
                train_gt_labels_left[i] = 66
                train_gt_labels_right[i] = 65
                train_gt_labels_left2[i] = 64
                train_gt_labels_right2[i] = 63
            elif rate == 66:
                train_gt_labels_left[i] = 65
                train_gt_labels_right[i] = 67
                train_gt_labels_left2[i] = 64
                train_gt_labels_right2[i] = 67
            else:
                train_gt_labels_left[i] = rate - 1
                train_gt_labels_right[i] = rate + 1
                train_gt_labels_left2[i] = rate - 2
                train_gt_labels_right2[i] = rate + 2
    
    train_gt_ratings = dataset.train_truths
    rating_values = dataset.test_truths

    ### prepare the logger
    train_loss_logger = MetricLogger(['iter', 'loss', 'mse'], ['%d', '%.4f', '%.4f'],
                                     os.path.join(args.save_dir, 'train_loss%d.csv' % args.save_id))
    valid_loss_logger = MetricLogger(['iter', 'mse'], ['%d', '%.4f'],
                                     os.path.join(args.save_dir, 'valid_loss%d.csv' % args.save_id))
    test_loss_logger = MetricLogger(['iter', 'mse'], ['%d', '%.4f'],
                                    os.path.join(args.save_dir, 'test_loss%d.csv' % args.save_id))

    ### declare the loss information
    best_test_mse = np.inf
    no_better_valid = 0
    best_iter = -1
    count_mse = 0
    count_num = 0
    count_loss = 0
    
    dataset.train_enc_graph = dataset.train_enc_graph.int().to(args.device)
    dataset.train_dec_graph = dataset.train_dec_graph.int().to(args.device)
    dataset.valid_enc_graph = dataset.train_enc_graph
    dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(args.device)
    dataset.test_enc_graph = dataset.test_enc_graph.int().to(args.device)
    dataset.test_dec_graph = dataset.test_dec_graph.int().to(args.device)
    train_user = dataset.train_rating_pairs[0]
    train_movie = dataset.train_rating_pairs[1]
    print("Start...")
    dur = []




    for iter_idx in range(1, args.train_max_iter):
        if iter_idx > 3:
            t0 = time.time()
        net.train()
        if iter_idx > 250:
            Two_Stage = True
        else:
            Two_Stage = False
        Two_Stage = False
        pred_ratings, reg_loss, closs, rcloss, predcl = net(dataset.train_enc_graph, dataset.train_dec_graph,
                           dataset.user_feature, dataset.movie_feature, dataset.user_feature, dataset.movie_feature, Two_Stage)
        real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                             nd_possible_rating_values.view(1, -1)).sum(dim=1)
        train_preds = pred_ratings.argmax(dim=1)


        if args.loss_func == "CE":
            CE = rating_loss_net(pred_ratings, train_gt_labels).mean()
            CE_left = rating_loss_net(pred_ratings, train_gt_labels_left).mean()
            CE_right = rating_loss_net(pred_ratings, train_gt_labels_right).mean()
            #C = CE  + rcloss * alpha + closs * beta + predcl * 0.05
            C = CE + 0.1 * CE_left + 0.1 * CE_right + rcloss * alpha + closs * beta + predcl * 0.05
            #C = CE + rcloss * alpha + closs * beta
            loss = C + args.ARR * reg_loss 
            print(loss, closs*0.01, rcloss*0.01, predcl * 0.05)



        elif args.loss_func == "Hinge":
            real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                                nd_possible_rating_values.view(1, -1)).sum(dim=1)
            gap = (real_pred_ratings - train_gt_labels) ** 2
            hinge_loss = th.where(gap > 1.0, gap*gap, gap).mean()
            loss = hinge_loss
        elif args.loss_func == "MSE":
            loss = th.mean((pred_ratings[:, 0] - nd_possible_rating_values[train_gt_labels]) ** 2) + args.ARR * reg_loss
            print(loss, closs * 0.01)
            loss = loss + 0.01 * closs

        print(learning_rate)
        count_loss += loss.item()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(net.parameters(), args.train_grad_clip)
        optimizer.step()

        if iter_idx > 3:
            dur.append(time.time() - t0)

        if iter_idx == 1:
            print("Total #Param of net: %d" % (torch_total_param_num(net)))
            print(torch_net_info(net, save_path=os.path.join(args.save_dir, 'net%d.txt' % args.save_id)))

        if args.loss_func == "CE":
            real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                                nd_possible_rating_values.view(1, -1)).sum(dim=1)
        elif args.loss_func == "MSE":
            real_pred_ratings = pred_ratings[:, 0]
        mse = ((real_pred_ratings - train_gt_ratings) ** 2).sum()
        count_mse += mse.item()
        count_num += pred_ratings.shape[0]

        if iter_idx % args.train_log_interval == 0:
            train_loss_logger.log(iter=iter_idx,
                                  loss=count_loss / (iter_idx + 1), mse=count_mse / count_num)
            logging_str = "Iter={}, loss={:.4f}, mse={:.4f}, time={:.4f}".format(
                iter_idx, count_loss / iter_idx, count_mse / count_num,
                np.average(dur))
            count_mse = 0
            count_num = 0

        if iter_idx % args.train_valid_interval == 0:
            valid_mse = evaluate(args=args, net=net, dataset=dataset, segment='valid', idx = iter_idx)
            valid_loss_logger.log(iter = iter_idx, mse = valid_mse)
            test_mse = evaluate(args=args, net=net, dataset=dataset, segment='test', idx=iter_idx)
            logging_str += ', Test MSE={:.4f}'.format(test_mse)
            test_loss_logger.log(iter=iter_idx, mse=test_mse)
            logging_str += ',\tVal MSE={:.4f}'.format(valid_mse)

            if test_mse < best_test_mse:
                no_better_test = 0
                best_iter = iter_idx
                best_test_mse = test_mse
            else:
                if iter_idx > args.train_early_stopping_patience:
                    no_better_test += 1
                    if learning_rate <= args.train_min_lr:
                        logging.info("Early stopping threshold reached. Stop training.")
                        break
                    if no_better_test > args.train_decay_patience:
                        new_lr = max(learning_rate * args.train_lr_decay_factor, args.train_min_lr)
                        if new_lr < learning_rate:
                            learning_rate = new_lr
                            logging.info("\tChange ""the LR to %g" % new_lr)
                            for p in optimizer.param_groups:
                                p['lr'] = learning_rate
                            no_better_test = 0
        if iter_idx % args.train_log_interval == 0:
            print(logging_str)
            
        print('Best Iter Idx={},Best Test MSE={:.4f}'.format(
            best_iter, best_test_mse))
        print("————————————————————————————————————————————————")
    train_loss_logger.close()
    valid_loss_logger.close()
    test_loss_logger.close()


def config():
    parser = argparse.ArgumentParser(description='PGMC')
    parser.add_argument('--seed', default=125, type=int) #123
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--data_name', default='yahoo_music', type=str,
                        help='The dataset name: ml-100k, ml-1m, ml-10m, flixster, douban, yahoo_music')
    parser.add_argument('--data_test_ratio', type=float, default=0.1) ## for ml-100k the test ration is 0.2
    parser.add_argument('--data_valid_ratio', type=float, default=0.05)
    parser.add_argument('--use_one_hot_fea', action='store_true', default=False)
    parser.add_argument('--model_activation', type=str, default="tanh")
    parser.add_argument('--gcn_dropout', type=float, default=0.7)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--hyperedge_num', type=int, default=32)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--gcn_agg_units', type=int, default=900)
    parser.add_argument('--gcn_agg_accum', type=str, default="stack")
    parser.add_argument('--gcn_out_units', type=int, default=75)
    parser.add_argument('--gen_r_num_basis_func', type=int, default=2)
    parser.add_argument('--train_max_iter', type=int, default=50000)
    parser.add_argument('--train_log_interval', type=int, default=1)
    parser.add_argument('--train_valid_interval', type=int, default=1)
    parser.add_argument('--train_optimizer', type=str, default="adam")
    parser.add_argument('--decoder', type=str, default="Bi")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_min_lr', type=float, default=0.001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=30)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--train_early_stopping_patience', type=int, default=600) 
    parser.add_argument('--share_param', default=True, action='store_true')
    parser.add_argument('--ARR', type=float, default='0.000004')
    parser.add_argument('--loss_func', type=str, default='CE')
    parser.add_argument('--sparse_ratio', type=float, default=0.0)
    args = parser.parse_args()
    args.device = th.device(args.device) if args.device >= 0 else th.device('cpu')
    ### configure save_fir to save all the info
    if args.save_dir is None:
        args.save_dir = args.data_name+"_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))
    if args.save_id is None:
        args.save_id = np.random.randint(20)
    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


if __name__ == '__main__':
    '''
    ml_1m : param, ARR = 0.0000004, gcn_agg_units = 1000, gcn_agg_accum = sum, tmse = 0.8322, valid_ratio = 0.05
    ml_100k : param, ARR = 0.000001, gcn_agg_units = 500, gcn_agg_accum = sum, tmse = 0.9046, valid_ratio = 0.05
    1lyaer ml_1m : param, ARR = 0.0000005, gcn_agg_units = 2400, gcn_agg_accum = sum, tmse = 0.8305, valid_ratio = 0.05, gcn_out_units = 75
    1layer ml_100k : param, pos_emb, ARR = 0.000005, gcn_agg_units = 750, gcn_agg_accum = sum, tmse = 0.8974, valid_ratio = 0.05, gcn_out_units = 75
    2layer ml_100k : param, pos_emb, ARR = 0.000005, gcn_agg_units = 750, gcn_agg_accum = sum, tmse = 0.8969, valid_ratio = 0.05, gcn_out_units = 75
    2lyaer ml_1m : param, ARR = 0.0000004, gcn_agg_units = 1800, gcn_agg_accum = sum, tmse = 0.8319, valid_ratio = 0.05, gcn_out_units = 75
    '''
    args = config()
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)
    train(args)

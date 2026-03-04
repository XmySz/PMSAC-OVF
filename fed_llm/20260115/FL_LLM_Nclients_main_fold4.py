# @Organization  : GUET
# @Author        : LuSenLiang
# @Time          : 2023/8/20 下午5:01
# @Function      :

import time
import numpy as np
import json
import os
import sys

# 获取当前脚本所在路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录路径
parent_dir = os.path.dirname(current_dir)

# 添加到 sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging

import copy
import datetime
import random
from model import *
from utils import *
from utils_Package.weight_perturbation import WPOptim
from lw2w import WeightNetwork, LossWeightNetwork, FeatureMatching, inner_objective, inner_objective1,outer_objective, validate
from l2t_ww.train.meta_optimizers import MetaSGD
from l2t_ww.check_model import check_model
from N_data_dataloaders_v1 import recorded_llm_multicenters_dataloader
from focal_loss import FocalLoss
from torch.nn.parallel import DataParallel
from torchsummary import summary
# dataset:camelyon17/prostate/Nuclei/5_hospital_lung_nodules/4_gastric_centers/cifar100/cifar10/tinyimage-net/LIDC
# fed:fedlwt/fedavg/fedprox/moon/harmofl
LLM_model = "dinov2"

pairs=[]
src_num=4
tgt_num=5
for i in range(src_num):
    for j in range(tgt_num):
        pairs.append((i,j))
print(pairs)
model = "resnet18"

dataset = "Orthopedics"
fed_structure = "llm_fedlwt"  # n个party/n+1个
num_classes = 2
input_shape = 224  # 32#
each_save_interval = 1
loss_function = "crossentropy"  # "crossentropy or focalloss" #只影响fedlwt，对比模型不用管
DATA_DIR = r"/mnt/yxyxlab/qyj/Orthopedics/0115/perfold/fold4"
save_roof = r"/mnt/yxyxlab/qyj/Orthopedics/llm_FL/model_save/20260120/fold4/%s_%s_%s_%s_%s_12" % (LLM_model, datetime.date.today(), fed_structure, model, dataset)
weights_path = r"/mnt/yxyxlab/qyj/Orthopedics/llm_FL/fed_llm/l2t_ww/resnet18-5c106cde.pth"

batch_size = 25
init_seed = 4000
comm_round = 12  # 框架大循环
epochs = 1  # 本地模型循环
# global_center_idx = 0  # 选择一个中心数据作为服务器模型的数据
if fed_structure == "fedavg":
    server_momentum = 1
else:
    server_momentum = 0
# publish_data1=["camelyon17","prostate","nuclei"]
# publish_data2=["cifar100","cifar10","tiny-imagenet"]#lidc还没封装
# private_data=["lung_nodules","gastric","lidc"]
if fed_structure == "HarmoFL":
    open_perturbe = True
else:
    open_perturbe = False
loss_alpha=None
if loss_function=="focalloss":
    loss_alpha={"东莞":0.75,"江门":0.75,"开平":0.75,"茂名":0.75,"粤北":0.75}
# print(loss_alpha)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=model, help='neural network used in training')
    parser.add_argument('--dataset', type=str, default=dataset, help='dataset_train used for training')
    # parser.add_argument('--global_center_idx', type=int, default=global_center_idx,
    #                     help='choose one center as the global')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid',
                        help='the data partitioning strategy only use for cifar100,cifar10,tinyimagenet,lidc:homo,iid,noniid-labeldir,noniid')
    parser.add_argument('--batch-size', type=int, default=batch_size,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.1)')
    parser.add_argument('--alpha_PI', type=float, default=0.3, help='prediction Imitation alpha(default: 0.1)')
    parser.add_argument('--temp', type=float, default=5, help='the temperature for FI')

    parser.add_argument('--epochs', type=int, default=epochs, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=4, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default=fed_structure,
                        help='communication strategy: fedavg/fedprox/moon/HarmoFL/fedlwt')
    parser.add_argument('--llm', type=str, default=LLM_model,
                        help=': dinov2')
    parser.add_argument('--comm_round', type=int, default=comm_round, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=init_seed, help="Random seed")
    parser.add_argument('--input-shape', type=int, default=input_shape, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default=DATA_DIR, help="Data directory")
    parser.add_argument('--reg', type=float, default=0.0001, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="%s/logs_auc_ratio/" % save_roof,
                        help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="%s/models_auc_ratio/" % save_roof,
                        help='Model directory path')
    parser.add_argument('--beta_distribution', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100,
                        help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=2,
                        help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None,
                        help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss_func', type=str, default=loss_function, help="crossentropy or focalloss")
    parser.add_argument('--focalloss_alpha', type=str, default=loss_alpha)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--use_project_head', type=int, default=0)
    parser.add_argument('--server_momentum', type=float, default=server_momentum, help='the server momentum (FedAvgM)')
    parser.add_argument('--global_pre_join_with_new_global', type=float, default=3,
                        help='devise the global_parameter_update')
    #######扰动参数#################
    parser.add_argument('--alpha', type=float, default=0.05, help='The hyper parameter of perturbation in HarmoFL')
    parser.add_argument('--open_perturbe', type=int, default=open_perturbe)
    ##########全局模型参数更新策略############
    parser.add_argument('--strategy', type=int, default=1, help='model_pre_matching')
    ##############异构参数################
    parser.add_argument('--num-classes', type=int, default=num_classes, help='the number of category')
    parser.add_argument('--source-model', default=LLM_model, type=str)
    parser.add_argument('--source-domain', default='imagenet', type=str)
    parser.add_argument('--source-path', type=str, default=None)
    parser.add_argument('--target-model', default=model, type=str)
    parser.add_argument('--weight-path', type=str, default=None)
    parser.add_argument('--wnet-path', type=str, default=None)
    parser.add_argument('--open_lw2w', type=int, default=False, help='model_pre_matching')
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--schedule', action='store_true', default=True)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--pairs', type=list, default=pairs)
    parser.add_argument('--meta-lr', type=float, default=1e-3, help='Initial learning rate for meta networks')
    parser.add_argument('--meta-wd', type=float, default=1e-3)
    parser.add_argument('--loss-weight', action='store_true', default=True)
    parser.add_argument('--loss-weight-type', type=str, default='relu6')
    parser.add_argument('--loss-weight-init', type=float, default=1.0)
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--source-optimizer', type=str, default='sgd')
    parser.add_argument('--experiment', default='logs', help='Where to store models')
    parser.add_argument('--target-mhsa', type=bool, default=False, help="utilize the mhsa")
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
    ###harmFL
    parser.add_argument('--imbalance', action='store_true', help='do not truncate train data to same length')
    ###########################
    args = parser.parse_args()
    return args


###=====================迁移===========================
def init_pairs(opt):
    return opt.pairs


def init_meta_model(opt, pairs, server_model, local_nets):
    local_models_name = opt.sites.copy()
    #server_model_name = [opt.sites[opt.global_center_idx]]
    #local_models_name.pop(opt.global_center_idx)
    server_model_name=[opt.llm]
    wnets = dict()
    lwnets = dict()
    wlw_weight_params = dict()
    target_params_dict = dict()
    target_branch_dict = dict()
    for net_i in local_models_name:
        wnet = WeightNetwork(opt.source_model, pairs)
        weight_params = list(wnet.parameters())
        if opt.loss_weight:
            lwnet = LossWeightNetwork(opt.source_model, pairs, opt.loss_weight_type, opt.loss_weight_init)
            weight_params = weight_params + list(lwnet.parameters())
        if opt.wnet_path is not None:
            ckpt = torch.load(opt.wnet_path)
            wnet.load_state_dict(ckpt['w'])
            if opt.loss_weight:
                lwnet.load_state_dict(ckpt['lw'])

        target_branch = FeatureMatching(opt.source_model,
                                        opt.target_model,
                                        pairs)
        # server_target_params = list(server_model[server_model_name[0]].parameters()) + copy.deepcopy(
        #     list(target_branch.parameters()))

        local_target_params = list(local_nets[net_i].parameters()) + copy.deepcopy(list(target_branch.parameters()))
        wnets["%sto%s" % (server_model_name[0], net_i)] = copy.deepcopy(wnet)
        #wnets["%sto%s" % (net_i, server_model_name[0])] = copy.deepcopy(wnet)
        lwnets["%sto%s" % (server_model_name[0], net_i)] = copy.deepcopy(lwnet)
        #lwnets["%sto%s" % (net_i, server_model_name[0])] = copy.deepcopy(lwnet)
        wlw_weight_params["%sto%s" % (server_model_name[0], net_i)] = copy.deepcopy(weight_params)
        #wlw_weight_params["%sto%s" % (net_i, server_model_name[0])] = copy.deepcopy(weight_params)
        target_params_dict["%sto%s" % (server_model_name[0], net_i)] = local_target_params
        #target_params_dict["%sto%s" % (net_i, server_model_name[0])] = server_target_params
        target_branch_dict["%sto%s" % (server_model_name[0], net_i)] = copy.deepcopy(target_branch)
        #target_branch_dict["%sto%s" % (net_i, server_model_name[0])] = copy.deepcopy(target_branch)
    return wnets, lwnets, wlw_weight_params, target_params_dict, target_branch_dict  # ,source_optimizer


def optimizer_init(opt, wlw_weight_params, target_model_dict, target_params_dict, target_branch_dict):
    source_optimizers = dict()  # n_parties,2n个优化器
    target_optimizers = dict()
    for i, (k, weight_params) in enumerate(wlw_weight_params.items()):
        if opt.source_optimizer == 'sgd':
            source_optimizer = optim.SGD(weight_params, lr=opt.meta_lr, weight_decay=opt.meta_wd, momentum=opt.momentum,
                                         nesterov=opt.nesterov)
        elif opt.source_optimizer == 'adam':
            source_optimizer = optim.Adam(weight_params, lr=opt.meta_lr, weight_decay=opt.meta_wd)
        source_optimizers[k] = source_optimizer
        if opt.meta_lr == 0:
            target_optimizer = optim.SGD(target_params_dict[k], lr=opt.lr, momentum=opt.momentum, weight_decay=opt.wd)
        else:
            target_optimizer = MetaSGD(target_params_dict[k],
                                       [target_model_dict[k.split("to")[-1]], target_branch_dict[k]],
                                       lr=opt.lr,
                                       momentum=opt.momentum,
                                       weight_decay=opt.wd, rollback=True, cpu=opt.T > 2)
        target_optimizers[k] = target_optimizer
    return source_optimizers, target_optimizers


# =====================================================================


# ==========================联邦学习====================================
def init_nets(args, device='cpu', server=False):
    if args.alg == "fedlwt":
        local_models = args.sites.copy()
        server_model = [args.sites[args.global_center_idx]]
        local_models.pop(args.global_center_idx)
    else:
        local_models = args.sites.copy()
        server_model = ["server"]
    ##fine tune weights
    checkpoint = torch.load(weights_path, map_location=device)
    #args.model = args.source_model if args.source_model == args.target_model else None  ####注意
    if server:
        nets = {net_i: None for net_i in server_model}
        for net_i in server_model:
            net = check_model(args).to(device)
            ###finetune imagenet
            new_params = net.state_dict().copy()
            for name, param in new_params.items():
                # print(name)
                if name in checkpoint and param.size() == checkpoint[name].size():
                    new_params[name].copy_(checkpoint[name])
                    # print('copy {}'.format(name))
            net.load_state_dict(new_params)
            nets[net_i] = net
        model_meta_data = []
        layer_type = []
        for (k, v) in nets[net_i].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)
        return nets, model_meta_data, layer_type
    else:
        nets = {net_i: None for net_i in local_models}
        for net_i in local_models:
            net = check_model(args).to(device)
            ###finetune imagenet
            new_params = net.state_dict().copy()
            #print(net)

            for name, param in new_params.items():
                #print(name)
                if name in checkpoint and param.size() == checkpoint[name].size():
                    new_params[name].copy_(checkpoint[name])
                    # print('copy {}'.format(name))
            net.load_state_dict(new_params)

            nets[net_i] = net
        model_meta_data = []
        layer_type = []
        for (k, v) in nets[net_i].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)

        return nets, model_meta_data, layer_type


# sim_dict={}

def train_net(net_id, net, train_dataloader, val_dataloader, test_dataloader, epochs, lr, args_optimizer, args,
              device="cpu", write_log=True):
    '''
    use for fedavg
    '''
    # net = nn.DataParallel(net)
    net.cuda(0)

    if write_log:
        logger.info('Training network %s' % str(net_id))
        logger.info('n_training: %d' % len(train_dataloader))
        logger.info('n_test: %d' % len(test_dataloader))
    print('Training network %s' % str(net_id))
    print('n_training: %d' % len(train_dataloader))
    print('n_test: %d' % len(test_dataloader))
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:  # 有些数据没有验证集
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    else:
        val_acc, val_auc = None, None
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)

    logger.info('>> Pre-Training Training accuracy: {}--->>auc:{}'.format(train_acc, train_auc))

    logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    logger.info('>> Pre-Training Test accuracy: {}------>>auc:{}'.format(test_acc, test_auc))
    print('>> Pre-Training Training accuracy: {}-------->>auc:{}'.format(train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Pre-Training Test accuracy: {}------------>>auc:{}'.format(test_acc, test_auc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda(0)

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(0), target.cuda(0)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out, _ = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)
    if write_log:
        logger.info('>> Training accuracy: {}---->>auc:{}'.format(train_acc, train_auc))
        logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
        logger.info('>> Test accuracy: {}----->>auc:{}'.format(test_acc, test_auc))
        logger.info(' ** Training complete **')
    print('>> Training accuracy: %f------->>auc:%f' % (train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Test accuracy: %f----------->>auc:%f' % (test_acc, test_auc))
    net.to('cpu')

    print(' ** Training complete **')
    return train_acc, val_acc, test_acc, train_auc, val_auc, test_auc


def train_net_fedprox(net_id, net, global_net, train_dataloader, val_dataloader, test_dataloader, epochs, lr,
                      args_optimizer, mu, args,
                      device="cpu", write_log=True):
    '''
    use for fedprox
    '''
    # global_net.to(device)
    # net = nn.DataParallel(net)
    net.cuda(0)
    # else:
    if write_log:
        logger.info('Training network %s' % str(net_id))
        logger.info('n_training: %d' % len(train_dataloader))
        logger.info('n_test: %d' % len(test_dataloader))
    print('Training network %s' % str(net_id))
    print('n_training: %d' % len(train_dataloader))
    print('n_test: %d' % len(test_dataloader))
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:  # 有些数据没有验证集
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    else:
        val_acc, val_auc = None, None
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)

    logger.info('>> Pre-Training Training accuracy: {}--->>auc:{}'.format(train_acc, train_auc))

    logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    logger.info('>> Pre-Training Test accuracy: {}------>>auc:{}'.format(test_acc, test_auc))
    print('>> Pre-Training Training accuracy: {}-------->>auc:{}'.format(train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Pre-Training Test accuracy: {}------------>>auc:{}'.format(test_acc, test_auc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda(0)

    cnt = 0
    global_weight_collector = list(global_net.cuda(0).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(0), target.cuda(0)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out, _ = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)
    if write_log:
        logger.info('>> Training accuracy: {}---->>auc:{}'.format(train_acc, train_auc))
        logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
        logger.info('>> Test accuracy: {}----->>auc:{}'.format(test_acc, test_auc))
        logger.info(' ** Training complete **')
    print('>> Training accuracy: %f------->>auc:%f' % (train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Test accuracy: %f----------->>auc:%f' % (test_acc, test_auc))
    net.to('cpu')

    print(' ** Training complete **')
    return train_acc, val_acc, test_acc, train_auc, val_auc, test_auc


def train_net_HarmoFL(net_id, net, global_net, train_dataloader, val_dataloader, test_dataloader, epochs, lr,
                      args_optimizer, mu, args,
                      device="cpu", write_log=True):
    '''
    use for HarmoFL
    '''
    from HarmoFL_utils.weight_perturbation import WPOptim
    # global_net.to(device)
    # net = nn.DataParallel(net)
    net.cuda(0)
    # else:
    if write_log:
        logger.info('Training network %s' % str(net_id))
        logger.info('n_training: %d' % len(train_dataloader))
        logger.info('n_test: %d' % len(test_dataloader))
    print('Training network %s' % str(net_id))
    print('n_training: %d' % len(train_dataloader))
    print('n_test: %d' % len(test_dataloader))
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:  # 有些数据没有验证集
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    else:
        val_acc, val_auc = None, None
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)

    logger.info('>> Pre-Training Training accuracy: {}--->>auc:{}'.format(train_acc, train_auc))

    logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    logger.info('>> Pre-Training Test accuracy: {}------>>auc:{}'.format(test_acc, test_auc))
    print('>> Pre-Training Training accuracy: {}-------->>auc:{}'.format(train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Pre-Training Test accuracy: {}------------>>auc:{}'.format(test_acc, test_auc))

    if args.dataset == 'prostate':
        optimizer = WPOptim(params=net.parameters(), base_optimizer=optim.Adam, lr=args.lr, alpha=args.alpha,
                            weight_decay=1e-4)
    else:
        optimizer = WPOptim(params=net.parameters(), base_optimizer=optim.SGD, lr=args.lr, alpha=args.alpha,
                            momentum=0.9, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss().cuda(0)

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(0), target.cuda(0)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out, _ = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.generate_delta(zero_grad=True)
            out, _ = net(x)
            criterion(out, target).backward()
            optimizer.step(zero_grad=True)

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)
    if write_log:
        logger.info('>> Training accuracy: {}---->>auc:{}'.format(train_acc, train_auc))
        logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
        logger.info('>> Test accuracy: {}----->>auc:{}'.format(test_acc, test_auc))
        logger.info(' ** Training complete **')
    print('>> Training accuracy: %f------->>auc:%f' % (train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Test accuracy: %f----------->>auc:%f' % (test_acc, test_auc))
    net.to('cpu')

    print(' ** Training complete **')
    return train_acc, val_acc, test_acc, train_auc, val_auc, test_auc


def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, val_dataloader, test_dataloader, epochs,
                     lr, args_optimizer, mu, temperature, args,
                     round, device="cpu", write_log=True):
    # net = nn.DataParallel(net)
    net.cuda(0)
    if write_log:
        logger.info('Training network %s' % str(net_id))
        logger.info('n_training: %d' % len(train_dataloader))
        logger.info('n_test: %d' % len(test_dataloader))
    print('Training network %s' % str(net_id))
    print('n_training: %d' % len(train_dataloader))
    print('n_test: %d' % len(test_dataloader))
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:  # 有些数据没有验证集
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    else:
        val_acc, val_auc = None, None
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)

    logger.info('>> Pre-Training Training accuracy: {}--->>auc:{}'.format(train_acc, train_auc))

    logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    logger.info('>> Pre-Training Test accuracy: {}------>>auc:{}'.format(test_acc, test_auc))
    print('>> Pre-Training Training accuracy: {}-------->>auc:{}'.format(train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Pre-Training Test accuracy: {}------------>>auc:{}'.format(test_acc, test_auc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda(0)
    # criterion=FocalLoss().cuda()
    # global_net.to(device)

    for previous_net in previous_nets:
        previous_net.cuda(0)
    global_w = global_net.state_dict()

    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(0), target.cuda(0)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out, pro1 = net(x)
            _, pro2 = global_net(x)

            posi = cos(pro1[-1], pro2[-1])
            logits = posi.reshape(-1, 1)

            for previous_net in previous_nets:
                previous_net.cuda(0)
                _, pro3 = previous_net(x)
                nega = cos(pro1[-1], pro3[-1])
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                previous_net.to('cpu')

            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda(0).long()

            loss2 = mu * criterion(logits, labels)

            loss1 = criterion(out, target)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))

    for previous_net in previous_nets:
        previous_net.to('cpu')
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)
    if write_log:
        logger.info('>> Training accuracy: {}---->>auc:{}'.format(train_acc, train_auc))
        logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
        logger.info('>> Test accuracy: {}----->>auc:{}'.format(test_acc, test_auc))
        logger.info(' ** Training complete **')
    print('>> Training accuracy: %f------->>auc:%f' % (train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Test accuracy: %f----------->>auc:%f' % (test_acc, test_auc))
    net.to('cpu')

    print(' ** Training complete **')
    return train_acc, val_acc, test_acc, train_auc, val_auc, test_auc


def train_net_fedlwt(net_id, net,  # target_model
                     source_optimizer, target_optimizer, wnet, lwnet, target_branch,
                     global_net,  # source_model
                     previous_nets, train_dataloader, val_dataloader, test_dataloader, epochs,
                     lr, args_optimizer, mu, temperature, args, round, device="cpu", write_log=True):
    server_model_name = args.llm
    if write_log:
        logger.info('Training network %s' % str(net_id))
        logger.info('n_training: %d' % len(train_dataloader))
        logger.info('n_test: %d' % len(test_dataloader))
    print('Training network %s' % str(net_id))
    print('n_training: %d' % len(train_dataloader))
    print('n_test: %d' % len(test_dataloader))
    
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:  # 有些数据没有验证集
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    else:
        val_acc, val_auc = None, None
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)

    logger.info('>> Pre-Training Training accuracy: {}--->>auc:{}'.format(train_acc, train_auc))

    logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    logger.info('>> Pre-Training Test accuracy: {}------>>auc:{}'.format(test_acc, test_auc))
    print('>> Pre-Training Training accuracy: {}-------->>auc:{}'.format(train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Pre-Training Test accuracy: {}------------>>auc:{}'.format(test_acc, test_auc))
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.momentum,
                              weight_decay=args.reg)
    elif args_optimizer == "WPOtim":
        optimizer = WPOptim(params=net.parameters(), base_optimizer=optim.SGD, lr=lr, alpha=args.alpha,
                            momentum=args.momentum, weight_decay=args.reg)
    '''
    for previous_net in previous_nets:
        previous_net.cuda()
    global_w = global_net.state_dict()'''
    if args.loss_func=="crossentropy":
        criterion = nn.CrossEntropyLoss()
    elif args.loss_func=="focalloss":
        criterion = FocalLoss(args.focalloss_alpha[net_id]).cuda(0)
    cnt = 0
    # cos = torch.nn.CosineSimilarity(dim=-1)
    state = dict()
    # mu = 0.001
    for epoch in range(epochs):
        state['epoch'] = epoch
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        epoch_transfer_loss_collector = []
        lw_avg_batch_collector=[]
        net.train()  # target_model
          # source_model
        global_net.eval()
        for batch_idx, data in enumerate(train_dataloader):
            ####迁移##########

            state['iter'] = batch_idx
            target_optimizer.zero_grad()
            loss,lw_avg = inner_objective(data, args, net, global_net, wnet, lwnet,
                                   target_branch, state=state, logger=logger,
                                   source_model_name=server_model_name, target_model_name=net_id, device=device)
            lw_avg_batch_collector.append(lw_avg)
            loss.backward()
            target_optimizer.step(None)

            for _ in range(args.T):
                target_optimizer.zero_grad()
                target_optimizer.step(inner_objective1, data, args, net, global_net, wnet, lwnet,
                                      target_branch, state, logger, server_model_name, net_id, True)
            target_optimizer.zero_grad()
            target_optimizer.step(outer_objective, data, args, net, state,net_id)
            target_optimizer.zero_grad()
            source_optimizer.zero_grad()
            loss = outer_objective(data, args, net, state,net_id, device=device)
            
            loss.backward()
            target_optimizer.meta_backward()
            source_optimizer.step()
            epoch_transfer_loss_collector.append(loss)
            x, target = data[0].cuda(0), data[2].cuda(0)
            out, pro1 = net(x)
            loss1 = criterion(out, target)  # 当前网络交叉熵损失
            
            loss1.backward()
            optimizer.step()
            epoch_loss1_collector.append(loss1.item())
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        lw_avg_batch_collector = np.stack(lw_avg_batch_collector)
        lw_avg_batch = np.mean(lw_avg_batch_collector, axis=0)
        top_three_indices = np.argpartition(lw_avg_batch, -3)[-3:]
        print(lw_avg_batch,"top3_lw",top_three_indices)
        '''
            ####contrast_learning#######
            x, target = data[0].cuda(), data[1].cuda()
            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()
            out, pro1 = net(x)
            _, pro2 = global_net(x)
            posi = cos(pro1[-1], pro2[-1])  # 求每个模型与全局模型的余弦相似度
            logits = posi.reshape(-1, 1)
            for previous_net in previous_nets:
                previous_net.cuda()
                _, pro3 = previous_net(x)
                nega = cos(pro1[-1], pro3[-1])  # 计算当前模型与之前各个模型的余弦相似度
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                previous_net.to('cpu')
            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda().long()
            loss2 = mu * criterion(logits, labels)  # 应该是对比损失
            loss1 = criterion(out, target)  # 当前网络交叉熵损失
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())
        # print(epoch,sim_dict)
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)'''
        epoch_transfer_loss = sum(epoch_transfer_loss_collector) / len(epoch_transfer_loss_collector)
        if write_log:
            # logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f transfer_loss:%f' % (
            # epoch, epoch_loss, epoch_loss1, epoch_loss2, epoch_transfer_loss))
            logger.info('Epoch: %d  transfer_loss:%f Loss1: %f' % (epoch, epoch_transfer_loss,epoch_loss1))
        # print('Epoch: %d Loss: %f Loss1: %f Loss2: %f transfer_loss:%f' % (
        # epoch, epoch_loss, epoch_loss1, epoch_loss2, epoch_transfer_loss))
        print('Epoch: %d  transfer_loss:%f Loss1: %f' % (epoch, epoch_transfer_loss,epoch_loss1))
    '''
    for previous_net in previous_nets:
        previous_net.to('cpu')'''
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)
    if write_log:
        logger.info('>> Training accuracy: {}---->>auc:{}'.format(train_acc, train_auc))
        logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
        logger.info('>> Test accuracy: {}----->>auc:{}'.format(test_acc, test_auc))
        logger.info(' ** Training complete **')
    print('>> Training accuracy: %f------->>auc:%f' % (train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Test accuracy: %f----------->>auc:%f' % (test_acc, test_auc))
    net.to('cpu')

    print(' ** Training complete **')
    return train_acc, val_acc, test_acc, train_auc, val_auc, test_auc

def train_net_llm_fedlwt(net_id, net,  # target_model
                     source_optimizer, target_optimizer, wnet, lwnet, target_branch,llm_net,
                     global_net,  # source_model
                     previous_nets, train_dataloader, val_dataloader, test_dataloader, epochs,
                     lr, args_optimizer, mu, temperature, args, round, device="cpu", write_log=True):
    server_model_name = args.llm
    if write_log:
        logger.info('Training network %s' % str(net_id))
        logger.info('n_training: %d' % len(train_dataloader))
        logger.info('n_test: %d' % len(test_dataloader))
    print('Training network %s' % str(net_id))
    print('n_training: %d' % len(train_dataloader))
    print('n_test: %d' % len(test_dataloader))
    
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:  # 有些数据没有验证集
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    else:
        val_acc, val_auc = None, None
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)

    logger.info('>> Pre-Training Training accuracy: {}--->>auc:{}'.format(train_acc, train_auc))

    logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    logger.info('>> Pre-Training Test accuracy: {}------>>auc:{}'.format(test_acc, test_auc))
    print('>> Pre-Training Training accuracy: {}-------->>auc:{}'.format(train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Pre-Training Test accuracy: {}------------>>auc:{}'.format(test_acc, test_auc))
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.momentum,
                              weight_decay=args.reg)
    elif args_optimizer == "WPOtim":
        optimizer = WPOptim(params=net.parameters(), base_optimizer=optim.SGD, lr=lr, alpha=args.alpha,
                            momentum=args.momentum, weight_decay=args.reg)
    '''
    for previous_net in previous_nets:
        previous_net.cuda()
    global_w = global_net.state_dict()'''
    if args.loss_func=="crossentropy":
        criterion = nn.CrossEntropyLoss()
    elif args.loss_func=="focalloss":
        criterion = FocalLoss(args.focalloss_alpha[net_id]).cuda(0)

    soft_loss = nn.KLDivLoss(reduction="batchmean")
    cnt = 0
    # cos = torch.nn.CosineSimilarity(dim=-1)
    state = dict()
    # mu = 0.001
    for epoch in range(epochs):
        state['epoch'] = epoch
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        epoch_transfer_loss_collector = []
        lw_avg_batch_collector=[]
        net.train()  # target_model
          # source_model
        global_net.eval()
        llm_net.eval()
        #llm_transfer
        wnet.cuda(0)
        lwnet.cuda(0)
        target_branch.cuda(0)
        llm_net.cuda(0)
        for name, param in net.named_parameters(): #确认所有层解开冻结
            #print(name,param.requires_grad)
            param.requires_grad = True
        for batch_idx, data in enumerate(train_dataloader):
            ####迁移##########
            state['iter'] = batch_idx
            target_optimizer.zero_grad()
            loss,lw_avg = inner_objective(data, args, net, llm_net, wnet, lwnet,
                                   target_branch, state=state, logger=logger,
                                   source_model_name=server_model_name, target_model_name=net_id, device=device)
            lw_avg_batch_collector.append(lw_avg)
            loss.backward()
            target_optimizer.step(None)

            for _ in range(args.T):
                target_optimizer.zero_grad()
                target_optimizer.step(inner_objective1, data, args, net, llm_net, wnet, lwnet,
                                      target_branch, state, logger, server_model_name, net_id, True)
            target_optimizer.zero_grad()
            target_optimizer.step(outer_objective, data, args, net, state,net_id)
            target_optimizer.zero_grad()
            source_optimizer.zero_grad()
            loss = outer_objective(data, args, net, state,net_id, device=device)
            
            loss.backward()
            target_optimizer.meta_backward()
            source_optimizer.step()
            epoch_transfer_loss_collector.append(loss)
            x, target = data[0].cuda(0), data[2].cuda(0)
            out, pro1 = net(x)
            loss1 = criterion(out, target)  # 当前网络交叉熵损失
            
            loss1.backward()
            optimizer.step()
            epoch_loss1_collector.append(loss1.item())
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_transfer_loss = sum(epoch_transfer_loss_collector) / len(epoch_transfer_loss_collector)
        if write_log:
            logger.info('Epoch: %d  transfer_loss:%f Loss1: %f' % (epoch, epoch_transfer_loss,epoch_loss1))
        print('Epoch: %d  transfer_loss:%f Loss1: %f' % (epoch, epoch_transfer_loss,epoch_loss1))
        wnet.to('cpu')#释放GPU内存
        lwnet.to('cpu')
        target_branch.to('cpu')
        llm_net.to('cpu')

        #freeze_layers
        lw_avg_batch_collector = np.stack(lw_avg_batch_collector)
        lw_avg_batch = np.mean(lw_avg_batch_collector, axis=0)
        top_indices = np.argpartition(lw_avg_batch, -3)[-3:]
        print(lw_avg_batch,"generalization_top_layers_lw",top_indices)
        if write_log:
                logger.info("{} generalization_top_layers_lw:{}".format(lw_avg_batch,top_indices))
        for name, param in net.named_parameters():
            if name.startswith('conv1'):
                if 0 in top_indices:
                    #print(name)
                    param.requires_grad = False
            for k in top_indices:
                if name.startswith('layer%d.1'%k):
                    #print(name)
                    param.requires_grad = False
        if round!=0:
            #联邦通信
            global_net.cuda(0)#装载全局模型
            alpha=args.alpha_PI  #1007，1009:0.5，*temp*temp;1008，1112:0.3；1112_2，1117_2:0.1
            temp=args.temp
            for batch_idx, data in enumerate(train_dataloader):
                global_preds,_ = global_net(data[0].cuda(0))
                # student model forward
                local_preds,enc= net(data[0].cuda(0))
                cross_global_preds=global_net.cross_head_forward(enc[-1])
                student_loss = criterion(local_preds, data[2].cuda(0))
                ditillation_loss = soft_loss(
                    F.log_softmax(cross_global_preds/temp, dim = 1),
                    F.softmax(global_preds/temp, dim = 1)
                )
                if ditillation_loss.item()>4*epoch_loss1: #当前batch,本地模型与全局模型差异极大，则跳过本次参数更新
                    print(batch_idx,ditillation_loss.item())
                    if write_log:
                        logger.info("batch:{},ditillation_loss{}".format(batch_idx,ditillation_loss.item()))
                    continue
                loss2 = alpha * student_loss + (1 - alpha) * ditillation_loss # 温度的平方
                # backward
                optimizer.zero_grad()			#梯度初始化为0
                loss2.backward()				#反向传播
                optimizer.step()			#参数优化
                epoch_loss2_collector.append(ditillation_loss)
            #print(epoch_loss2_collector)
            if len(epoch_loss2_collector)!=0:
                epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
            else:
                epoch_loss2=0.0
            if write_log:
                logger.info('PI_loss:%f' % (epoch_loss2))
            print('PI_loss:%f' % (epoch_loss2))
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device)
    if val_dataloader is not None:
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device)
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device)
    if write_log:
        logger.info('>> Training accuracy: {}---->>auc:{}'.format(train_acc, train_auc))
        logger.info('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
        logger.info('>> Test accuracy: {}----->>auc:{}'.format(test_acc, test_auc))
        logger.info(' ** Training complete **')
    print('>> Training accuracy: %f------->>auc:%f' % (train_acc, train_auc))
    print('>> Pre-Training Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Test accuracy: %f----------->>auc:%f' % (test_acc, test_auc))
    net.to('cpu')
    global_net.to('cpu')
    print(' ** Training complete **')
    return train_acc, val_acc, test_acc, train_auc, val_auc, test_auc

def global_train_net(args,
                     source_optimizers_dict, target_optimizers_dict,
                     wnet_dict, lwnet_dict, target_branch_dict,
                     nets,
                     global_model=None, train_global_dl=None, val_global_dl=None, test_global_dl=None, device="cpu",
                     write_log=True):
    if args.alg == "fedlwt":
        server_model_name = [args.sites[args.global_center_idx]][0]
    else:
        server_model_name = "server"
    if global_model:
        global_model.cuda(0)
    if write_log:
        logger.info('global_model_Training network 1')
        logger.info('n_training: %d' % len(train_global_dl))
        logger.info('n_test: %d' % len(test_global_dl))
    state = dict()
    for epoch in range(args.epochs):
        state['epoch'] = epoch
        # torch.cuda.empty_cache()
        for source_model_name, source_model in nets.items():
            if source_model_name == server_model_name:
                continue
            epoch_loss1_collector = []
            global_model.train()  # target_model
            source_model.to(device)
            source_model.eval()
            dict_name = "%sto%s" % (source_model_name, server_model_name)
            target_optimizer = target_optimizers_dict[dict_name]
            source_optimizer = source_optimizers_dict[dict_name]
            # wnet_dict[dict_name].to(device)
            # lwnet_dict[dict_name].to(device)
            # target_branch_dict[dict_name].to(device)
            for batch_idx, data in enumerate(train_global_dl):
                state['iter'] = batch_idx
                target_optimizer.zero_grad()
                loss = inner_objective(data, args, global_model, source_model, wnet_dict[dict_name],
                                       lwnet_dict[dict_name], target_branch_dict[dict_name],
                                       state, logger=logger, source_model_name=source_model_name,
                                       target_model_name=server_model_name, device=device)
                loss.backward()
                target_optimizer.step(None)

                for _ in range(args.T):
                    target_optimizer.zero_grad()
                    target_optimizer.step(inner_objective, data, args, global_model, source_model,
                                          wnet_dict[dict_name], lwnet_dict[dict_name],
                                          target_branch_dict[dict_name], state, logger, source_model_name,
                                          server_model_name, True)
                target_optimizer.zero_grad()
                target_optimizer.step(outer_objective, data, args, global_model, state)
                target_optimizer.zero_grad()
                source_optimizer.zero_grad()
                loss = outer_objective(data, args, global_model, state, device=device)
                loss.backward()
                target_optimizer.meta_backward()
                source_optimizer.step()
                epoch_loss1_collector.append(loss)
            epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            if write_log:
                logger.info('%s model transfer to %s_Epoch: %d  Loss1: %f ' % (
                source_model_name, server_model_name, epoch, epoch_loss1))
            print('%s model transfer to %s_Epoch: %d  Loss1: %f ' % (
            source_model_name, server_model_name, epoch, epoch_loss1))
    train_acc, _, train_auc = compute_accuracy(global_model, train_global_dl, device=device)
    if val_global_dl is not None:
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(global_model, val_global_dl,
                                                                get_confusion_matrix=True,
                                                                device=device)
    else:
        val_acc, val_auc = None, None
    test_acc, conf_matrix, _, test_auc = compute_accuracy(global_model, test_global_dl,
                                                          get_confusion_matrix=True,
                                                          device=device)
    if write_log:
        logger.info('>> global_model_Training accuracy: {}---->>auc:{}'.format(train_acc, train_auc))
        logger.info('>> global_model_Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
        logger.info('>> global_model_Test accuracy: {}----->>auc:{}'.format(test_acc, test_auc))
        logger.info(' ** Training complete **')
    print('>> global_model_Training accuracy: %f------->>auc:%f' % (train_acc, train_auc))
    print('>> global_model_Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> global_model_Test accuracy: %f----------->>auc:%f' % (test_acc, test_auc))
    global_model.to('cpu')
    print(' ** Training complete **')
    return train_acc, val_acc, test_acc, train_auc, val_auc, test_auc


def local_train_net(nets, args,
                    source_optimizers_dict=None, target_optimizers_dict=None,
                    wnet_dict=None, lwnet_dict=None, target_branch_dict=None,
                    train_dl=None, val_dl=None, test_dl=None,llm_model=None,
                    global_model=None, prev_model_pool=None, server_c=None, clients_c=None, round=None, device="cpu",
                    write_log=True):
    if args.alg == "fedlwt":#仅用于fedlwt、llm_fedlwt
        local_models = args.sites.copy()
        server_model_name = [args.sites[args.global_center_idx]][0]
        local_models.pop(args.global_center_idx)
    else:
        local_models = args.sites.copy()
        server_model_name = args.llm

    avg_acc = 0.0
    acc_list = []
    auc = {}

    for idx, (net_id, net) in enumerate(nets.items()):
        # dataidxs = net_dataidx_map[net_id]
        # dataidxs=int(net_id)
        if write_log:
            logger.info("Training network %s. batch_id: %s" % (str(net_id), str(net_id)))
        print("Training network %s. batch_id: %s" % (str(net_id), str(net_id)))
        train_dl_local = train_dl[idx]
        if val_dl is not None:
            val_dl_local = val_dl[idx]
        else:
            val_dl_local = None
        test_dl_local = test_dl[idx]
        n_epoch = args.epochs

        if args.alg == 'fedlwt':
            #####迁移需要用的
            source_optimizer = source_optimizers_dict["%sto%s" % (server_model_name, net_id)]
            target_optimizer = target_optimizers_dict["%sto%s" % (server_model_name, net_id)]
            wnet = wnet_dict["%sto%s" % (server_model_name, net_id)]
            lwnet = lwnet_dict["%sto%s" % (server_model_name, net_id)]
            target_branch = target_branch_dict["%sto%s" % (server_model_name, net_id)]
            ##############
            prev_models = []
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            train_acc, val_acc, test_acc, train_auc, val_auc, test_auc = train_net_fedlwt(net_id, net,
                                                                                          source_optimizer,
                                                                                          target_optimizer,
                                                                                          wnet, lwnet, target_branch,
                                                                                          global_model, prev_models,
                                                                                          train_dl_local, val_dl_local,
                                                                                          test_dl_local,
                                                                                          n_epoch, args.lr,
                                                                                          args.optimizer, args.mu,
                                                                                          args.temperature, args, round,
                                                                                          device=device,
                                                                                          write_log=write_log)
            auc[net_id] = [train_auc, test_auc]
        elif args.alg == 'fedavg':
            train_acc, val_acc, test_acc, train_auc, val_auc, test_auc = \
                train_net(net_id, net, train_dl_local, val_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer,
                          args, device=device)
            auc[net_id] = [train_auc, test_auc]
        elif args.alg == 'fedprox':
            train_acc, val_acc, test_acc, train_auc, val_auc, test_auc = \
                train_net_fedprox(net_id, net, global_model, train_dl_local, val_dl_local, test_dl_local, n_epoch,
                                  args.lr, args.optimizer, args.mu, args, device=device)
            auc[net_id] = [train_auc, test_auc]
        elif args.alg == 'moon':
            prev_models = []
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            train_acc, val_acc, test_acc, train_auc, val_auc, test_auc = \
                train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local, val_dl_local, test_dl_local,
                                 n_epoch, args.lr, args.optimizer, args.mu, args.temperature, args, round,
                                 device=device)
            auc[net_id] = [train_auc, test_auc]
        elif args.alg == "HarmoFL":
            train_acc, val_acc, test_acc, train_auc, val_auc, test_auc = \
                train_net_HarmoFL(net_id, net, global_model, train_dl_local, val_dl_local, test_dl_local, n_epoch,
                                  args.lr, args.optimizer, args.mu, args, device=device)
        elif args.alg == 'llm_fedlwt':
            #####迁移需要用的
            source_optimizer = source_optimizers_dict["%sto%s" % (server_model_name, net_id)]
            target_optimizer = target_optimizers_dict["%sto%s" % (server_model_name, net_id)]
            net=net.cuda(0)
            wnet = wnet_dict["%sto%s" % (server_model_name, net_id)]
            lwnet = lwnet_dict["%sto%s" % (server_model_name, net_id)]
            target_branch = target_branch_dict["%sto%s" % (server_model_name, net_id)]
            ##############
            prev_models = []
            # for i in range(len(prev_model_pool)):
            #     prev_models.append(prev_model_pool[i][net_id])
            train_acc, val_acc, test_acc, train_auc, val_auc, test_auc = train_net_llm_fedlwt(net_id, net,
                                                                                          source_optimizer,
                                                                                          target_optimizer,
                                                                                          wnet, lwnet, target_branch,llm_model,
                                                                                          global_model, prev_models,
                                                                                          train_dl_local, val_dl_local,
                                                                                          test_dl_local,
                                                                                          n_epoch, args.lr,
                                                                                          args.optimizer, args.mu,
                                                                                          args.temperature, args, round,
                                                                                          device=device,
                                                                                          write_log=write_log)
            auc[net_id] = [train_auc, test_auc]
            
        if write_log:
            logger.info("net %s final test acc %f" % (net_id, test_acc))
        print("net %s final test acc %f" % (net_id, test_acc))
        avg_acc += test_acc
        acc_list.append(test_acc)
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))
    
    return nets, auc

def llm_init(llm_name):
    if llm_name.split("_")[0]=="dinov2":
        llm_model = torch.hub.load('/mnt/yxyxlab/qyj/Orthopedics/llm_FL/fed_llm', '%s_vitb14'%llm_name, source='local')
    else:
        pass
    return llm_model




if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)

    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    ###########################data loading

    sites, _, _, train_loaders, val_loaders, test_loaders, net_dataidx_map=recorded_llm_multicenters_dataloader(args)
    # print("len train_dl_global:", len(train_ds_global))
    ######分离全局模型数据和本地模型数据
    if args.alg == "fedlwt":  # n个本地模型，1个全局模型
        args.sites = sites
        train_dl_global = train_loaders[args.global_center_idx]
        train_loaders.pop(args.global_center_idx)
        if len(val_loaders) != 0:
            val_dl_global = val_loaders[args.global_center_idx]
            val_loaders.pop(args.global_center_idx)
        else:
            val_dl_global = None
            val_loaders = None
        test_dl_global = test_loaders[args.global_center_idx]
        test_loaders.pop(args.global_center_idx)
        ############################设置对比学习所需
        n_party_per_round = int(args.n_parties * args.sample_fraction)
        party_list = sites.copy()
        party_list.pop(args.global_center_idx)
    else:  # n+1个本地模型
        args.sites = sites
        if len(val_loaders) == 0:
            val_dl_global = None
            val_loaders = None
        n_party_per_round = int(args.n_parties * args.sample_fraction)
        party_list = sites
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    train_dl = None
    data_size = len(test_loaders[0])
    test_dl = None
    logger.info("Initializing nets")
    print("Initializing nets")
    # =================net_init=========================
    if args.alg == "fedlwt":  # n+1个本地模型，其他一个充当中心模型
        nets, local_model_meta_data, layer_type = init_nets(args, device='cpu')
        global_models, global_model_meta_data, global_layer_type = init_nets(args, device='cpu', server=True)
        global_model = global_models[args.sites[args.global_center_idx]]
    elif args.alg=="llm_fedlwt":
        nets, local_model_meta_data, layer_type = init_nets(args, device='cpu')
        # for i, net_id in enumerate(nets.keys()):
        #     print(net_id)
        llm_model=llm_init(args.llm)
        global_models, global_model_meta_data, global_layer_type = init_nets(args, device='cpu', server=True)
        global_model = global_models["server"]
    else:  # n个本地模型，1个全局模型
        nets, local_model_meta_data, layer_type = init_nets(args, device='cpu')
        global_models, global_model_meta_data, global_layer_type = init_nets(args, device='cpu', server=True)
        global_model = global_models["server"]

    # nets[]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round
    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0

    if args.alg == 'fedlwt':
        # ==================meta net init===================
        pairs = init_pairs(args)
        ###全局模型向本地模型迁移用的meta_model 4个,本地模型向全局模型迁移 4个
        wnet_dict, lwnet_dict, weight_params_dict, target_params_dict, target_branch_dict = init_meta_model(args, pairs,
                                                                                                            global_models,
                                                                                                            nets)  # nets or global_model
        nets[args.sites[args.global_center_idx]] = global_model
        source_optimizers_dict, target_optimizers_dict = optimizer_init(args, weight_params_dict, nets,
                                                                        target_params_dict, target_branch_dict)
        nets.pop(args.sites[args.global_center_idx])
        old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_' + 'net' + str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False

        for round in range(n_comm_rounds):
            time1 = time.time()
            logger.info("in comm round:" + str(round))
            print("in comm round:" + str(round))
            global_model.train()
            for param in global_model.parameters():
                param.requires_grad = True

            global_train_acc, global_val_acc, global_test_acc, global_train_auc, global_val_auc, global_test_auc = \
                global_train_net(args,
                                 source_optimizers_dict, target_optimizers_dict,
                                 wnet_dict, lwnet_dict, target_branch_dict, nets,
                                 global_model=global_model,
                                 train_global_dl=train_dl_global, val_global_dl=val_dl_global,
                                 test_global_dl=test_dl_global, device=device, write_log=True)
            # torch.cuda.empty_cache()
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()
            old_global_w = global_w
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            # 注释掉直接替换参数的迁移方式
            # for net in nets_this_round.values():
            #     net.load_state_dict(global_w)

            # torch.cuda.empty_cache()

            _, local_auc = local_train_net(nets_this_round, args,
                                           source_optimizers_dict=source_optimizers_dict,
                                           target_optimizers_dict=target_optimizers_dict,
                                           wnet_dict=wnet_dict, lwnet_dict=lwnet_dict,
                                           target_branch_dict=target_branch_dict,
                                           train_dl=train_loaders, val_dl=val_loaders, test_dl=test_loaders,
                                           global_model=global_model, prev_model_pool=old_nets_pool, round=round,
                                           device=device, write_log=True)
            # torch.cuda.empty_cache()

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1 - args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            # summary(global_model.to(device), (3, 32, 32))

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))
            print('global n_training: %d' % len(train_dl_global))
            print('global n_test: %d' % len(test_dl_global))
            global_model.cuda()
            train_acc, train_loss, train_auc = compute_accuracy(global_model, train_dl_global, device=device)
            if val_dl_global is not None:
                val_acc, conf_matrix_val, _, val_auc = compute_accuracy(global_model, val_dl_global,
                                                                        get_confusion_matrix=True,
                                                                        device=device)
            else:
                val_acc, val_auc = None, None
            test_acc, conf_matrix, _, test_auc = compute_accuracy(global_model, test_dl_global,
                                                                  get_confusion_matrix=True, device=device)
            global_model.to('cpu')
            logger.info('>> Global Model Train accuracy: {}----->>auc:{}'.format(train_acc, train_auc))
            logger.info('>> global_model_Val accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
            logger.info('>> Global Model Test accuracy: {}------->>auc:{}'.format(test_acc, test_auc))
            logger.info('>> Global Model Train loss: %f' % train_loss)
            print('>> Global Model Train accuracy: %f' % train_acc)
            print('>> Global Model val accuracy: {}'.format(val_acc))
            print('>> Global Model Test accuracy: %f' % test_acc)
            print('>> Global Model Train loss: %f' % train_loss)
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size - 2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i + 1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets
            if round % each_save_interval == 0:
                if args.save_model:
                    mkdirs(args.modeldir + '%s/' % args.alg)
                    torch.save(global_model.state_dict(),
                               args.modeldir + '%s/global_model_%d' % (args.alg, round) + args.log_file_name + '.pth')
                    for k, net_id in enumerate(nets):
                        torch.save(nets[net_id].state_dict(), args.modeldir + '%s/localmodel_%s_%d' % (
                        args.alg, net_id, round) + args.log_file_name + '.pth')
                    for k, net_id in enumerate(wnet_dict):
                        torch.save(wnet_dict[net_id].state_dict(), args.modeldir + '%s/wnet_model_%s_%d' % (args.alg,
                                                                                                            net_id,
                                                                                                            round) + args.log_file_name + '.pth')
                    for k, net_id in enumerate(lwnet_dict):
                        torch.save(lwnet_dict[net_id].state_dict(), args.modeldir + '%s/lwnet_model_%s_%d' % (args.alg,
                                                                                                              net_id,
                                                                                                              round) + args.log_file_name + '.pth')
                    for nets_id, old_nets in enumerate(old_nets_pool):
                        torch.save(
                            {'pool' + str(nets_id) + '_' + 'net' + str(net_id): net.state_dict() for net_id, net in
                             old_nets.items()},
                            args.modeldir + '%s/prev_model_pool_%d' % (args.alg, round) + args.log_file_name + '.pth')
            time2 = time.time()
            print("round:%d,consume:%s minutes" % (round, (time2 - time1) / 60.0))

    elif args.alg == 'moon':
        old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_' + 'net' + str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, train_dl=train_loaders, val_dl=val_loaders, test_dl=test_loaders,
                            global_model=global_model, prev_model_pool=old_nets_pool, round=round, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1 - args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            # summary(global_model.to(device), (3, 32, 32))

            # logger.info('global n_training: %d' % len(train_dl_global))
            # logger.info('global n_test: %d' % len(test_dl))
            # global_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            # test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            # global_model.to('cpu')
            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            # logger.info('>> Global Model Test accuracy: %f' % test_acc)
            # logger.info('>> Global Model Train loss: %f' % train_loss)

            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size - 2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i + 1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            if round % each_save_interval == 0:
                if args.save_model:
                    mkdirs(args.modeldir + '%s/' % args.alg)
                    torch.save(global_model.state_dict(),
                               args.modeldir + '%s/global_model_%d' % (args.alg, round) + args.log_file_name + '.pth')
                    for k, net_id in enumerate(nets):
                        torch.save(nets[net_id].state_dict(), args.modeldir + '%s/localmodel_%s_%d' % (
                        args.alg, net_id, round) + args.log_file_name + '.pth')
                    for nets_id, old_nets in enumerate(old_nets_pool):
                        torch.save(
                            {'pool' + str(nets_id) + '_' + 'net' + str(net_id): net.state_dict() for net_id, net in
                             old_nets.items()},
                            args.modeldir + '%s/prev_model_pool_%d' % (args.alg, round) + args.log_file_name + '.pth')

    elif args.alg == 'fedavg':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, train_dl=train_loaders, val_dl=val_loaders, test_dl=test_loaders,
                            device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1 - args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)

            # logger.info('global n_training: %d' % len(train_dl_global))
            # logger.info('global n_test: %d' % len(test_dl))
            # global_model.cuda()
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            # test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            #
            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            # logger.info('>> Global Model Test accuracy: %f' % test_acc)
            # logger.info('>> Global Model Train loss: %f' % train_loss)

            if round % each_save_interval == 0:
                if args.save_model:
                    mkdirs(args.modeldir + '%s/' % args.alg)
                    torch.save(global_model.state_dict(),
                               args.modeldir + '%s/global_model_%d' % (args.alg, round) + args.log_file_name + '.pth')
                    for k, net_id in enumerate(nets):
                        torch.save(nets[net_id].state_dict(), args.modeldir + '%s/localmodel_%s_%d' % (
                        args.alg, net_id, round) + args.log_file_name + '.pth')

    elif args.alg == 'fedprox':

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, train_dl=train_loaders, val_dl=val_loaders, test_dl=test_loaders,
                            global_model=global_model, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)

            mkdirs(args.modeldir + 'fedprox/')
            # global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir + 'fedprox/' + args.log_file_name + '.pth')
            if round % each_save_interval == 0:
                if args.save_model:
                    mkdirs(args.modeldir + '%s/' % args.alg)
                    torch.save(global_model.state_dict(),
                               args.modeldir + '%s/global_model_%d' % (args.alg, round) + args.log_file_name + '.pth')
                    for k, net_id in enumerate(nets):
                        torch.save(nets[net_id].state_dict(), args.modeldir + '%s/localmodel_%s_%d' % (
                        args.alg, net_id, round) + args.log_file_name + '.pth')

    elif args.alg == 'HarmoFL':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, train_dl=train_loaders, val_dl=val_loaders, test_dl=test_loaders,
                            global_model=global_model, device=device)
            global_model.to('cpu')

            # update global model
            # total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            # fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            client_num = len(party_list_this_round)
            fed_avg_freqs = [1. / client_num for i in range(client_num)]
            print(fed_avg_freqs)
            '''
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:

                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]'''
            for key in global_model.state_dict().keys():
                temp = torch.zeros_like(global_model.state_dict()[key])
                for net_id, net in enumerate(nets_this_round.values()):
                    temp += fed_avg_freqs[net_id] * net.state_dict()[key]
                global_model.state_dict()[key].data.copy_(temp)
                for net_id, net in enumerate(nets_this_round.values()):
                    net.state_dict()[key].data.copy_(global_model.state_dict()[key])
                if 'running_amp' in key:
                    # aggregate at first round only to save communication cost
                    global_model.amp_norm.fix_amp = True
                    for model in nets_this_round:
                        model.amp_norm.fix_amp = True

            # global_model.load_state_dict(global_w)
            mkdirs(args.modeldir + 'HarmoFL/')
            # global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir + 'HarmoFL/' + args.log_file_name + '.pth')
            if round % each_save_interval == 0:
                if args.save_model:
                    mkdirs(args.modeldir + '%s/' % args.alg)
                    torch.save(global_model.state_dict(),
                               args.modeldir + '%s/global_model_%d' % (args.alg, round) + args.log_file_name + '.pth')
                    for k, net_id in enumerate(nets):
                        torch.save(nets[net_id].state_dict(), args.modeldir + '%s/localmodel_%s_%d' % (
                            args.alg, net_id, round) + args.log_file_name + '.pth')

    elif args.alg == 'llm_fedlwt':
        # ==================meta net init===================
        pairs = init_pairs(args)
        ###全局模型向本地模型迁移用的meta_model 4个,本地模型向全局模型迁移 4个
        wnet_dict, lwnet_dict, weight_params_dict, target_params_dict, target_branch_dict = init_meta_model(args, pairs,
                                                                                                            global_model,
                                                                                                            nets)  # nets or global_model

        source_optimizers_dict, target_optimizers_dict = optimizer_init(args, weight_params_dict, nets,
                                                                        target_params_dict, target_branch_dict)

        old_nets_pool = []
        
        for round in range(n_comm_rounds):
            time1 = time.time()
            logger.info("in comm round:" + str(round))
            print("in comm round:" + str(round))

            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            
            _, local_auc = local_train_net(nets_this_round, args,
                                           source_optimizers_dict=source_optimizers_dict,
                                           target_optimizers_dict=target_optimizers_dict,
                                           wnet_dict=wnet_dict, lwnet_dict=lwnet_dict,
                                           target_branch_dict=target_branch_dict,
                                           train_dl=train_loaders, val_dl=val_loaders, test_dl=test_loaders,llm_model=llm_model,
                                           global_model=global_model, prev_model_pool=old_nets_pool, round=round,
                                           device=device, write_log=True)
            
            # client_num = len(party_list_this_round)
            # fed_avg_freqs = [1. / client_num for i in range(client_num)]
            
            # print(fed_avg_freqs)
            # for key in global_model.state_dict().keys():
            #     temp = torch.zeros_like(global_model.state_dict()[key],dtype=torch.float32)
            #     for net_id, net in enumerate(nets_this_round.values()):
            #         temp += fed_avg_freqs[net_id] * net.state_dict()[key]
            #     global_model.state_dict()[key].data.copy_(temp)
            
            sample_counts = {
                'CenterA(jm)': 1316,
                'CenterB(mm)': 84,
                'CenterC(bs)': 261,
                'CenterD(gz)': 179,
                'CenterE(zd)': 178
            }
            
            party_ids = party_list_this_round  # 当前这一轮参与的客户端列表
            total_samples = sum(sample_counts[k] for k in party_ids)
            fed_avg_freqs = [sample_counts[k] / total_samples for k in party_ids]

            # 执行样本加权聚合
            for key in global_model.state_dict().keys():
                temp = torch.zeros_like(global_model.state_dict()[key], dtype=torch.float32)
                for i, net_id in enumerate(party_ids):
                    net = nets[net_id]
                    temp += fed_avg_freqs[i] * net.state_dict()[key]
                global_model.state_dict()[key].data.copy_(temp)
            

            if round % each_save_interval == 0:
                if args.save_model:
                    mkdirs(args.modeldir + '%s/' % args.alg)
                    torch.save(global_model.state_dict(),
                               args.modeldir + '%s/global_model_%d' % (args.alg, round) + args.log_file_name + '.pth')
                    for k, net_id in enumerate(nets):
                        print(net_id,"model_saved")
                        torch.save(nets[net_id].state_dict(), args.modeldir + '%s/localmodel_%s_%d' % (
                        args.alg, net_id, round) + args.log_file_name + '.pth')
                    for k, net_id in enumerate(wnet_dict):
                        torch.save(wnet_dict[net_id].state_dict(), args.modeldir + '%s/wnet_model_%s_%d' % (args.alg,
                                                                                                            net_id,
                                                                                                            round) + args.log_file_name + '.pth')
                    for k, net_id in enumerate(lwnet_dict):
                        torch.save(lwnet_dict[net_id].state_dict(), args.modeldir + '%s/lwnet_model_%s_%d' % (args.alg,
                                                                                                              net_id,
                                                                                                              round) + args.log_file_name + '.pth')
                    
            time2 = time.time()
            print("round:%d,consume:%s minutes" % (round, (time2 - time1) / 60.0))


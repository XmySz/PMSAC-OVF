import os
import logging
import numpy as np
import cv2
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch

import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix
import torch.optim as optim
from model import *
#from typicl_model import *
from datasets import ImageFolder_custom
from utils import roc_auc_score
import scipy.io as sio
import matplotlib.pyplot as plt
from shutil import copyfile,copytree,copy,move
from l2t_ww.check_model import check_model
import argparse
import xlwt,xlrd
from PIL import Image

class DataSet(data.Dataset):
    def __init__(self, root=None,
                 data_set="train_data",center=""):
        self.root = root
        self.files = []
        self.patient_img_num=[]
        img_num=0

        for leison_class in os.listdir(os.path.join(self.root,center,data_set)):
            for patient in os.listdir(os.path.join(self.root,center,data_set,leison_class)):
            # for split in ["train", "trainval", "val"]:
                img_num=len(os.listdir(os.path.join(
                            self.root, center, data_set, leison_class, patient)))
                patient_file=os.path.join(
                            self.root, center, data_set, leison_class, patient)
                self.files.append({
                    "img": patient_file,
                    "label": 1 if leison_class in ["肺腺癌","复发","1"] else 0,
                    "center": center,
                    "leison_class": leison_class,
                    # "label": label_file,
                    "patient_name": patient,
                    "img_num": img_num
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        patient_name = datafiles["patient_name"]
        # img_name = datafiles["img_name"]
        center_name = datafiles["center"]
        leisonclass = datafiles["leison_class"]
        label = datafiles["label"]
        patient_img=[]
        patient_label=[]
        for img in os.listdir(datafiles["img"]):
            img_path=os.path.join(datafiles["img"],img)
            image = Image.open(img_path).convert('RGB')
            image=image.resize((224,224))
            image = np.asarray(image, np.float32)
            image = image.transpose((2, 0, 1))
            if datafiles["img_num"]==1:
                patient_img.append(image)
                patient_label.append(label)
            patient_img.append(image)
            patient_label.append(label)
        # print(patient_name)
        patient_img=np.array(patient_img)
        patient_label=np.array(patient_label)
        size = patient_img.shape
        return patient_img.copy(), np.array(size), patient_name,center_name,leisonclass,label



def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False,compute_auc=True):
    was_training = False
    label_list=[]
    pred_list=[]
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    avg_auc=0
    if "cuda" == device:
        # print("cuda")
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    loss_collector = []
    # criterion=nn.CrossEntropyLoss()
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    #print("x:",x)
                    #print("target:",target)
                    if device != 'cpu':
                        x, target = x.cuda(), target.to(dtype=torch.int64).cuda().long()
                    _, _, out = model(x)
                    if len(target)==1:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.data.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                #print("x:",x)
                if device != 'cpu':
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda().long()
                _,_,out = model(x)
                # print(out,target)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                # print(out.data.cpu().numpy())
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
                #auc
                label_list.extend(target.cpu().numpy().tolist())

                pred_list.extend(out.data.cpu().numpy()[:,1])
                # batch_auc = roc_auc_score(target.cpu().numpy(), pred_label.cpu().numpy())

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)
            if compute_auc:
                avg_auc=roc_auc_score(label_list,pred_list)
            # print("auc：",avg_auc)
    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix and compute_auc:
        return correct / float(total), conf_matrix, avg_loss,avg_auc
    elif get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss
    if compute_auc:
        return correct / float(total), avg_loss,avg_auc
    return correct / float(total), avg_loss


def feature_hstack(features):
    all_feature=None

    for i,feature in enumerate(features):
        feature = feature.data.cpu().numpy()
        size_list = list(feature.shape)

        feature_temp = np.zeros((size_list[0], size_list[1]), dtype="float32")

        for k in range(size_list[1]):
            for j in range(size_list[0]):
                if i == 5:
                    feature_temp[j, k] = feature[j, k]
                else:
                    feature_temp[j, k] = np.mean(np.squeeze(feature[j, k, :, :]))

        if i==0:
            all_feature=feature_temp
        else:
            all_feature=np.hstack([all_feature,feature_temp])
    mean_features = np.mean(all_feature, axis=0)
    return mean_features
if __name__ == '__main__':
    # RANDOM_SEED = 41
    data_total_path=r"/home/lsl/python_project/fed_llm/data/Endometrial_Cancer_增强"
    model_total_path = \
    r"/home/lsl/python_project/fed_llm/model_save/dinov2_2024-01-13_llm_fedlwt_resnet18_Endometrial_Cancer_nofreeze/models_auc_ratio/llm_fedlwt"
    model_time = "experiment_log-2024-01-13-2255-14.pth"
    #model_time = ".pth"
    epochs = 15
    save_path = r"/home/lsl/python_project/fed_llm/extract_feature_save_for_name"

    args = argparse.ArgumentParser()
    args.dataset='Endometrial_Cancer'
    args.num_classes=2
    args.target_mhsa=False
    args.dropout_p = 0.0
    args.input_shape=224
    args.open_perturbe =False
    args.model='resnet18'

    #for epoch in range(0,epochs,10):
    for epoch in range(epochs, epochs+1):
        device="cuda"
        wb = xlwt.Workbook(encoding='utf-8')
        # data_source=["江门医院","广东省医","湛江医院","中大五院","中山肿瘤"]
        data_source=["江门","东莞", "开平", "粤北"]
        for data_name in data_source:
            print("=======================",data_name,"model predicting","=====================")
            data_path=os.path.join(data_total_path)
            save_path1=os.path.join(save_path,model_total_path.split("/")[-3]
                                           ,"epoch_"+str(epoch))
            sheet = wb.add_sheet(data_name, cell_overwrite_ok=False)
            if os.path.exists(save_path1) is not True:
                os.makedirs(save_path1)
            #feature_save_path=os.path.join(save_path1,data_name+"_predict_茂名"+".mat")
            feature_save_path=os.path.join(save_path1,data_name+".mat")
            if data_name=="江门医院":
                data_name1="global_model"
                model_name="global_model"+"_%d%s"%(epoch,model_time)

            else:
                # data_name1 = "global_model"
                # model_name = "global_model" +"_%d%s"%(epoch*10,model_time)
                model_name = "localmodel_" + data_name+"_%d%s"%(epoch,model_time)
                #model_name = "_%dlocalmodel_"%epoch + data_name+"_.pth"
            # print(model_name)
            trained_model_path = os.path.join(model_total_path, model_name)
            print(trained_model_path)
            # net = ModelFedCon_noheader('resnet18',out_dim=256, n_classes=2)

            net = check_model(args).to(device)
            model_dict=torch.load(trained_model_path)
            net.load_state_dict(model_dict)
            #net=torch.load(trained_model_path)
            net.cuda()
            criterion = nn.CrossEntropyLoss()
            #=============
            
            train_dataset=DataSet(data_path,"traindata",data_name)
            test_dataset=DataSet(data_path,"testdata",data_name)
            #test_dataset=DataSet(data_path,"testdata","茂名")
            train_DL = data.DataLoader(dataset=train_dataset, batch_size=1, drop_last=False, shuffle=False,
                                       num_workers=6)
            test_DL = data.DataLoader(dataset=test_dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=6)
            #=============

            with torch.no_grad():
                # ============train==========
                train_epoch_Conv_feature_collector = []

                train_epoch_pred_collector = []
                train_label = []
                net.eval()
                print(epoch,data_name,"===============train_data_predicting=============")
                for batch_idx, (imgs,_,patient_name,center,leison_name,patient_label) in enumerate(train_DL):
                    #print(data_name,"train_data",patient_name)
                    # print("x:",x)
                    # tn=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                    # x=tn.__call__(torch.squeeze(x))
                    sheet.write(batch_idx, 0, patient_name[0])
                    sheet.write(batch_idx, 1, "train_data")
                    sheet.write(batch_idx, 2, str(patient_label.numpy()[0]))
                    imgs=imgs.squeeze()
                    # x=x.unsqueeze(dim=0)
                    if device != 'cpu':

                        x, target = imgs.cuda(), patient_label.to(dtype=torch.int64).cuda().long()
                        out,feature = net(x/torch.tensor(255.0))
                        # print(out,target)
                        _, pred_label = torch.max(out.data, 0)
                        mean_pred_value=np.mean(out.data.cpu().numpy()[:,1])

                        mean_feature=feature_hstack(feature)

                    else:
                        out,feature = net(x/torch.tensor(255.0))
                        # print(out,target)
                        _, pred_label = torch.max(out.data, 0)
                        mean_pred_value=np.mean(out.data.numpy()[:,1])
                        mean_feature=feature_hstack(feature)


                    train_epoch_pred_collector.append(mean_pred_value)
                    train_label.append(patient_label.data.numpy())
                    # print(patient_label.data.numpy())
                    train_epoch_Conv_feature_collector.append(mean_feature)
                plus_index = len(train_DL)


                #============test==========
                test_epoch_Conv_feature_collector = []

                test_epoch_pred_collector = []
                test_label = []
                net.eval()
                print(epoch,data_name,"=============test_data_predicting==============")
                for batch_idx, (imgs, _, patient_name, center, leison_name, patient_label) in enumerate(test_DL):
                    sheet.write(batch_idx+plus_index, 0, patient_name[0])
                    sheet.write(batch_idx+plus_index, 1, "test_data")
                    sheet.write(batch_idx+plus_index, 2, str(patient_label.numpy()[0]))
                    imgs = torch.squeeze(imgs)
                    if device != 'cpu':
                        x, target = imgs.cuda(), patient_label.to(dtype=torch.int64).cuda().long()
                        out,feature = net(x/torch.tensor(255.0))
                        # print(out,target)
                        _, pred_label = torch.max(out.data, 0)
                        mean_pred_value = np.mean(out.data.cpu().numpy()[:, 1])
                        mean_feature=feature_hstack(feature)

                    else:
                        out,feature = net(x/torch.tensor(255.0))
                        # print(out,target)
                        _, pred_label = torch.max(out.data, 0)
                        mean_pred_value = np.mean(out.data.numpy()[:, 1])
                        mean_feature=feature_hstack(feature)


                    test_epoch_pred_collector.append(mean_pred_value)
                    test_label.append(patient_label.data.numpy())
                    test_epoch_Conv_feature_collector.append(mean_feature)

            wb.save(os.path.join(save_path1, "name_all.xls"))

            sio.savemat(feature_save_path,{"train_data":train_epoch_Conv_feature_collector,"train_label":train_label,
                                          "test_data":test_epoch_Conv_feature_collector,"test_label":test_label})
                                            #"train_res18_pred":train_epoch_pred_collector,
                                            #"test_res18_pred":test_epoch_pred_collector})



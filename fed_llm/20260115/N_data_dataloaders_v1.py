'''

'''
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
#import SimpleITK as sitk
import random
import cv2
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.datasets import ImageFolder, DatasetFolder, CIFAR10, CIFAR100
import math
import logging
import shutil
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from torch.utils import data
from torchvision import transforms

class llm_med_DataSet(data.Dataset):
    def __init__(self, root=None,
                 data_set="",center="",transform=None,patch_h=112,patch_w=112):
        self.root = root
        self.files = []
        self.patient_img_num=[]
        self.transform=transform
        self.transform1 = transforms.Compose([
            # transforms.GaussianBlur(9, sigma=(0.1, 2.0)),
            transforms.Resize((patch_h * 14, patch_w * 14)),
            transforms.CenterCrop((patch_h * 14, patch_w * 14)),
            transforms.ToTensor(),
            # T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        leison1_list=["1"]#LABEL
        for leison_class in os.listdir(os.path.join(self.root,center,data_set)):
            for patient in os.listdir(os.path.join(self.root,center,data_set,leison_class)):
            # for split in ["train", "trainval", "val"]:
                for img_name in os.listdir(os.path.join(self.root,center,data_set,leison_class,patient)):
                    if ".mat" in img_name:
                        continue
                    img_path=os.path.join(
                                self.root, center, data_set, leison_class, patient,img_name)
                    self.files.append({
                        "img_path": img_path,
                        "label": 1 if leison_class in leison1_list else 0,
                        "center": center,
                        "leison_class": leison_class,
                        "patient_name": patient})
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        label = datafiles["label"]
        image = Image.open(datafiles["img_path"]).convert('RGB')
        image=image.resize((224,224))
        llm_image=self.transform1(image)
        if self.transform is not None:
            image = self.transform(image)

        return image,llm_image,label


def recorded_llm_dataloader(args):
    
    private_data = ["osteo"]
    
    val_dl_local = []

    if args.dataset in private_data:
        if args.dataset == "osteo":
            train_set = llm_med_DataSet(root=args.datadir, center="DWI_data", data_set="train_data", patch_h=64,
                                        patch_w=64, transform=transforms.ToTensor())
            test_set = llm_med_DataSet(root=args.datadir, center="DWI_data", data_set="test_data", patch_h=64,
                                        patch_w=64, transform=transforms.ToTensor())
            train_dl_local = data.DataLoader(train_set, batch_size=args.batch_size,drop_last=False, shuffle=True,
                                             num_workers=args.num_workers, pin_memory=True)
            test_dl_local = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True)

    return train_dl_local, val_dl_local, test_dl_local



class llm_med_DataSet_v2(data.Dataset):
    def __init__(self,
                 txt_path,
                 transform=None,
                 patch_h=112,
                 patch_w=112):

        self.samples = []   # 每一项是一张 image
        self.transform = transform

        self.transform1 = transforms.Compose([
            transforms.Resize((patch_h * 14, patch_w * 14)),
            transforms.CenterCrop((patch_h * 14, patch_w * 14)),
            transforms.ToTensor(),
        ])

        assert os.path.exists(txt_path), f"txt not found: {txt_path}"

        # --------
        # 读 txt
        # --------
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            case_dir, label = line.split("\t")
            label = int(label)

            if not os.path.isdir(case_dir):
                continue

            # 遍历该病例下所有 png
            for fname in os.listdir(case_dir):
                if not fname.lower().endswith(".png"):
                    continue
                if fname.endswith(".mat"):
                    continue

                img_path = os.path.join(case_dir, fname)
                self.samples.append({
                    "img_path": img_path,
                    "label": label,
                    "case_dir": case_dir
                })

        if len(self.samples) == 0:
            raise RuntimeError(f"No png images found using txt: {txt_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]

        image = Image.open(item["img_path"]).convert("RGB")
        image = image.resize((224, 224))

        llm_image = self.transform1(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, llm_image, item["label"]



def recorded_llm_dataloader_v2(args):

    val_dl_local = []

    train_set = llm_med_DataSet_v2(
        txt_path=args.train_txt,
        patch_h=64,
        patch_w=64,
        transform=transforms.ToTensor()
    )
    test_set = llm_med_DataSet_v2(
        txt_path=args.test_txt,
        patch_h=64,
        patch_w=64,
        transform=transforms.ToTensor()
    )

    train_dl_local = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_dl_local = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # -------- 外部验证：多个 dataloader --------
    external_dl_dict = {}
    if hasattr(args, "external_txts") and args.external_txts:
        for name, txt_path in args.external_txts.items():
            ext_set = llm_med_DataSet_v2(
                txt_path=txt_path,
                patch_h=64,
                patch_w=64,
                transform=transforms.ToTensor()
            )
            external_dl_dict[name] = data.DataLoader(
                ext_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )

    return train_dl_local, val_dl_local, test_dl_local, external_dl_dict



def recorded_llm_multicenters_dataloader(args):
    private_data = ["Orthopedics"]
    train_loaders, test_loaders = [], []
    val_loaders = []
    trainsets, testsets = [], []
    valsets = []
    sites = []
    if args.dataset in private_data:
        if args.dataset == "Orthopedics":
            net_dataidx_map = {}
            sites = ["CenterA(jm)", "CenterB(mm)", "CenterC(bs)", "CenterD(gz)", "CenterE(zd)"]
            for site in sites:
                train_txt_path = os.path.join(args.datadir, "{}_train.txt".format(site))
                val_txt_path = os.path.join(args.datadir, "{}_val.txt".format(site))
                test_txt_path = os.path.join(args.datadir, "{}_test.txt".format(site))
                train_set = llm_med_DataSet_v2(txt_path=train_txt_path, patch_h=64, patch_w=64, transform=transforms.ToTensor())
                val_set = llm_med_DataSet_v2(txt_path=val_txt_path, patch_h=64, patch_w=64, transform=transforms.ToTensor())
                test_set = llm_med_DataSet_v2(txt_path=test_txt_path, patch_h=64, patch_w=64, transform=transforms.ToTensor())
                
                train_dl_local = data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,num_workers=4,pin_memory=True)
                val_dl_local = data.DataLoader(val_set, batch_size=args.batch_size, drop_last=False, shuffle=False,num_workers=4,pin_memory=True)
                test_dl_local = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,num_workers=4,pin_memory=True)
                
                train_loaders.append(train_dl_local)
                val_loaders.append(val_dl_local)
                test_loaders.append(test_dl_local)
                net_dataidx_map[site] = train_set
                
    return sites, trainsets, testsets, train_loaders, val_loaders, test_loaders, net_dataidx_map




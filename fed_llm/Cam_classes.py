import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import cv2
import torch.nn as nn
from l2t_ww.models import resnet_ilsvrc
def model_init(ModelPath):

    #model = models.resnet18(pretrained=False)

    ResNet = resnet_ilsvrc.__dict__["resnet18"]  # (pretrained=True)
    model = ResNet(num_classes=2, mhsa=False, dropout_p=0,
                 resolution=(224, 224), open_perturbe=False)
    model.load_state_dict(torch.load(ModelPath))

    model.fc = nn.Linear(in_features=512, out_features=2, bias=True)

    #print('Model use:->> ResNet18')

    #将训练好的参数导入
    checkpoint = torch.load(ModelPath, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    return model


def hook_feature(module, input, output):
    # 512*7*7 这个是展平了。
    features_blobs.append(output.data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:

        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def conv_heatmap_onebyone(model, layer_name, input_image, img_path, class_id,dstpath):
    object_output = model.output[:, class_id]  # 预测向量中class_id索引下的类别元素
    try:  # try：想要执行的内容。except：try里的命令出现异常则执行except中的命令
        last_conv_layer = model.get_layer(layer_name)  # 目标layer的信息

    except:
        raise Exception('Not layer named {}!'.format(layer_name))

# normalize = transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # normalize
])


if __name__ == '__main__':

    ModelRoot = r'/home/lsl/python_project/fed_llm/model_save/本方法存/dinov2_2023-11-29_llm_fedlwt_resnet18_Endometrial_Cancer/models_auc_ratio/llm_fedlwt'
    model_train_setting="experiment_log-2023-11-29-1852-54"
    PictureRoot = r'/home/lsl/python_project/fed_llm/data/Endometrial_Cancer_增强'

    CenterName = ['江门','东莞',  '开平', '粤北']
    output_path = r"/home/lsl/python_project/fed_llm/Cam_visualize/EC_contribution"

    PitureResize = True
    finalconv_name = 'layer4'
    classes = {0: "0", 1: "1"}

    for i,Center in enumerate(CenterName):
        if Center=="江门医院":
            Center1="global_model"
        else:
            Center1='localmodel_'+Center
        ModelPath = os.path.join(ModelRoot,Center1+'_15'+model_train_setting+'.pth')
        net = model_init(ModelPath)
        #print(net)
        
        net._modules.get(finalconv_name).register_forward_hook(hook_feature)
        print(net)



        for TrainOrTest in os.listdir(os.path.join(PictureRoot, Center)):
            for classer in os.listdir(os.path.join(PictureRoot, Center, TrainOrTest)): #
                for patient in os.listdir(os.path.join(PictureRoot, Center, TrainOrTest,classer)):
                    for image in os.listdir(os.path.join(PictureRoot, Center, TrainOrTest,classer,patient)):
                        #图像加载
                        features_blobs = []
                        params = list(net.parameters())
                        weight_softmax = np.squeeze(params[-2].data.numpy())
                        imagePath = os.path.join(PictureRoot, Center, TrainOrTest,classer,patient,image)
                        img_pil = Image.open(imagePath).convert('RGB')
                        img_tensor = preprocess(img_pil)
                        img_variable = Variable(img_tensor.unsqueeze(0))

                        logit, aux = net(img_variable)
                        h_x = F.softmax(logit, dim=1).data.squeeze()
                        probs, idx = h_x.sort(0, True)
                        probs = probs.numpy()
                        idx = idx.numpy()
                        for i in range(0, 2):
                            print('{}->{}:{:.3f} -> {}'.format(Center,patient, probs[i], classes[idx[i]]))
                        '''
                        if classer == '1':
                            idx[0] = 1
                        else:
                            idx[0] = 0
                        '''
                        CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
                        print('output CAM.jpg for the top1 prediction: %s label: %s' % (classes[idx[0]], classer))

                        img = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), 1)
                        height, width, _ = img.shape
                        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
                        result = heatmap * 0.5 + img * 0.5

                        if PitureResize == True:
                            result = cv2.resize(result,(224,224))
                            heatmap = cv2.resize(heatmap,(224,224))

                        dst_path = os.path.join(output_path,Center,TrainOrTest,classer,patient)
                        #k = os.path.join(dst_path,'CAM.jpg')

                        os.makedirs(dst_path,exist_ok=True)

                        cv2.imencode('_{}_heatmap_.jpg'.format(image.split(".")[0]), heatmap)[1].tofile(
                            '%s/' % (dst_path) + '_{}_heatmap_.jpg'.format(image.split(".")[0]))
                        # cv2.imencode('_{}_concentration_.jpg'.format(image.split(".")[0]), result)
                        cv2.imencode('_{}_Cam_.jpg'.format(image.split(".")[0]), result)[1].tofile(
                            '%s/' % (dst_path) + '_{}_Cam_.jpg'.format(image.split(".")[0]))
                        cv2.imencode('_{}_leison_.jpg'.format(image.split(".")[0]), img)[1].tofile(
                            '%s/' % (dst_path) + '_{}_leison_.jpg'.format(image.split(".")[0]))

















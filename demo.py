import matplotlib 
matplotlib.use('Agg')
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans, KMeans

from torchvision import transforms
from tqdm import tqdm
import json
import os
import argparse
import numpy as np 

from datasets.asl_dataset import TrainDataset, TestDataset
from datasets.asl_dataset import make_train_test_split
import config
from models.resnet import resnet50, resnet18 
from models.densenet import densenet121
from sklearn.manifold import TSNE
from utils import plot_embedding_with_image, plot_embedding_with_label, plot_embedding_with_circle
import cv2
from PIL import Image

def infer(model:nn.Sequential, image_path, num_classes, device:int, args):
    lookup = {}
    path_dir = './original_images/original_images'
    idx = 0
    for classname in os.listdir(path_dir):
        lookup[idx] = classname
        idx += 1
    model.eval()
    transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.CenterCrop((config.img_w, config.img_h)), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    # 不需要反向传播，关闭求导
    with torch.no_grad():
        image_ori = cv2.imread(image_path)
        image = image_ori[:, :, ::-1]
        image = Image.fromarray(image)
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        image = image.to(device)
        y_, feature = model(image)
        # 收集prediction和ground truth
        y_ = y_.argmax(dim=1).cpu().numpy()[0]
        class_name = lookup[y_]
        save_dir = args.output_dir
        cv2.putText(image_ori, class_name, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(save_dir, 'demo.jpg'), image_ori)
def parse_args():
    parser = argparse.ArgumentParser(usage='python3 train.py -i path/to/data -r path/to/checkpoint')
    parser.add_argument('-i', '--image_path', help='path to your datasets', default='./original_images/original_images/1/d0.jpg')
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint', default=None)
    parser.add_argument('-v', '--visualization', help='whether to generate clustering visualization', action='store_true')
    parser.add_argument('-o', '--output_dir', help='The dir to save the demo file', default='./demo_output')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    image_path = args.image_path
    restore_from = args.restore_from
    base_model = config.model_name
    num_classes = config.num_classes
    output_dir =  args.output_dir 
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 配置训练时环境
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # 实例化计算图模型
    if base_model == 'resnet50':
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
    elif base_model == 'densenet':
        model = densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, num_classes)
    elif base_model =='resnet18':
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(512, num_classes)
    model.to(device)

    ckpt = {}
    # 从断点继续训练
    if restore_from is not None:
        ckpt = torch.load(restore_from)
        model.load_state_dict(ckpt['model_state_dict'])
        print('Model is loaded from %s' % (restore_from))
    infer(model, image_path, num_classes, device, args)


    print('*********** Inference Finished **************')
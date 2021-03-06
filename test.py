import matplotlib 
matplotlib.use('Agg')
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans, KMeans

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
from utils import plot_embedding_with_image, plot_embedding_with_label, plot_embedding_with_circle, plot_multi_label_circle

def test(model:nn.Sequential, test_loader:torch.utils.data.DataLoader, num_classes, device:int, args):
    model.eval()

    print('Size of Test Set: ', len(test_loader.dataset))

    # 准备在测试集上验证模型性能
    test_loss = 0
    X_in = []
    y_gd = []
    y_pred = []
    features = []
    # 不需要反向传播，关闭求导
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc='Validating'):
            # 对测试集中的数据进行预测
            X, y = X.to(device), y.to(device)
            y_, feature = model(X)

            # 计算loss
            loss = F.cross_entropy(y_, y, reduction='sum')
            test_loss += loss.item()

            # 收集prediction和ground truth
            y_ = y_.argmax(dim=1)
            features.append(np.squeeze(feature.cpu().numpy(), 0))
            X_in.append(np.transpose(np.squeeze(X.cpu().numpy(), 0), (1,2,0)))
            y_gd += y.cpu().numpy().tolist()
            y_pred += y_.cpu().numpy().tolist()
    features = np.array(features)
    
    # 计算loss
    test_loss /= len(test_loader)
    # 计算正确率
    test_acc = accuracy_score(y_gd, y_pred)

    test_info_score = adjusted_mutual_info_score(y_gd, y_pred)

    test_rand_score = adjusted_rand_score(y_gd, y_pred)
    
    info_rand_average = 2 / ((1 / test_info_score) + (1 / test_rand_score))

    print('Test avg loss: %0.4f, acc: %0.2f, matual_info_score: %0.2f, rand_score: %0.2f, info_rand_average:%0.2f\n' % (test_loss, test_acc, test_info_score, test_rand_score, info_rand_average))

    # 利用tSNE算法可视化聚类结果
    if args.visualization:
        tsne = TSNE(n_components=2, init='pca')
        
        # kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
        epoch = args.restore_from.split('/')[-1].split('.')[0]
        embedding = tsne.fit_transform(features) # (len(test_loader), 2)
        # plot_embedding_with_label(embedding, y_gd, epoch, num_classes, 't-SNE visualization')
        # plot_embedding_with_image(embedding, X_in, epoch, num_classes, 't-SNE visualization')
        # plot_embedding_with_circle(embedding, y_gd, epoch, num_classes, 't-SNE visualization')
        plot_multi_label_circle(embedding, y_gd, epoch, num_classes, 't-SNE visualization')
    return test_loss, test_acc

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 train.py -i path/to/data -r path/to/checkpoint')
    parser.add_argument('-i', '--data_path', help='path to your datasets', default='E:\研2018-代码合集\王恒毅\实验数据集\ASL\\archive\dataset5\D\\a//color_0_0002.png')
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint', default='E:\研2018-代码合集\王恒毅\训练模型\0226_asl_lr3e-7_64\0226_asl_lr3e-7_64\ep-39.pth')
    parser.add_argument('-v', '--visualization', help='whether to generate clustering visualization', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    restore_from = args.restore_from
    base_model = config.model_name
    num_classes = config.num_classes
    # data_split = '/'.join(restore_from.split('/')[:-1]) + '/data_info.json'
    data_split = 'E:\研2018-代码合集\王恒毅\训练模型\0226_asl_lr3e-7_64\0226_asl_lr3e-7_64\ep-39.pth\data_info.json'
    
    # 划分训练集和测试集
    with open(data_split, 'r') as f:
        data_info = json.load(f)
        x_train,x_test,y_train,y_test = data_info['x_train'], data_info['x_test'], data_info['y_train'], data_info['y_test']


    # x_test = x_test[:1000]
    # y_test = y_test[:1000]
    # 准备数据加载器
    test_loader = DataLoader(TestDataset(x_test, y_test), 1, shuffle=False)
    
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
    test_loss, test_score = test(model, test_loader, num_classes, device, args)


    print('*********** Testing Finished **************')
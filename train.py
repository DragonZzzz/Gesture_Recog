import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json
import os
import argparse
from datasets.asl_dataset import TrainDataset, TestDataset
from datasets.asl_dataset import make_train_test_split

from models.resnet import resnet50, resnet18 
from models.densenet import densenet121
from utils import plot_acc_loss
import config
    
def train(model:nn.Sequential, dataloader:torch.utils.data.DataLoader, optimizer:torch.optim.Optimizer, epoch, device):
    model.train()

    print('Size of Training Set: ', len(dataloader.dataset))

    for i, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # 初始化优化器参数
        optimizer.zero_grad()
        # 执行前向传播
        y_, _ = model(X)

        # 计算loss
        loss = F.cross_entropy(y_, y)
        # 反向传播梯度
        loss.backward()
        optimizer.step()

        y_ = y_.argmax(dim=1)
        acc = accuracy_score(y_.cpu().numpy(), y.cpu().numpy())


        if (i + 1) % config.log_interval == 0:
            print('[Epoch %3d]Training %3d of %3d: acc = %.2f, loss = %.2f' % (epoch, i + 1, len(dataloader), acc, loss.item()))
        # 保存loss等信息
        train_losses = loss.item()
        train_scores = acc

    return train_losses, train_scores

def validation(model:nn.Sequential, test_loader:torch.utils.data.DataLoader, optimizer:torch.optim.Optimizer, epoch:int, device:int):
    model.eval()

    print('Size of Test Set: ', len(test_loader.dataset))

    # 准备在测试集上验证模型性能
    test_loss = 0
    y_gd = []
    y_pred = []

    # 不需要反向传播，关闭求导
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc='Validating'):
            # 对测试集中的数据进行预测
            X, y = X.to(device), y.to(device)
            y_, _ = model(X)

            # 计算loss
            loss = F.cross_entropy(y_, y, reduction='sum')
            test_loss += loss.item()

            # 收集prediction和ground truth
            y_ = y_.argmax(dim=1)
            y_gd += y.cpu().numpy().tolist()
            y_pred += y_.cpu().numpy().tolist()

    # 计算loss
    test_loss /= len(test_loader)
    # 计算正确率
    test_acc = accuracy_score(y_gd, y_pred)

    print('[Epoch %3d]Test avg loss: %0.4f, acc: %0.2f\n' % (epoch, test_loss, test_acc))

    return test_loss, test_acc

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 train.py -i path/to/data -r path/to/checkpoint')
    parser.add_argument('-i', '--data_path', help='path to your datasets', default='./dataset5')
    parser.add_argument('-d', '--data_file_path', help='path to your datasets split', default=None)
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint', default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    restore_from = args.restore_from
    base_model = config.model_name
    num_classes = config.num_classes    
    data_split = args.data_file_path


    # 划分训练集和测试集
    # 若未事先划分好，则本次划分，并在最后保存划分结果用于复现。
    if not data_split:
        data_info = {}
        x_train,x_test,y_train,y_test = make_train_test_split(data_path)
        data_info['x_train'] = x_train
        data_info['x_test'] = x_test
        data_info['y_train'] = y_train
        data_info['y_test'] = y_test 
    # 加载划分结果
    else:
        with open(data_split, 'r') as f:
            data_info = json.load(f)
        x_train,x_test,y_train,y_test = data_info['x_train'], data_info['x_test'], data_info['y_train'], data_info['y_test']

    # 准备数据加载器
    train_loader = DataLoader(TrainDataset(x_train, y_train), config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(TestDataset(x_test, y_test), config.batch_size, shuffle=False)
    
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

    # 多GPU训练
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print('使用{}个GPU训练'.format(device_count))
        model = nn.DataParallel(model)

    ckpt = {}
    # 从断点继续训练
    if restore_from is not None:
        ckpt = torch.load(restore_from)
        model.load_state_dict(ckpt['model_state_dict'])
        print('Model is loaded from %s' % (restore_from))

    # 提取网络参数，准备进行训练
    model_params = model.parameters()

    # 设定优化器
    optimizer = torch.optim.Adam(model_params, lr=config.learning_rate)

    if restore_from is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # 训练时数据
    info = {
        'train_losses': [],
        'train_scores': [],
        'test_losses': [],
        'test_scores': []
    }

    start_ep = ckpt['epoch'] + 1 if 'epoch' in ckpt else 0

    save_path = './checkpoints'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 开始训练
    for ep in range(start_ep, config.epoches):
        train_losses, train_scores = train(model, train_loader, optimizer, ep, device)
        test_loss, test_score = validation(model, test_loader, optimizer, ep, device)

        # 保存信息
        info['train_losses'].append(train_losses)
        info['train_scores'].append(train_scores)
        info['test_losses'].append(test_loss)
        info['test_scores'].append(test_score)

        # 保存模型
        sub_save_path = os.path.join(save_path, config.experiments_name)
        if not os.path.exists(sub_save_path):
            os.mkdir(sub_save_path)
        ckpt_path = os.path.join(sub_save_path, 'ep-%d.pth' % ep)
        if (ep + 1) % config.save_interval == 0:
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, ckpt_path)
            print('Model of Epoch %3d has been saved to: %s' % (ep, ckpt_path))

    with open(os.path.join(sub_save_path, './train_info.json'), 'w') as f:
        json.dump(info, f)
    

    # 保存本次训练的损失函数和准确度曲线到模型对应对应的文件夹
    plot_acc_loss(sub_save_path, info['test_losses'], info['test_scores'])

    # 保存本次生成的数据划分
    if not data_split:
        with open(os.path.join(sub_save_path, './data_info.json'), 'w') as f:
            json.dump(data_info, f)
    
    print('*********** Training Finished **************')
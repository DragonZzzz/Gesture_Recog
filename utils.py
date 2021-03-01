import numpy as np 
from PIL import Image
import matplotlib 
matplotlib.use('Agg')
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import host_subplot
import json
import os 

def plot_acc_loss(path_prefix, loss, acc):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    par1 = host.twinx()   # 共享x轴
 
    # set labels
    host.set_xlabel("steps")
    host.set_ylabel("test-loss")
    par1.set_ylabel("test-accuracy")
 
    # plot curves
    p1, = host.plot(range(len(loss)), loss, label="loss")
    p2, = par1.plot(range(len(acc)), acc, label="accuracy")
 
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
 
    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    par1.set_ylim([-0.1, 1.1])
    
    plt.savefig(os.path.join(path_prefix, 'training_loss_acc.png'))
    plt.draw()
    plt.show()

def plot_embedding_with_label(data, label, epoch, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    # 对数据进行归一化处理
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  
    # 创建图形实例   
    fig = plt.figure()      
    
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 24),
                 fontdict={'weight': 'bold', 'size': 10})
    # 指定坐标的刻度
    plt.xticks()        
    plt.yticks()
    plt.title(title, fontsize=14)
    plt.savefig('./plots/figure_'+ epoch +'.png')
    plt.show()
    

def plot_embedding_with_image(data, images, epoch, title):

    fig, ax = plt.subplots()
    fig.set_size_inches(21.6, 14.4)
    plt.axis('off')
    imscatter(data[:, 0], data[:, 1], images, zoom=0.1, ax=ax)
    plt.savefig('./plots/figure_img_'+ epoch +'.png')
    plt.show()


def plot_embedding_with_circle(data, label, epoch, title):
    # 对数据进行归一化处理
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  
    fig, ax = plt.subplots()
    fig.set_size_inches(21.6, 14.4)
    plt.axis('off')
    for i in range(data.shape[0]):
        plt.scatter(data[i,0], data[i,1], c=plt.cm.Set1(label[i] / 24), marker='o', edgecolors='none')
    plt.savefig('./plots/figure_circle_' + epoch + '.png')
    plt.show()

def imscatter(x, y, images, ax=None, zoom=1):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, image in zip(x, y, images):
        image = image * std + mean
        im_f = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im_f, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists 


if __name__ == "__main__":
    with open('./checkpoints/0217_lr1e-5_resnet18/train_info.json') as f:
        train_info = json.load(f)
    plot_acc_loss('./', train_info['test_losses'], train_info['test_scores'])

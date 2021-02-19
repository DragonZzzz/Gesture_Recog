img_w = 224
img_h = 224

batch_size = 64
learning_rate = 1e-6   # 网络学习率
epoches = 10 # 网络训练迭代次数
log_interval = 2 # 打印间隔，默认每2个batch_size打印一次
save_interval = 1 # 模型保存间隔，默认每个epoch保存一次
num_classes = 10 # 训练分类类别数量
model_name = 'resnet18' # 分类网络
experiments_name = '0218_lr1e-6_resnet18_norandom'


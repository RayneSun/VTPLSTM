import torch
import numpy as np

class myError(BaseException):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

class logger(object):
    def __init__(self, dir):
        import datetime
        import visualdl
        self.now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        self.dir = dir + '/' + self.now_time
        self.train_vsdl = visualdl.LogWriter(logdir=self.dir+'/train')
        self.test_vsdl = visualdl.LogWriter(logdir=self.dir+'/test')
        self.flag=1

    def writeTxt(self, strs):
        '''
        写初始文件并打印
        :param strs: 写入的strs
        :return: none
        '''
        print(strs)
        with open(self.dir + '/model.txt', 'a') as f:
            f.write(strs + '\n')

    def task(self):
        import os
        os.system('visualdl --logdir {} --cache-timeout 5'.format(self.dir))
    def runConsole(self):
        '''
        在新的线程生成console
        :return:
        '''
        self.flag=0
        import threading
        runConsole = threading.Thread(target=self.task)
        runConsole.start()


def Gaussian2DLikelihood(outputs, targets):
    '''
    params:
    outputs : 对应99个二维高斯函数[seq_length=99,vehicle_num=26,output_size=5]
    targets : [seq_length=99,vehicle_num=26,size=2]
    return: 每个车每帧的损失值

    '''
    # 提取五个[seq_length=99,vehicle_num=26,1]
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]
    sx, sy = torch.exp(sx), torch.exp(sx)
    corr = torch.tanh(corr)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))

    loss = result.sum()
    counter = outputs.shape[0] * outputs.shape[1]
    return loss / counter

def RMSE(outputs,targets):
    RMSE_loss = torch.nn.MSELoss(reduce=True, size_average=True)
    return RMSE_loss(outputs,targets)


def lrDecline(optimizer, epoch, lr_decay=0.5, lr_decay_epoch=10):
    '''
    Learning_rate随着epoch下降
    para:
    optimizer:优化器
    epoch：当前epoch
    lr_decay：学习率下降多少
    lr_decay_epoch：学习率多少epoch后下降
    return:
    优化器
    '''
    if epoch % lr_decay_epoch:
        return optimizer

    print("Optimizer learning rate has been decreased.")

    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1. / (1. + lr_decay * epoch))
    return optimizer


def optimizerChoose(net, lr, optimizer_name):
    '''

    :param net: 网络模型
    :param lr: 学习率
    :param optimizer_name:优化器选择：RMSprop，Adagrad，Adam
    :return: 优化器
    '''

    RMSprop = torch.optim.RMSprop(net.parameters(), lr=lr)
    Adagrad = torch.optim.Adagrad(net.parameters(), weight_decay=lr)  # lamda_param=0.0005
    Adam = torch.optim.Adam(net.parameters(), weight_decay=lr)
    if optimizer_name == "RMSprop":
        return RMSprop
    elif optimizer_name == "Adagrad":
        return Adagrad
    elif optimizer_name == "Adam":
        return Adam
    else:
        raise myError("optimizer名称有误")


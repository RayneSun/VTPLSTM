import torch
from torch.utils.data import DataLoader

from data_loader import myDataSet
from model import VPTLSTM
from utils import myError, lrDecline, optimizerChoose, lossCaculate
from parameters import train_conf


class trainer(object):
    def __init__(self):
        # 参数初始化
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conf = train_conf()
        # logger初始化
        from utils import logger
        self.logger = logger(dir=self.conf.log_dir)
        # 设置参数检查
        if self.conf.grids_width < 3 or self.conf.grids_height < 3:
            raise myError("请确保grids边长均大于3")
        elif self.conf.grids_width % 2 == 0 or self.conf.grids_height % 2 == 0:
            raise myError("请确保grids_width为奇数")

    def train_init(self, conf):
        print("*" * 40)
        print("载入数据中")
        self.train_data = myDataSet(csv_source=conf.train_csv_source, need_col=conf.need_col,
                                    output_col=conf.output_col,
                                    grids_width=conf.grids_width, grids_height=conf.grids_height,
                                    meter_per_grid=conf.meter_per_grid, road=conf.road_name,long_term=conf.long_term)

        self.test_data = myDataSet(csv_source=conf.test_csv_source, need_col=conf.need_col, output_col=conf.output_col,
                                   grids_width=conf.grids_width, grids_height=conf.grids_height,
                                   meter_per_grid=conf.meter_per_grid, road=conf.road_name,long_term=conf.long_term)
        self.train_data_length, self.test_data_length = self.train_data.__len__(), self.test_data.__len__()
        print("数据载入完成")

        log = 'train_source: {},\n' \
              'length_of_train_data: {},\n' \
              'grids: [{}, {}, meter_per_grid={}],\n' \
              'test_source: {},\n' \
              'length_of_train_data: {}\n' \
              'long_term: {}\n{}'.format(conf.train_csv_source, self.train_data_length, conf.grids_width,
                                                    conf.grids_height, conf.meter_per_grid, conf.test_csv_source,
                                                    self.test_data_length, conf.long_term,'*' * 40)
        self.logger.writeTxt(log)

        # 网络初始化
        self.net = VPTLSTM(rnn_size=conf.rnn_size, embedding_size=conf.embedding_size, input_size=conf.input_size,
                          output_size=conf.output_size,
                          grids_width=conf.grids_width, grids_height=conf.grids_height, dropout_par=conf.dropout_par,
                          device=self.device).to(self.device)
        # log
        if conf.load_model or conf.long_term:  #long_term必须载入模型
            self.net.load_state_dict(torch.load(conf.pretrained_model))
            print("载入历史训练结果成功! ")
        self.logger.writeTxt(str(self.net) + '\n{}'.format('*' * 40))
        self.logger.writeTxt('RMSE_LOSS: {}'.format(conf.add_RMSE))
        self.logger.writeTxt(
            'Optimizer: {},\nLearningRate: {}\n{}'.format(conf.optimizer, conf.learning_rate, '*' * 40))

        # 优化器及所需参数
        self.optimizer = optimizerChoose(net=self.net, lr=conf.learning_rate, optimizer_name=conf.optimizer)

    def trainThread(self, conf):
        print("开始训练，共计{}个epoch".format(conf.epoches))
        for epoch in range(conf.epoches):
            self.optimizer = lrDecline(optimizer=self.optimizer, epoch=epoch, lr_decay_epoch=5)

            train_loss_baches = []
            for batch, (train_x, train_y, train_grids) in enumerate(DataLoader(self.train_data, batch_size=1, shuffle=True)):

                # 处理一组数据
                train_x, train_y, train_grids, hidden_states, cell_states = self.batchExec(x=train_x,
                                                                                           y=train_y,
                                                                                           grids=train_grids,
                                                                                           conf=conf)
                # 返回对应seq_length个二维高斯函数[seq_length, vehicle_num, output_size = 5]
                if conf.long_term:
                    self.net.getFunction(getGrid=self.train_data.getGrid,road_info=self.train_data.road_info,min_Local_Y=self.train_data.min_Local_Y,max_Local_Y=self.train_data.max_Local_Y)
                out = self.net(x_seq=train_x, grids=train_grids, hidden_states=hidden_states, cell_states=cell_states,long_term=conf.long_term)

                #loss计算
                loss=lossCaculate(pred=out, true=train_y, conf=conf)

                train_loss_baches.append(loss.item())

                # 反向传播
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.net.zero_grad()

                # log
                if batch % 5 == 4:
                    self.logger.train_vsdl.add_scalar(tag='loss/batches', step=self.train_data_length * epoch + batch,
                                                      value=sum(train_loss_baches) / len(train_loss_baches))
                    train_loss_baches = []
                    self.test(conf=conf, epoch=epoch, batch=batch)
                    if self.logger.flag:
                        self.logger.runConsole()

                if batch % 50 == 49:
                    pred_x, pred_y = out[9, 0, 0] * 24, out[9, 0, 1] * (self.train_data.max_Local_Y - self.train_data.min_Local_Y) + self.train_data.min_Local_Y
                    true_x, true_y = train_y[9, 0, 0] * 24, train_y[9, 0, 1] * (self.train_data.max_Local_Y - self.train_data.min_Local_Y) + self.train_data.min_Local_Y
                    log='Truth: ({}, {}), Predict:({}, {})'.format(true_x, true_y, pred_x, pred_y)
                    self.logger.writeTxt(log)
            # 每个epoch保存
            if conf.save_model:
                dirs = self.logger.dir + '/net.pkl'
                torch.save(self.net.state_dict(), dirs)
                print("epoch={}, 模型已刷新存储至{}".format(epoch, dirs))

    def batchExec(self, x, y, grids, conf):

        # 迁移数据至GPU
        x = torch.as_tensor(torch.squeeze(x), dtype=torch.float32, device=self.device)
        y = torch.as_tensor(torch.squeeze(y), dtype=torch.float32, device=self.device)
        grids = torch.as_tensor(torch.squeeze(grids), dtype=torch.float32, device=self.device)

        # hidden_state初始化
        vehicle_num = x.shape[1]
        hidden_states = torch.zeros(vehicle_num, conf.rnn_size, device=self.device)
        cell_states = torch.zeros(vehicle_num, conf.rnn_size, device=self.device)
        return x, y, grids, hidden_states, cell_states

    def test(self, conf, epoch, batch):
        test_loss_batches = []
        with torch.no_grad():
            for test_x, test_y, test_grids in DataLoader(self.test_data, batch_size=1, shuffle=True):
                test_x, test_y, test_grids,hidden_states,cell_states=self.batchExec(x=test_x,y=test_y,grids=test_grids,conf=conf)

                if conf.long_term:
                    self.net.getFunction(getGrid=self.test_data.getGrid,road_info=self.test_data.road_info,min_Local_Y=self.test_data.min_Local_Y,max_Local_Y=self.test_data.max_Local_Y)
                out = self.net(x_seq=test_x, grids=test_grids, hidden_states=hidden_states, cell_states=cell_states,long_term=conf.long_term)

                loss=lossCaculate(pred=out, true=test_y, conf=conf)

                test_loss_batches.append(loss.item())
            self.logger.test_vsdl.add_scalar(tag='loss/batches', step=self.train_data_length * epoch + batch,value=sum(test_loss_batches) / len(test_loss_batches))



if __name__ =='__main__':
    trainer = trainer()
    trainer.train_init(conf=trainer.conf)
    trainer.trainThread(conf=trainer.conf)


import numpy as np
import matplotlib.pyplot as plt
from utils import myError

class visualize(object):
    def __init__(self):
        self.colors = ['aqua',
                       'black',
                       'blue',
                       'brown',
                       'darkcyan',
                       'darkgreen',
                       'darkmagenta',
                       'darkorchid',
                       'darkred',
                       'darkslategray',
                       'darkviolet',
                       'deeppink',
                       'fuchsia',
                       'indigo',
                       'lime',
                       'magenta',
                       'maroon',
                       'navy',
                       'orangered']

    def trajectoryDisplay(self,true,pred,max_Local_Y,min_Local_Y,road_width,vehicle_list=None):
        '''
        :param vehicle_list: #要打印的车ID
        :param true: tensor(seq_len,vehicle_num,2)
        :param pred: tensor(seq_len,vehicle_num,5)
        :param max_Local_Y: 归一化用
        :param min_Local_Y: 归一化用
        :param road_wight: 归一化用
        :return:
        '''

        true,pred=self.unormalize(true=true, pred=pred, max_Local_Y=max_Local_Y, min_Local_Y=min_Local_Y, road_width=road_width)

        vehicle_num=true.shape[1]

        if vehicle_list==None:
            vehicle_list=range(vehicle_num)
        if max(vehicle_list) >vehicle_num-1:
            raise myError('车数很少')

        _, ax = plt.subplots()


        for vehicle in vehicle_list:

            true_vehicle_traj = true[:, vehicle, :]
            true_y,true_x = true_vehicle_traj[:, 1],true_vehicle_traj[:, 0]

            pred_vehicle_traj = pred[:, vehicle, :]
            pred_y, pred_x = pred_vehicle_traj[:, 1], pred_vehicle_traj[:, 0]

            color = self.colors[vehicle%len(self.colors)]
            label='vehicle-'+str(vehicle)
            ax.plot(true_y, true_x, color=color,label=label,linestyle='-',marker='o',markersize=3)
            ax.plot(pred_y, pred_x, color=color,linestyle='--',marker='x',markersize=5)

        plt.legend()
        plt.xlim(min_Local_Y, max_Local_Y)
        plt.ylim(0, road_width)
        plt.show()

    def unormalize(self,true, pred, max_Local_Y, min_Local_Y, road_width):
        '''

        :param true: tensor(seq_len,vehicle_num,2)
        :param pred: tensor(seq_len,vehicle_num,5)
        :param max_Local_Y: 最大y
        :param min_Local_Y: 最小y
        :param road_wight: 路宽
        :return: true: array(seq_len,vehicle_num,2)
                 pred: array(seq_len,vehicle_num,2)
        '''
        true = true.cpu().numpy()
        pred = pred.cpu().numpy()

        true[:, :, 0] = true[:, :, 0] * road_width
        true[:, :, 1] = true[:, :, 1] * (max_Local_Y - min_Local_Y) + min_Local_Y
        pred[:, :, 0] = pred[:, :, 0] * road_width
        pred[:, :, 1] = pred[:, :, 1] * (max_Local_Y - min_Local_Y) + min_Local_Y

        return true,pred[:,:,0:2]

# test
# from data_loader import myDataSet
# from parameters import longTerm
# import torch
# from torch.utils.data import DataLoader
#
# conf = longTerm()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# dataSource = myDataSet(csv_source=conf.csv_source, need_col=conf.need_col,
#                        output_col=conf.output_col,
#                        grids_width=conf.grids_width, grids_height=conf.grids_height,
#                        meter_per_grid=conf.meter_per_grid, long_term=True, road=conf.road_name)
#
# drawer = visualize()
# for x, y, _ in DataLoader(dataSource, batch_size=1, shuffle=True):
#     true = torch.as_tensor(torch.squeeze(x), dtype=torch.float32, device=device)
#     pred = torch.as_tensor(torch.squeeze(y), dtype=torch.float32, device=device)
#
#     drawer.trajectoryDisplay(true=true, pred=pred, max_Local_Y=dataSource.max_Local_Y, min_Local_Y=dataSource.min_Local_Y,
#                              road_wight=dataSource.road_info['max_Local_X'])

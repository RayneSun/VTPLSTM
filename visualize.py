import torch
from torch.utils.data import DataLoader

from data_loader import myDataSet
from model import VPTLSTM
from utils import myError, visualize
from parameters import vis_conf


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    conf = vis_conf()

    # 设置参数检查
    if conf.grids_width < 3 or conf.grids_height < 3:
        raise myError("请确保grids边长均大于3")
    elif conf.grids_width % 2 == 0 or conf.grids_height % 2 == 0:
        raise myError("请确保grids_width为奇数")

    predict(conf, device)


def predict(conf, device):
    # 载入数据
    print("*" * 40)
    dataSource = myDataSet(csv_source=conf.csv_source, need_col=conf.need_col,
                           output_col=conf.output_col,
                           grids_width=conf.grids_width, grids_height=conf.grids_height,
                           meter_per_grid=conf.meter_per_grid, long_term=conf.long_term, road=conf.road_name)

    # 网络初始化
    print("*" * 40)
    net = VPTLSTM(rnn_size=conf.rnn_size, embedding_size=conf.embedding_size, input_size=conf.input_size,
                  output_size=conf.output_size,
                  grids_width=conf.grids_width, grids_height=conf.grids_height, dropout_par=0,
                  device=device).to(device)

    # 载入模型
    net.load_state_dict(torch.load(conf.pretrained_model))
    print("载入历史训练结果成功! ")
    print(net)

    # visualize初始化
    visualizer = visualize()

    # 开始
    with torch.no_grad():
        for train_x, train_y, head_grids in DataLoader(dataSource, batch_size=1, shuffle=True):

            net.getFunction(dataSource.getGrid, dataSource.road_info, min_Local_Y=dataSource.min_Local_Y,
                            max_Local_Y=dataSource.max_Local_Y)

            # 迁移数据至GPU
            train_x = torch.as_tensor(torch.squeeze(train_x), dtype=torch.float32, device=device)
            train_y = torch.as_tensor(torch.squeeze(train_y), dtype=torch.float32, device=device)
            head_grids = torch.as_tensor(torch.squeeze(head_grids), dtype=torch.float32, device=device)

            # hidden_state初始化
            vehicle_num = train_x.shape[1]
            hidden_states = torch.zeros(vehicle_num, conf.rnn_size, device=device)
            cell_states = torch.zeros(vehicle_num, conf.rnn_size, device=device)

            #onnx输出
            # input_names=['train_x','head_grids','hidden_states','cell_states','long_term']
            # output_names=['pred']
            # torch.onnx.export(net,(train_x,head_grids,hidden_states,cell_states),'example.onnx',export_params=True,verbose=True)

            #预测
            pred = net(x_seq=train_x, grids=head_grids, hidden_states=hidden_states, cell_states=cell_states, long_term=conf.long_term)

            if conf.long_term:
                train_y = torch.cat([train_x[:, :, :2], train_y], dim=0)
            else:
                pred = pred[2:]


            visualizer.trajectoryDisplay(true=train_y, pred=pred, max_Local_Y=dataSource.max_Local_Y,
                                         min_Local_Y=dataSource.min_Local_Y,
                                         road_width=dataSource.road_info['max_Local_X'], vehicle_list=None)


main()

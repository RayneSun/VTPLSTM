import pandas as pd
import os

import random
from parameters import dataExecute


def saveCsv(df, file_name):
    '''
    :param df: 存入df
    :param file_name: 文件名
    :return: 无
    '''
    df=df.reset_index(drop=True)  #重置索引
    if not os.path.exists('\\'):
        os.makedirs('\\')
    df.to_csv('data\\' + file_name + '.csv', mode='a', header=True)


def cutbyRoad(df=None, road=None):
    '''

    :param df: 打开后文件
    :param road: 路段名称
    :return: 切路df,按照全局时间排序
    '''
    road_df = df[df['Location'] == road]
    return road_df.sort_values(by='Global_Time', ascending=True)


def cutbyPosition(road_df, start_y=0, start_time=0, area_length=50):
    '''
    给定起始时间，起始y，区间长度，输出区间内车辆list
    :param road_df:限定路段后的df
    :param start_y: 区域开始段，单位为ft
    :param start_time: 起始时间，0.1s
    :param area_length: 区域长度单位为m
    :return: vehicle_list为起始框内部车辆编号
    '''
    area_df = road_df[road_df['Global_Time'] == start_time]
    area_df = area_df[(area_df['Global_Y'] - start_y <= area_length) & (
            area_df['Global_Y'] - start_y >= 0)]
    vehicle_list = area_df['Vehicle_ID'].unique()
    if len(list(vehicle_list)) <= 2:
        return None
    else:
        return list(vehicle_list)


def cutbyTime(road_df, start_time=0, vehicle_list=None, time_length=10.0, stride=1):
    '''
    :param road_df:road_df
    :param start_time: 开始实践    :param time_length: 采样时间长度,单位为s
    :param stride: 采样步长
    :return: 返回一组清洗完数据time
    '''
    temp_df = road_df[road_df['Vehicle_ID'].isin(vehicle_list)]
    one_sequence = pd.DataFrame()
    for vehicle in vehicle_list:
        for time in range(int(time_length * 10 / stride)):
            df = temp_df[
                (temp_df['Vehicle_ID'] == vehicle) & (temp_df['Global_Time'] == (start_time + time * stride))]
            if df.shape[0] == 1 and int(df['Lane_ID'])<=5:
                one_sequence = pd.concat([one_sequence, df])

            else:
                return None
    return one_sequence


def unitConversion(df):
    '''
    转换后长度单位为m，时间单位为0.1秒
    :param df: 被转换df
    :return: 转换后df
    '''
    ft_to_m = 0.3048
    df['Global_Time'] = df['Global_Time'] / 100
    for strs in ["Global_X", "Global_Y", "Local_X", "Local_Y", "v_length", "v_Width"]:
        df[strs] = df[strs] * ft_to_m
    df["v_Vel"] = df["v_Vel"] * ft_to_m*3.6
    return df


def main():
    conf = dataExecute()
    init_df = pd.read_csv(conf.data_source, usecols=conf.useCols)
    road_df = cutbyRoad(init_df, road=conf.road)

    road_df = unitConversion(road_df)

    min_Global_Y, max_Global_Y = road_df['Global_Y'].min()+100, road_df['Global_Y'].max()
    min_Global_Time, max_Global_Time = road_df['Global_Time'].min(), road_df['Global_Time'].max()
    total_dist=int((max_Global_Y - min_Global_Y) / (conf.area_step))
    total_time=int((max_Global_Time - min_Global_Time) / (conf.time_step * 10))
    print("共计{}--{}组数据，时间步长为{}，距离步长为{}".format(total_dist,total_time,conf.time_step,conf.area_step))

    total_data=0
    for dist_index in range(conf.hist_dist,total_dist):

        for time_index in range(conf.hist_time,total_time):
            if conf.noise:
                time_noise=random.randint(0,100)
                dist_noise=random.randint(0,100)
            else:
                time_noise,dist_noise=0,0

            start_time = min_Global_Time + time_index * conf.time_step * 10+time_noise*10
            start_y = min_Global_Y + dist_index * conf.area_step+dist_noise

            vehicle_list = cutbyPosition(road_df, start_y=start_y, start_time=start_time,
                                         area_length=conf.area_length)

            if vehicle_list is None:
                # print('{}秒时刻，{}为起点区域内车辆过少，进入下个时段'.format(start_time * 0.1, start_y))
                print('{}--{}内车辆过少，进入下个时段'.format(dist_index, time_index))
                continue

            one_sequence = cutbyTime(road_df, start_time=start_time, vehicle_list=vehicle_list,
                                     time_length=conf.time_length,
                                     stride=conf.stride)
            if one_sequence is None:
                # print('{}时刻，{}为起点区域内车辆存在消失，进入下个时段'.format(start_time * 0.1, start_y))
                print('{}--{}内车辆存在消失或驶入辅助匝道，进入下个时段'.format(dist_index, time_index))
            else:
                total_data+=1
                saveCsv(one_sequence, file_name=conf.road)
                print('{}/{} - {}/{} saved! Exist {} data! '.format(dist_index, total_dist,time_index, total_time,total_data))

            if total_data == conf.need_num:
                print("数据采集完成")
                break
        if total_data ==conf.need_num:
            print("数据采集完成")
            break
main()

class dataExecute(object):
    def __init__(self):
        self.data_source = "./data/data.csv"
        self.road = "us-101"
        self.useCols = [
            "Vehicle_ID",
            "Frame_ID",
            "Total_Frames",
            "Global_Time",
            "Global_X",
            "Global_Y",
            "Local_X",
            "Local_Y",
            "v_length",
            "v_Width",
            "v_Class",
            "Location",
            "v_Vel",
            "Lane_ID"
        ]
        self.area_length = 80
        self.time_length = 10
        self.area_step = 20
        self.time_step = 30
        self.stride = 5.0
        self.hist_dist = 6
        self.hist_time = 3
        self.noise = True
        self.need_num = 5


class trainconf(object):
    def __init__(self):
        self.train_csv_source = "./data/train.csv"
        self.test_csv_source = "./data/test.csv"
        self.road_name = "US101_info"
        self.need_col = [
            "Vehicle_ID",
            "Global_Time",
            "Local_X",
            "Local_Y",
            "v_length",
            "v_Width",
            "v_Class",
            "Lane_ID"
        ]
        self.output_col = [
            "Local_X",
            "Local_Y",
            "v_length",
            "v_Width",
            "v_Class",
            "Lane_ID"
        ]
        self.grids_width = 5
        self.grids_height = 19
        self.meter_per_grid = 2
        self.load_model = 0
        self.pretrained_model = "./log/2020-10-22-11-09/net.pkl"
        self.save_model = 1
        self.log_dir = "./log"
        self.rnn_size = 128
        self.embedding_size = 32
        self.input_size = 9
        self.output_size = 5
        self.dropout_par = 0.4
        self.epoches = 20
        self.learning_rate = 0.01
        self.optimizer = "Adagrad"
        self.add_RMSE=True


class longTerm(object):
    def __init__(self):
        self.csv_source = "./data/test.csv"
        self.need_col = [
            "Vehicle_ID",
            "Global_Time",
            "Local_X",
            "Local_Y",
            "v_length",
            "v_Width",
            "v_Class",
            "Lane_ID"
        ]
        self.output_col = [
            "Local_X",
            "Local_Y",
            "v_length",
            "v_Width",
            "v_Class",
            "Lane_ID"
        ]
        self.road_name = "US101_info"
        self.grids_width = 5
        self.grids_height = 19
        self.meter_per_grid = 2
        self.load_model = 1
        self.pretrained_model = "./log/2020-10-22-11-09/net.pkl"
        self.rnn_size = 128
        self.embedding_size = 64
        self.input_size = 9
        self.output_size = 5

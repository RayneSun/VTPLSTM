
**Project state: Still Fixing**

Inspired by the performence of Social LSTM. We created a model that could predict vehicle's trajectory in 5s. The model uses LSTM as center. We are still working on improving its performance!

**Implement detail:**

Baseline implementation: [https://github.com/quancore/social-lstm](https://github.com/quancore/social-lstm)

Baseline paper: [http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf](http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf)

**Documents Introduction**

**Data(sample):** There are two .csv files obtained by data_generator.py. The number of sequences is too small for study, so you need to generate dataset in your computers.

**Log:** Including our training result and the curve of loss.

**Parameters.py:** All of the setting are in this files including train, long-term train, visualize, data generator.

**Data_generator.py:** To generate sequences for training and visualize.

**Data_loader.py:** To read .csv files and execute them. (Normalization, social pooling…)

**Model.py:** Kernel file, including the definition of our model

**Train.py:** The entrance of training.

**Visualize.py:** Using matplotlib to visualize predicted trajectory.

**Model information**

VPTLSTM(

(cell): LSTMCell(64, 32)

(input_embedding_layer): Linear(in_features=9, out_features=32, bias=True)

(social_tensor_conv1): Conv2d(32, 16, kernel_size=(5, 3), stride=(2, 1))

(social_tensor_conv2): Conv2d(16, 8, kernel_size=(5, 3), stride=(1, 1))

(social_tensor_embed): Linear(in_features=32, out_features=32, bias=True)

(output_layer): Linear(in_features=32, out_features=5, bias=True)

(relu): ReLU()

(dropout): Dropout(p=0, inplace=False)

)

**Visualization**

The predicted trajectory is correct but it lack of accuracy on intention recognition. Maybe there are too less vehicles to change lane so that the model couldn’t get the conditions of changing lane. We are trying to get more relevant datasets about it.

<![if !vml]>![](file:///C:/Users/Rayne/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png)<![endif]>

<![if !vml]>![](file:///C:/Users/Rayne/AppData/Local/Temp/msohtmlclip1/01/clip_image004.png)<![endif]><![if !vml]>![](file:///C:/Users/Rayne/AppData/Local/Temp/msohtmlclip1/01/clip_image006.png)<![endif]>

<![if !vml]>![](file:///C:/Users/Rayne/AppData/Local/Temp/msohtmlclip1/01/clip_image008.png)<![endif]>

(The solid line is the real track and the dashed line is the predicted track)

**Requirements:**

Pytorch

Numpy

Matplotlib

Pandas

visualdl

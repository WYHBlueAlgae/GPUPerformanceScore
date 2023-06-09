import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 输入数据
#           [2520,16384,176,512,128,128,512,72,1008,384,24],#4090
#           [2505,9728,112,304,76,76,304,64,716,256,16],#4080
#           [2610,7680,80,240,60,60,240,48,504,192,12],#4070ti
#           [2475,5888,64,184,46,46,184,36,504,192,12],#4070 
X_train = np.array([
            

            [1860,10752,112,336,84,84,336,6,1008,384,24],#3090ti
            [1695,10496,112,328,82,82,328,6,936,384,24],#3090
            [1665,10240,112,320,80,80,320,6,912,384,12],#3080ti
            [1710,8960,96,280,70,70,280,5,912,384,12],#3080 12
            [1710,8704,96,272,68,68,272,5,760,320,10],#3080
            [1770,6144,96,192,48,48,192,4,608,256,8],#3070ti
            [1725,5888,96,184,46,46,184,4,448,256,8],#3070
            [1665,4864,80,152,38,38,152,4,448,256,8],#3060ti
            [1777,3584,48,112,28,28,112,3,360,192,12],#3060 12
            [1777,2560,32,80,20,20,80,2,224,128,8],#3050
            [1545,4352,88,272,68,68,544,5.5,616,352,11],#2080ti
            [1815,3072,64,192,48,48,384,4,496,256,8],#2080s
            [1710,2944,64,184,46,46,368,4,448,256,8],#2080
            [1770,2560,64,160,40,40,320,4,448,256,8],#2070s
            [1620,2304,64,144,36,36,288,4,448,256,8],#2070
            [1650,2176,64,136,34,34,272,4,448,256,8],#2060s
            [1680,1920,48,120,30,30,240,3,336,192,6],#2060
            [1770,1536,48,96,24,0,0,1.5,288,192,6],#1660ti
            [1785,1408,48,88,22,0,0,1.5,336,192,6],#1660s
            [1785,1408,48,88,22,0,0,1.5,192,192,6],#1660
            [1725,1280,32,80,20,0,0,1,192,128,4],#1650s
            [1665,896,32,56,14,0,0,1,128,128,4],#1650
            [1582,3584,88,224,28,0,0,2.75,484.4,352,11],#1080ti
            [1733,2560,64,160,20,0,0,2,320.3,256,8],#1080
            [1683,2432,64,152,19,0,0,2,256.3,256,8],#1070ti
            [1683,1920,64,120,15,0,0,2,256.3,256,8],#1070
            [1709,1280,48,80,10,0,0,1.5,192.2,192,6],#1066
            [1392,768,32,48,6,0,0,1,112,128,4],#1050ti
            [1455,640,32,40,5,0,0,1,112,128,2],#1050
            [1468,384,16,24,3,0,0,0.5,48,64,2],#1030 d5
            [1076,2816,96,176,22,0,0,3,336.6,384,6],#980ti
            [1216,2048,64,128,16,0,0,2,224.4,256,4],#980
            [1178,1664,56,104,13,0,0,2,224.4,256,4],#970
            [1178,1024,32,64,8,0,0,1,112.2,128,2],#960
            [1188,768,32,48,6,0,0,1,105.8,128,2],#950
            [928,2880,48,240,25,0,0,1.5,336.6,384,3]
])

# 目标值
           #36164,28124,22752,17876,
y_train = np.array([
           21748,19799,19568,18584,17565,
           14826,13496,11577,8696,6192,14657,11589,10995,10124,
           8926,8660,7498,6268,5991,5414,4688,3596,9874,7542,
           6802,6049,4178,2338,1743,1083,5776,4337,3638,2280,
           1874,3365
])

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 定义神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(11,)),  # 隐藏层
    tf.keras.layers.Dense(1)  # 输出层
])

# 编译模型
learning_rate = 0.05
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X_train_scaled, y_train, epochs=5000, verbose=0)

# 对新的输入数据进行预测
X_test = np.array([
          [2505,9728,112,304,76,76,304,64,716,256,16],#4080 28124 18124
          [2520,16384,176,512,128,128,512,72,1008,384,24],#4090 36164 26164
          [1710,8704,96,272,68,68,272,5,760,320,10],#3080 17565 7565
          [1860,10752,112,336,84,84,336,6,1008,384,24],#3090ti 21748 11748
          [1665,4864,80,152,38,38,152,4,448,256,8],#3060ti 11577 1577
          [2610,7680,80,240,60,60,240,48,504,192,12],#4070ti 22752 12752
          [1710,2944,64,184,46,46,368,4,448,256,8],#2080 10995 995
          [1709,1280,48,80,10,0,0,1.5,192.2,192,6]#1066 4178 -5822
])

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
print('Predicted score:', y_pred.flatten())

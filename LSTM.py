# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 18:12:37 2022

@author: c
"""

# =======基础库===============
import pandas as pd
import numpy as np

# =======数据处理库============
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# =========GRU相关库=============
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras import optimizers


# =========画图==========
import matplotlib.pyplot as plt



# 在此处输入视频的ID
# aid = 7105150861512379690
aid = 7105150861512379690



## 处理数据
df = pd.read_csv(r'.\videos.csv',encoding='gbk')
print(len(df))
df.dropna(inplace = True)
print(len(df))
F_df = pd.read_csv(r'.\F.csv',encoding='gbk')


df = pd.merge(df,F_df,on='username')
df = df[df['aid']==aid]
df = df.loc[:,['dig_count','comment_count','down_count','share_count','f1_language', 'f2_pattern']]

# df.columns
# print(df)

#%% 
## 模型参数
''' GRU输入为三维数据，分别为 样本、时间、特征  '''
mem_his_days = 12    # 记忆时期数
pre_days = 12     # 预测时期数

## 产生输入和输出，每个输入有三维，输出为一维
df['label'] = df['dig_count'].shift(-pre_days)

scaler1 = MinMaxScaler(feature_range=(0,1))
sca_X = scaler1.fit_transform(df.iloc[:,:-1])


y = df['label'].values[mem_his_days-1:-pre_days]

deq = deque(maxlen=mem_his_days)

X=[]
for i in sca_X:
    deq.append(list(i))
    if len(deq)==mem_his_days:
        # print(deq)
        X.append(list(deq))


X = X[:-pre_days]
X = np.array(X)
y = np.array(y)



## Model 构建
X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=False,test_size=0.1)

model = Sequential()

# model.add(GRU(5,input_shape=X.shape[1:],activation='relu',return_sequences=True)) 
# model.add(Dropout(0.1))

model.add(GRU(12,input_shape=X.shape[1:],activation='relu'))
model.add(Dropout(0.1))


# model.add(GRU(5,input_shape=X.shape[1:],activation='relu'))
# model.add(Dropout(0.1))


model.add(Dense(5,activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1))

# adam （学习率），loss是目标指标，mape平均方差百分率
model.compile(optimizer=optimizers.Adam(0.01),loss='mse',metrics=['mape']) # 

print(model.summary())

# Checkpoint=ModelCheckpoint(r'./models/best_model',
#                            monitor=CheckPointMethod,
#                            verbose=1,
#                            save_best_only=True,
#                            mode='auto')
# callbacks=[checkpoint] validation_data=(X_test,y_test)




# 输出归一化
# scaler2 = MinMaxScaler(feature_range=(0,1))
# y_train  = scaler2.fit_transform(y_train .reshape(-1,1))

early_stop = EarlyStopping(monitor='loss', patience=20, verbose=2) # 提前终止拟合过程，以防过拟合

history = model.fit(X_train,y_train,batch_size=12,epochs=1000,validation_data=(X_test,y_test),callbacks=[early_stop],verbose=2, shuffle=False)


model.evaluate(X_train ,y_train)

pre = model.predict(X_train)


pl_x = [i for i in range(len(y_train ))]
plt.plot(pl_x, y_train,color='red',label='real')
plt.plot(pl_x, pre,color='green',label='predict')
plt.legend()
plt.show()

# 测试过程mse的变化
plt.plot(history.history['val_loss'],label='mse')
plt.legend()
plt.show()

plt.plot(history.history['mape'],label='mape')
plt.legend()
plt.show()
# from tensorflow.keras.models import load_model
# best_model = load_model('./models/best_model')
# best_model.evaluate(X_test,y_test)
# pre = best_model.predict(X_test)
# print(len(pre))

#%% figure

from sklearn.metrics import r2_score
from numpy import polyfit,poly1d,polyval
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
sns.pairplot(df.iloc[:,[0,1,2,3,4]].dropna(),diag_kind='kde',markers = '+',plot_kws=dict(s=50,edgecolor='b',linewidth=1))


p1 = df.plot.box(title="Box Chart")
plt.show()

y1 = y
xx = np.linspace(1, len(y1),len(y1))
para = polyfit(xx,y1,3) 
ry = polyval(para,xx)
p1 = poly1d(para)
print(f'拟合高次多项式为\n{p1}')
rr1 = r2_score(ry, y1)



rr2 = r2_score(y_train,pre)


pl_x = [i for i in range(len(y_train ))]
plt.plot(pl_x, y_train,color='red',label='real')
plt.plot(pl_x, pre,color='green',label='GRU predict')
plt.plot(xx, ry,color='blue',label='linear predict')
plt.legend()
plt.show()

print('===========================')
print('线性拟合的可决系数为:', rr1)
print('===========================')
print('GRU进行数据预测的可决系数为:', rr2)
print('GRU进行数据预测的mse为:', min(history.history['val_loss']))
print('GRU进行数据预测的mape为:', min(history.history['val_mape']))
# 计算RMSE
rmse = sqrt(mean_squared_error(y_train,pre))
print('Test RMSE: %.3f' % rmse)


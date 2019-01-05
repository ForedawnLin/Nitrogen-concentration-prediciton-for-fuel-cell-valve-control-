import numpy as np 
import data_loader as dl 

import torch 
import torch.nn as nn
from torch.autograd import Variable 
import torch.optim as optim
from LSTM_model import Regressor_v1

import matplotlib.pyplot as plt  



file_path='../data/'
file_name='SAIC_Purging_Simulation_n2.xlsx'
time,data=dl.excel_load(file_path,file_name) 
N2=data[-1]

#### data correlation filtering #### 
coeff_th=0.995
feature_flted,_=dl.feature_correlation_removal(data,coeff_th,1,0)
# print (data_flted)
print ('feature_flted_shape',np.shape(np.array(feature_flted).T))

#### stadnardize data without gt #### 
processed_feature,train_std,train_mean=dl.standardize_data(feature_flted,1,0,0)
print ('processed_data:',np.shape(processed_feature))
print ('train_std',train_std)
print ('train_mean',train_mean)
# train_std_check=[np.std(one_type) for one_type in processed_data]
# print ('std_check:',train_std_check)

 
features=list(np.array(processed_feature).T) ### make first dim data pt
# print ('features',features[:10])
# print ('N2',N2[9])
total_data=len(features)
print ('total_data',total_data)
# print (features)
# print ('features_shape',np.shape(features))

time_step=100
batch_size=64
epochs=500

# feature_seqs=[features[i:i+total_data-time_step+1] for i in np.arange(time_step)]


feature_seqs=[features[i:i+time_step] for i in np.arange(total_data-time_step+1)] 
N2_gt=[N2[i] for i in np.arange(time_step-1,total_data)]
print ('LSTM input feature:',np.shape(feature_seqs))
print ('output:',np.shape(N2_gt))
model=Regressor_v1([feature_seqs,N2_gt])
model.train_network(epochs,batch_size)
pred=model.prediction(feature_seqs)
# print (np.shape(pred))
# print (list(pred))
plt.figure(1)
plt.plot(range(len(pred)),N2_gt,'bo-')
plt.plot(range(len(pred)),list(pred),'ro-')
plt.legend(['N2_gt','pred'])
plt.show()


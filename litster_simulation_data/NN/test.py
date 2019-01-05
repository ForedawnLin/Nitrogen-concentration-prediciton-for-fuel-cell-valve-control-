import sys 
sys.path.insert(0,'../LSTM')

import numpy as np 
import data_loader as dl 
import matplotlib.pyplot as plt 


import torch 
import torch.nn as nn
from torch.autograd import Variable 
import torch.optim as optim

import numpy as np

from torch.autograd import Function


### try different param init 



file_path='../data/'
file_name='SAIC_Purging_Simulation_n2.xlsx'
time,data=dl.excel_load(file_path,file_name) 
N2=data[-1]

#### data correlation filtering #### 
coeff_th=0.995
feature_flted,_=dl.feature_correlation_removal(data,coeff_th,1,0)
# feature_flted=[feature_flted[i] for i in range(len(feature_flted)) if i != 6 and i !=4]  ## for n1 data set only 
# print (data_flted)
print ('feature_flted_shape',np.shape(np.array(feature_flted)))

#### stadnardize data without gt #### 
# processed_feature,train_std,train_mean=dl.standardize_data(feature_flted,1,0,0)
# print ('processed_data:',np.shape(processed_feature))
# print ('train_std',train_std)
# print ('train_mean',train_mean)
processed_feature=feature_flted
# del processed_feature[1]
# del processed_feature[1]

### use N2 diff as target value ###
N2_diff=[[N2[i]-N2[i-1]] for i in np.arange(1,len(N2))] 
# N2_diff.insert(0,N2[0]) ### since N2 diff, first output should be zero 
# N2_diff=np.multiply(N2_diff,10)
# print ('N2_diff',N2_diff)
gt=Variable(torch.FloatTensor(N2_diff))

### transpose feature ### 
features=list(np.array(processed_feature).T)
features=features[:-1]
# features=np.reshape(features,(2822,1))
features_2_fed=Variable(torch.FloatTensor(features))
# print ('check features',features_2_fed)


# plt.plot(range(len(N2_diff)),N2_diff,'o-')
# plt.show()




### MLP strcture ####
## to construct MLP structure, haven't finish yet, no time included 

LSTM_h1=1024
LSTM_h2=512
LSTM_h3=256


Linear1=nn.Linear(in_features=7,out_features=LSTM_h1,bias=True)
Linear2=nn.Linear(in_features=LSTM_h1,out_features=LSTM_h2,bias=True)
Linear3=nn.Linear(in_features=LSTM_h2,out_features=LSTM_h3,bias=True)
Linear4=nn.Linear(in_features=LSTM_h3,out_features=1,bias=True)



Relu1=nn.ReLU()
Tanh1=nn.Tanh()
Sigmoid1=nn.Sigmoid()
optimizer1 = optim.SGD([{'params':Linear1.parameters()},{'params':Linear2.parameters()},{'params':Linear3.parameters()},{'params':Linear4.parameters()}], lr = 0.000001, momentum=0.8)
# lambda1 = lambda epoch: epoch // 30
# lambda2 = lambda epoch: 0.95 ** epoch
# scheduler = optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda=[lambda1, lambda2])
# optimizer1 = optim.RMSprop(Linear1.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
# optimizer2 = optim.RMSprop(Linear2.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
loss = nn.L1Loss()

train_ord=np.random.permutation(len(gt))
nn.init.xavier_uniform(Linear1.weight)
nn.init.xavier_uniform(Linear2.weight)
nn.init.xavier_uniform(Linear3.weight)
nn.init.xavier_uniform(Linear4.weight)

for i in range(250):
	### randomize the feautre and gt 
	# N2_diff_rand=[N2_diff[n] for n in train_ord]
	# gt=Variable(torch.FloatTensor(N2_diff_rand))
	# features_rand=[features[n] for n in train_ord]
	# features_2_fed=Variable(torch.FloatTensor(features_rand))
	#### randomize ends 

	Linear1.zero_grad()
	Linear2.zero_grad()
	Linear3.zero_grad()
	Linear4.zero_grad()
	# print('in_shape',np.shape(features_2_fed))
	fc1_out=Linear1(features_2_fed)
	sig1_out=Tanh1(fc1_out)
	fc2_out=Linear2(sig1_out)
	sig2_out=Tanh1(fc2_out)
	fc3_out=Linear3(sig2_out)
	sig3_out=Tanh1(fc3_out)
	fc4_out=Linear4(sig3_out)
	output=fc4_out
	# print('out_shape',np.shape(output))

	err=loss(output,gt)
	err.backward()
	# scheduler.step() 
	optimizer1.step()
	print ('err',err)
	print (i)


print ('output',output)
print ('gt',gt)
# print ('params',list(Linear1.parameters()))
plt.figure(1)
plt.plot(range(len(output)),list(gt),'bo-')
plt.plot(range(len(output)),list(output),'ro-')
plt.ylabel('The difference of N2 concentration',fontsize=16)
plt.xlabel('Time sequence',fontsize=16)
plt.title('The difference of N2 concentration')
plt.legend(['Ground truth','Prediction'],prop={'size':16})
plt.show()




 

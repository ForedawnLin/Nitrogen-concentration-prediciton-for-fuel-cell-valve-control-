import numpy as np 
import data_loader as dl 

import torch 
import torch.nn as nn
from torch.autograd import Variable 
import torch.optim as optim
from LSTM_model import Regressor_v1

import matplotlib.pyplot as plt  



time_step=10
batch_size=64
epochs=30

# feature_seqs=[features[i:i+total_data-time_step+1] for i in np.arange(time_step)]
x1=np.arange(2000)
y=np.sin(x1)
features=[x1[i] for i in range(len(x1))]
# print (np.shape(features))
N2=y
total_data=len(x1)

feature_seqs=[features[i:i+time_step] for i in np.arange(total_data-time_step+1)] 
N2_gt=[N2[i] for i in np.arange(time_step-1,total_data)]
print ('LSTM input feature:',np.shape(feature_seqs))
print ('output:',np.shape(N2_gt))
# model=Regressor_v1([feature_seqs,N2_gt])
# model.train_network(epochs,batch_size)
# pred=model.prediction(feature_seqs)
# print (np.shape(pred))
# print (list(pred))
# plt.figure(1)
# plt.plot(range(len(pred)),N2_gt,'bo-')
# plt.plot(range(len(pred)),list(pred),'ro-')
# plt.legend(['N2_gt','pred'])
# plt.show()


batch_size = 2000
in_size = 1
LSTM1_no = 15
output_no=1

LSTM1 = nn.LSTM(in_size, LSTM1_no, 5,batch_first=True)
Linear1 = nn.Linear(in_features=LSTM1_no,out_features=output_no)
Sigmoid=nn.Sigmoid()

# lstm1=nn.LSTMCell(in_size, LSTM1_no)
# lstm2 = nn.LSTMCell(LSTM1_no, LSTM1_no)
# linear1 = nn.Linear(in_features=LSTM1_no,out_features=output_no)

feature_seqs=np.reshape(feature_seqs,(1991,10,1))
feature_seq = Variable(torch.FloatTensor(feature_seqs))
target = Variable(torch.FloatTensor(N2_gt))
# feature_seq=Variable(torch.randn(time_steps, batch_size, in_size))
# target=Variable(torch.FloatTensor(batch_size).random_(0, 1000))
loss = nn.MSELoss()
opti1 = optim.RMSprop(LSTM1.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
opti2 = optim.RMSprop(Linear1.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

for i in range(1000):
	LSTM1.zero_grad()
	Linear1.zero_grad()

	output_lstm1, _ = LSTM1(feature_seq)
	
	# print ('input',feature_seq[3,:,:])
	# print ('output',target[3])
	# print ('LSTM_output',np.shape(output_lstm1[:,-1,:]))
	last_output = output_lstm1[:,-1,:]
	# lstm1_o,_=lstm1(feature_seq)
	# lstm2_o,_=lstm2(lstm1_o)
	# linear1_o=linear1(lstm2_o)	



	sig=Sigmoid(last_output)
	output=Linear1(sig)
	err = loss(output, target)
	err.backward()
	opti1.step()
	opti2.step()
	print ('error',err)



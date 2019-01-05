import torch 
import torch.nn as nn
from torch.autograd import Variable 
import torch.optim as optim

import numpy as np

from torch.autograd import Function


class Regressor_v1(nn.Module):
	def __init__(self,data):
		###  input: data: [feature_seq, target_value]: feature_seq shape: (batch_size,time_step,input_dim); targe_value shape: a list of feature_seq length  
		###	 hidden_num: number of hidden units for LSTM block   
		super(Regressor_v1, self).__init__()
		self.data_num=len(data[0])
		input_dim=np.shape(data[0])[2]
		# self.data=[Variable(torch.FloatTensor(data[0])),Variable(torch.FloatTensor(data[1]))]
		self.data=data
		self.LSTM_h1=40
		self.Linear1_h1=20 

		self.LSTM1=nn.LSTM(input_size=input_dim,hidden_size=self.LSTM_h1,num_layers=1,batch_first=True)
		# self.RNN1=nn.RNN(input_size=input_dim,hidden_size=self.LSTM_h1,num_layers=2,batch_first=True)
		self.Linear1=nn.Linear(in_features=self.LSTM_h1,out_features=self.Linear1_h1)
		self.Linear2=nn.Linear(in_features=self.Linear1_h1,out_features=1)

		# self.ReLU=nn.ReLU()
		self.Sigmoid=nn.Sigmoid()

		# self.optimizer = optim.SGD(self.parameters(), lr=0.01)
		self.optimizer = optim.RMSprop(self.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
		self.loss = nn.MSELoss()


	# def forward(self,data): 
	# 	### data: features (no target value)
	# 	LSTM1,_=self.LSTM1(data)
	# 	LSTM1_last=LSTM1[:,-1,:]
	# 	# RNN1,_=self.RNN1(data)
	# 	# RNN1_last=RNN1[:,-1,:]
	# 	# print('LSTM1_shape',np.shape(LSTM1[0][-1]))
	# 	# print('LSTM1_last_shape',np.shape(LSTM1_last))
	# 	Sigmoid1=self.Sigmoid(LSTM1_last)		
	# 	Linear1=self.Linear1(Sigmoid1)
	# 	Sigmoid2=self.Sigmoid(Linear1)		
	# 	Linear2=self.Linear2(Sigmoid2)
	# 	# ReLU1=self.ReLU(Linear1)
	# 	output_f=Linear2
	# 	return output_f


	def forward(self,data): 
		# print('shape',np.shape(data))
		LSTM1,_=self.LSTM1(data)
		# print ('LSTM shape',np.shape(LSTM1))
		LSTM1_last=LSTM1[:,-1,:]
		# print('forward output shape',np.shape(LSTM1_last))
		# print('forward output',LSTM1_last)
		# Sigmoid1=self.Sigmoid(LSTM1_last)		
		Linear1=self.Linear1(LSTM1_last)
		Sigmoid2=self.Sigmoid(Linear1)		
		Linear2=self.Linear2(Sigmoid2)
		output_f=Linear2
		# print ('output shape',np.shape(output_f))
		return output_f		


	def mse_loss(self,pred,gt):
		# print ('mean',torch.mean((pred-gt)**2))
		return torch.mean((pred-gt)**2)

	def train_network(self,epochs,batch_size):
		itera=int(np.ceil(np.true_divide(self.data_num,batch_size)))
		# print ('itera',itera)
		left_data_num=self.data_num-(itera-1)*batch_size
		# print (left_data_num)
		for i in np.arange(epochs):
			print ('epoch',i)
			train_ord=np.random.permutation(self.data_num)
			feature_rand=[self.data[0][n] for n in train_ord]
			n2_gt_rand=[self.data[1][n] for n in train_ord]
			# feature_rand=self.data[0]
			# n2_gt_rand=self.data[1]
			# print (np.shape(feature_rand),np.shape(n2_gt_rand))
			# print (n2_gt_rand[-1])
			for j in np.arange(itera):
				self.zero_grad()
				feature_feed=[] ### init feature_feed
				n2_gt_feed=[] ### init n2_gt_feed
				if j==itera-1: 
					feature_feed=Variable(torch.FloatTensor(feature_rand[-left_data_num:]))	
					n2_gt_feed=Variable(torch.FloatTensor(n2_gt_rand[-left_data_num:]))
					# print ('feature_feed',feature_feed[-1])
					# print ('n2_gt_feed',n2_gt_feed[-1])
				else:
					# print ('feature_rand_shape',np.shape(feature_rand[0:10]))
					# print ('j,batch_size',j,batch_size)
					feature_feed=Variable(torch.FloatTensor(feature_rand[j*batch_size:(j+1)*batch_size]))
					n2_gt_feed=Variable(torch.FloatTensor(n2_gt_rand[j*batch_size:(j+1)*batch_size]))
				output=self.forward(feature_feed)
				output=torch.squeeze(output)
				# print ('output',output)
				# print ('n2_gt_feed',n2_gt_feed)
				# print ('mse_loss',self.mse_loss(output,n2_gt_feed))
				err=self.loss(output,n2_gt_feed)#self.mse_loss(output,n2_gt_feed)
				# print ('feature_feed_shape',np.shape(feature_feed))
				# print ('output_shape',np.shape(output))
				# print ('n2_gt_feed_shape',np.shape(n2_gt_feed))
				err.backward()
				self.optimizer.step()
			print(err)
			# print ('finished')

	
	def prediction(self,data):
	### data: features, no target
	### output: prediction resutls shape: [# of points,1]
		feature_feed=Variable(torch.FloatTensor(data))
		return self.forward(feature_feed) 	



# class mse_loss(Function):
# 	def forward(self,input):


	 
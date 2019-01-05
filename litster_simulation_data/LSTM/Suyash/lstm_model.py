import data_loader_csv
import pdb
import numpy as np
import argparse
import torch.nn as nn 
import sys
import torch.optim as optim
import torch
import torch.optim as optim
from torch.distributions import Categorical
import network,os,datetime
from torch.utils.data.dataset import Dataset
import network


folder_to_save = os.path.join(os.path.join('Weights'+'/'+str(datetime.datetime.now().time()).split('.')[0].replace(':','_')))
if not os.path.exists(folder_to_save):
		os.makedirs(folder_to_save)
results_file_name = os.path.join(folder_to_save,'stats.txt')
results_file = open(results_file_name,'w')



class CustomDataSet(Dataset):

	def __init__ (self,X,Y):

	
		self.X,self.Y = X,Y
		# if (self.mode=='train'):
		#     self.X,self.Y = data_loader.load(mode='train')
		# elif (self.mode=='eval'):
		#     self.X,self.Y = data_loader.load(mode='eval')
		# else:
		#     self.X,_= data_loader.load(mode='test')

		print(self.X.shape)

	def __getitem__(self, index):


		return self.X[index],self.Y[index]



	def __len__(self):
		return self.X.shape[0]


class FuelModel(nn.Module):

	
	def __init__(self,lr,network_parameters,seq_len,batch_size):
		super(FuelModel,self).__init__()
		
	
		input_features = network_parameters['input_size'] 
		self.hidden_size = network_parameters['hidden_size']
		num_layers = network_parameters ['num_layers'] 

		self.batch_size =batch_size
		self.seq_len = seq_len
		self.input_features = input_features

		input_size = (seq_len,batch_size,input_features)

		self.rnn = nn.LSTM(input_size=input_features,hidden_size=self.hidden_size,num_layers=num_layers,bidirectional=True,batch_first=True,dropout=0.3)

		self.linear1 = nn.Linear(2*self.hidden_size*self.seq_len,1)
		# self.linear2 = nn.Linear(128,128)
		# self.linear3 = nn.Linear(128,output_size)


		self.optimizer = torch.optim.Adam(self.parameters(),lr=0.001)


	def forward(self,ret_batch): #L x N
		# returns 3D logits

		X = ret_batch
		# Y = ret_batch[1].cuda()
		# X_len = ret_batch[2].cuda()
		# Y_len = ret_batch[3].cuda()


		# X = R.pack_padded_sequence(X,X_len,batch_first=True)

		# embed = self.embedding(X) #L x N x E
		# hidden = None

		X = X.reshape(self.seq_len,X.shape[0],self.input_features)

		output_lstm,hidden = self.rnn(X) #L x N x H

		output_lstm_flatten = output_lstm.reshape(-1,self.seq_len*2*self.hidden_size)
		linear1 = self.linear1(output_lstm_flatten)
		# linear2 = self.linear2(linear1)
		# linear3 = self.linear3(linear2)



		return linear1








def parse_arguments():
	parser = argparse.ArgumentParser(description='Network Parser')
	parser.add_argument('--lr',type=float,default=0.01)
	parser.add_argument('--batch_size',type=int,default=32)
	parser.add_argument('--Ne',type=int,default=200)
	parser.add_argument('--load_path',type=str,default=None)
	parser.add_argument('--seq_len',type=int,default=3)

	return parser.parse_args()


def main(args):
	args = parse_arguments()
	lr = args.lr
	batch_size = args.batch_size
	n_epochs = args.Ne
	load_path = args.load_path
	seq_len = args.seq_len

	network_parameters = {}

	network_parameters['input_size'] = 7
	network_parameters['hidden_size'] = 40
	network_parameters['num_layers'] = 2


	net = FuelModel(lr,network_parameters,seq_len,batch_size)


	if (load_path is not None):
		print (load_path)
		network.load_net(os.path.join(os.getcwd(),load_path),net)
		print ('loaded weights')

	if torch.cuda.is_available():
		net  = net.cuda()
		print ('Using GPU')


	trainX,trainY = data_loader_csv.data_return(mode='train',batch_size = batch_size ,lstm=True,seq_len=seq_len)
	evalX,evalY = data_loader_csv.data_return(mode='eval',batch_size = batch_size ,lstm=True,seq_len=seq_len)
	testX,testY = data_loader_csv.data_return(mode='test',batch_size = batch_size ,lstm=True,seq_len=seq_len)


	total_data_len = trainX.shape[0] + evalX.shape[0] + testX.shape[0]


	train_data = CustomDataSet(trainX,trainY)
	eval_data = CustomDataSet(evalX,evalY)
	test_data = CustomDataSet(testX,testY)


	print ('Datapoints: {}'.format(train_data.__len__()))
	train_loader = torch.utils.data.DataLoader(train_data, 
										   batch_size,True)
	eval_loader = torch.utils.data.DataLoader(eval_data, 
										  1,False)
	test_loader = torch.utils.data.DataLoader(test_data,1,False)

	total_batch_ids = train_data.__len__()/batch_size

	results_file.write('Sequence Length: '+ str(seq_len) + '\n')
	print ('Training commencing')

	train(train_loader,eval_loader,batch_size,lr,net,n_epochs,total_batch_ids)

	print ('done')

	test(test_loader,net,testY)
	np.save(os.path.join('./'+folder_to_save,'AcutalY'),testY)

def train(train_loader,eval_loader,batch_size,lr,net,n_epochs,total_batch_ids):

		
		criterion = nn.MSELoss()       
		net = net.double()
		net = net.cuda()
		training_loss = np.zeros(n_epochs)
		valid_loss = np.zeros(n_epochs)

		for e in range(n_epochs):
			correct = 0
			epoch_loss = 0
			for batch_id,X in enumerate(train_loader):
				net.train()
				if torch.cuda.is_available():
					X[0] = X[0].cuda()
					X[1] = X[1].cuda()
					net.cuda()
				
				# print (type(X[0]))
				# exit()


				output = net.forward(X[0])
				train_loss = criterion(output,X[1].unsqueeze(1)).cuda()

		   

				net.optimizer.zero_grad()
				train_loss.backward()
				net.optimizer.step()

				epoch_loss += train_loss.item()
		  

			   
			epoch_loss /= ((batch_id+1))
			print ('Epoch No {} Training Loss {} '.format(e+1,epoch_loss))

			print('Evaluate')
			val_loss = 0
			correct_eval = 0
			
			training_loss[e] = epoch_loss

			for batch_id_eval,X_eval in enumerate(eval_loader):
				net.eval()
				if torch.cuda.is_available():
					X_eval[0] = X_eval[0].cuda()
					X_eval[1] = X_eval[1].cuda()
					net.cuda()

				output = net.forward(X_eval[0])
				eval_loss = criterion(output,X_eval[1].unsqueeze(0)).cuda()


				val_loss += eval_loss.item()

			val_loss/=(batch_id_eval+1)
			valid_loss[e] = val_loss
			print ('Epoch No {} validation Loss {} '.format(e+1,val_loss))
			if (e%4==0):
				save_name = os.path.join('./'+folder_to_save, 'epoch_'+str(e+1)+'.h5')
				network.save_net(save_name,net)
				print('Saved model to {}'.format(save_name))
			results_file.write('Epoch Number: '+str(e+1)+','+'Training Loss: '+str(epoch_loss)+','+'Validation Loss: '+str(val_loss)+','+'\n')
			results_file.flush()
		np.save(os.path.join('./'+folder_to_save,'training_loss.npy'),training_loss)
		np.save(os.path.join('./'+folder_to_save,'valid_loss.npy'),valid_loss)
		


def test(test_loader,net,testY):
		criterion = nn.MSELoss()       
		net.eval()
		net = net.cuda()
		net.double()
		correct_test = 0
		predictions = []
		with torch.no_grad():
			score =0 
			for batch_idx, x in enumerate(test_loader):

				x[0] = x[0].cuda()
				x[1] = x[1].cuda()
				output = net.forward(x[0])
				test_loss = criterion(output,x[1].unsqueeze(0)).cuda()
				score+= np.abs(output.item()-testY[batch_idx])
				predictions.append(output.item())


		score /= len(predictions)
		print ('Testing Loss {}'.format(score))
		results_file.write('Testing MAE {}'.format(score))
		np.save(os.path.join('./'+folder_to_save,'predictions.npy'),predictions)
		results_file.close()

if __name__ == '__main__':
	main(sys.argv)


import sys 
sys.path.insert(0,'../LSTM')


import numpy as np 
import data_loader as dl 

from sklearn import neighbors

import matplotlib.pyplot as plt  



file_path='../data/'
file_name='SAIC_Purging_Simulation_n2.xlsx'
time,data=dl.excel_load(file_path,file_name) 
N2=data[-1]
print ('N2_size',np.shape(N2))

#### data correlation filtering #### 
coeff_th=0.995
feature_flted,I2_remove=dl.feature_correlation_removal(data,coeff_th,1,0)
print ('feature_flted_shape',np.array(feature_flted).T)


#### stadnardize data without gt #### 
processed_feature,train_std,train_mean=dl.standardize_data(feature_flted,1,0,0)
print ('processed_data:',np.shape(processed_feature))
print ('train_std',train_std)
print ('train_mean',train_mean)

 

# features=list(np.array(processed_feature).T) ### make first dim data pt

features=[[processed_feature[0][n],processed_feature[1][n],processed_feature[2][n],processed_feature[3][n],processed_feature[4][n],processed_feature[5][n],processed_feature[6][n]] for n in np.arange(len(N2))] ### make first dim data pt
print ('features shape',np.shape(features)) 
### use input feature + previously predicted N2 concentration 
data_X=[[features[i][0],features[i][1],features[i][2],features[i][3],features[i][4],features[i][5],features[i][6],N2[i-1]] for i in np.arange(1,len(N2))]
data_Y=N2[1:]

### use input feature + previously predicted N2 concentration 
# data_X=[[features[i][0],features[i][1],features[i][2],features[i][3],features[i][4],features[i][5],features[i][6]] for i in np.arange(0,len(N2)-1)]
# data_Y=[N2[i]-N2[i-1] for i in np.arange(1,len(N2))]






########### get test dataset ##############
file_path='../data/'
file_name='SAIC_Purging_Simulation_n3.xlsx'
time,data=dl.excel_load(file_path,file_name) 
N2=data[-1]
print ('N2_size',np.shape(N2))

#### data correlation filtering #### 
coeff_th=0.995
feature_flted,_=dl.feature_correlation_removal(data,coeff_th,0,I2_remove)
print ('feature_flted_shape',np.array(feature_flted).T)


#### stadnardize data without gt #### 
processed_feature,train_std,train_mean=dl.standardize_data(feature_flted,0,train_std,train_mean)
print ('processed_data:',np.shape(processed_feature))
print ('train_std',train_std)
print ('train_mean',train_mean)

 

# features=list(np.array(processed_feature).T) ### make first dim data pt

#for k in np.arange(1,50):
features=[[processed_feature[0][n],processed_feature[1][n],processed_feature[2][n],processed_feature[3][n],processed_feature[4][n],processed_feature[5][n],processed_feature[6][n]] for n in np.arange(len(N2))] ### make first dim data pt
print ('features shape',np.shape(features)) 

data_X_test=[[features[i][0],features[i][1],features[i][2],features[i][3],features[i][4],features[i][5],features[i][6],N2[i-1]] for i in np.arange(1,len(N2))]
data_Y_test=N2[1:]

### use input feature + previously predicted N2 concentration 

# data_X_test=[[features[i][0],features[i][1],features[i][2],features[i][3],features[i][4],features[i][5],features[i][6]] for i in np.arange(0,len(N2)-1)]
# data_Y_test=[N2[i]-N2[i-1] for i in np.arange(1,len(N2))]



########## get testset ends ################

k=2
MSE=[] 
knn_pred=[]
n_neighbors=k
knn = neighbors.KNeighborsRegressor(n_neighbors)
knn_fit = knn.fit(data_X,data_Y)
for i in np.arange(1,len(N2)):
	if i==1:
		print (data_X_test[0])
		knn_pred.append(knn_fit.predict([data_X_test[0]]))
	else:
		data_X_test=[features[i][0],features[i][1],features[i][2],features[i][3],features[i][4],features[i][5],features[i][6],knn_pred[i-2]] ### include predicted value, for accumulation test 
		# data_X_test=[features[i][0],features[i][1],features[i][2],features[i][3],features[i][4],features[i][5],features[i][6]]  ### exclude previous prediction
		knn_pred.append(knn_fit.predict([data_X_test]))

mse=np.divide(np.sum(np.abs(np.array(knn_pred)-np.array(data_Y_test))),len(data_Y_test))
# mse=np.divide(np.linalg.norm(np.array(knn_pred)-np.array(data_Y_test)),len(data_Y_test))
print ('k=',k,'MAE',mse)
MSE.append(mse)


### reconstruction #### 
concen=[0]
i=0
for elem in knn_pred: 
	concen.append(elem+concen[i])
	i+=1

print ('test_data_range',np.max(data_Y_test)-np.min(data_Y_test))



plt.figure(1)
plt.plot(np.arange(len(data_Y_test)),data_Y_test,'bo-')
plt.plot(np.arange(len(data_Y_test)),knn_pred,'ro-')
plt.title('The prediction of the N2 concentration',fontsize=16)
plt.xlabel('time sequence',fontsize=16)
plt.ylabel('The difference of N2 concentration',fontsize=16)
plt.legend(['ground truth','Prediction'],prop={'size':16})


plt.figure(2)
plt.plot(np.arange(1,len(MSE)+1),MSE,'bo-')
plt.title('The effect of k value')
plt.xlabel('k value')
plt.ylabel('MAE loss')


plt.figure(3)
plt.plot(data_Y_test,data_Y_test,'bo')
plt.plot(data_Y_test,knn_pred,'ro')
plt.xlabel('ground truth')
plt.ylabel('Prediction')
plt.legend(['ground truth','Prediction'])


plt.figure(4)
plt.plot(np.arange(len(concen)),N2,'bo-')
plt.plot(np.arange(len(concen)),concen,'ro-')
plt.title('N2 concnetration',fontsize=16)
plt.xlabel('time',fontsize=16)
plt.ylabel('N2 concentration',fontsize=16)
plt.legend(['ground truth','Prediction'],prop={'size':16})




plt.show()


# print (data_X[0])
# print (features[0])
# print ('data_X shape',np.shape(data_X)) 


# print ('feautre_shape',np.shape(features)) 
# total_data=len(features)
# print ('total_data',total_data)


# time_step=10






# ### LSTM data set (comment out) ###
# batch_size=2813
# epochs=1000000


# feature_seqs=[features[i:i+time_step] for i in np.arange(total_data-time_step+1)] 
# N2_gt=[N2[i] for i in np.arange(time_step-1,total_data)]
# print ('LSTM input feature:',np.shape(feature_seqs))
# print ('output:',np.shape(N2_gt))






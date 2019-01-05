import pandas as pd 
import numpy as np



def excel_load(file_path,file_name):
	### path: path of the data folder 
	### name: the name of the file 
	### return: time coln (list (# of data pts,)) and data list (number of features, number of data pts)
	df = pd.read_excel(file_path + file_name)

	data=[]; ### init data matrix 
	time=[]; ### init time 
	i=0

	#print (len(df['Time (s)'])) 
	total_num=len(df['Time (s)'])

	for header in df.keys():
		#print (header) 
		if i==0: 
			time=[df[header][n] for n in np.arange(total_num)]
		else: 
			INPUT=[df[header][n] for n in np.arange(total_num)]
			data.append(INPUT)
		i+=1
	print ('size of time:',np.shape(time))	
	print ('size of data:',np.shape(data)) 
	return time,data 


###### test ###########
# file_path='../data/'
# file_name='SAIC_Purging_Simulation_n2.xlsx'
# time,data=excel_load(file_path,file_name) 
# data_array=np.array(data)
# print ('data_array',np.shape(data_array))
# print ('coeff_matrix',np.corrcoef(data_array))
#######################

def feature_correlation_removal(data,coeff_th,train_set,I2remove):
	### data: original data with target value 
	### coeff_th: correlation coefficient threshold  
	### train_Set: 1, using training set; 0, using test set 
	### I2remove: Index 2 remove (for test set only) 
	features=data[:-1] ### remove target value 
	#N2=data[-1]
	feature_num=len(features)
	if train_set==1:
		feature_array=np.array(features)
		coeff_matrix=np.corrcoef(feature_array)

		### get rid of correlations ###
		#corr_th= 0.995
		I2remove= [] 
		j=0
		Inputs=list(np.arange(feature_num)) ## -1 to exclude gt 
		for coeffs in coeff_matrix: 
			# print (Inputs)
			Inputs.remove(j) ## remove self 
			for i in Inputs: 
				if coeffs[i] >coeff_th:
					I2remove.append(i)
			
			print (Inputs)
			j+=1
		print ('Removed features due to high correlations',I2remove)
		I2remove_final=set(I2remove) 
	else:
		I2remove_final=I2remove
	feature_coe_remove=[data[i] for i in np.arange(feature_num) if i not in I2remove_final]
	return feature_coe_remove,I2remove_final


def standardize_data(data,train_set,train_std,train_mean):
	### data: input data list , shape (feature_type,# of pts)
	#### train_set: 1 if process training set 
	#### return: standardized data, original data's mean and std
	input_types_num=len(data)
	if train_set ==1:
		mean=[np.mean(one_type) for one_type in data]
		std=[np.std(one_type) for one_type in data]
		# print ('train_mean',train_mean)
		# print('train_std',train_std)
		processed_data=[np.true_divide(data[i]-mean[i],std[i])+mean[i] for i in np.arange(input_types_num)]
	else: ### if test set, use train std and train mean 
		processed_data=[np.true_divide(data[i]-train_mean[i],train_std[i])+train_mean[i] for i in np.arange(input_types_num)]		
		std=0 ## trivial
		mean=0  ## trivial 
	return processed_data,std,mean    

#### test ####
# processed_data,train_std,train_mean=standardize_train_feature_include_y(data)
# print (processed_data,train_std,train_mean)
# std=[np.std(one_type) for one_type in processed_data[:-1]]
# train_mean=[np.mean(one_type) for one_type in data[:-1]]
# print (std,train_mean)


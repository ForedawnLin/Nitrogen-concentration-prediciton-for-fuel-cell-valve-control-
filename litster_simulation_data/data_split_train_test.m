clc
clear all


% fileName='data/data_11_28.xlsx'; 
% M=readtable(fileName); 
% 
% %%%% organize data %%% 
% table_size=size(M);
% X_feature=table_size(2)-1; %%% didn't include data (first coln) 
% data_num=table_size(1); %%% total number of data 

% 
% %%% omit coln 7 and 9 (anode inlet RH and anode gas velocity)
% D1=M{:,2:6};
% D2=M{:,8};
% D3=M{:,10:11};
% data_matrix=[D1 D2 D3]; 
% train_test_ratio=4; 
% number=floor(data_num*train_test_ratio/(train_test_ratio+1)); 
% train_data=data_matrix(1:number,:);
% test_data=data_matrix(number+1:end,:);
% data=struct();
% data.train_data=train_data;
% data.test_data=test_data;
% save('data/data_sim1.mat','data'); 
% 



%%% load data %%% 
% File=load('data/data_sim1.mat'); 
% train_data=File.data.train_data;
% 
% feature_mean=mean(train_data(:,1:end-1));  %%% -1 due to last one is target 
% feature_std=std(train_data(:,1:end-1));
% train_data(:,1:end-1)=(train_data(:,1:end-1)-feature_mean)./(feature_std);


%%%% load feature matrix %%%%
% FILE=load('data/Mining_raw_data.mat'); 
% feature_matrix=FILE.feature_matrix; %%% 23 colns feature matrix (the last coln is the target value) 
% FILE2=load('data/time_raw_data.mat'); 
% time=FILE2.time; 

%%% group data based on the same time points %%%
% [index,group]=findgroups(time);
% 
% FM_meaned=splitapply(@mean,feature_matrix,index); %%% mean each group 
% 
% %%% Split into train and test set
% 
% 
% %%% train/test 
% matrix_size=size(FM_meaned); 
% data_num=matrix_size(1); 
% train_test_ratio=4; 
% number=floor(data_num*train_test_ratio/(train_test_ratio+1)); 
% train_data=FM_meaned(1:number,:);
% test_data=FM_meaned(number+1:end,:);
% save('data/train_data','train_data'); 
% save('data/test_data','test_data');

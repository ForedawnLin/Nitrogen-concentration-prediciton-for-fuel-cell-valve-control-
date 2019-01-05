clc 
clear all 

%% load train data %%%
fileName_train='data/SAIC_Purging_Simulation_n2.xlsx'; 
M_train=readtable(fileName_train);
train_data_us=M_train{:,:};

%% load test data %%%
fileName_test='data/SAIC_Purging_Simulation_n3.xlsx'; 
M_test=readtable(fileName_test);
test_data=M_test{:,:}; 



%%% remove high correlation data %%% 
data2pro=train_data_us(:,2:end-1);
corr_th=0.995; %%% corrlaton threshold 
[new_data,index_left]=remove_high_corr(data2pro,corr_th);
train_data_2=[new_data train_data_us(:,end)]; %%% train_data_2 exluded first coln (time)
SIZE=size(train_data_us); 
input_dim=SIZE(2); 
index_left=[index_left+1 input_dim]; %%% features that have been removed , +1 due to removal of time 
test_data_2=test_data(:,index_left);


%%%% standardlization without y%%%%
[train_data_3,train_mean,train_std]=standlization(train_data_2(:,1:end-1),1,0,0); %%% 0,0, trivial for train set 
train_data_us_final=[train_data_3 train_data_us(:,end)];
[test_data_3,~,~]=standlization(test_data_2(:,1:end-1),0,train_mean,train_std);
test_data_final=[test_data_3 test_data(:,end)];


params=lwpparams('COS',3,0,10,0,0,0,1);
train_Y=train_data_us_final(:,end);
train_feature=[train_data_3(2:end,:) train_Y(1:end-1,:)]; 
test_feature=[test_data_3(2:end,:) test_data(1:end-1,end)];
% train_feature=train_data_3;
% train_Y=train_data_us(:,end);
% test_feature=test_data_3;
[Y_p]=lwppredict(train_feature,train_Y(2:end,:),params,test_feature);

% 
% figure(1)
% plot(1:length(test_data(:,end)),test_data(:,end),'bo-')
% hold on
% plot(1:length(test_data(:,end)),Y_p,'ro-')
% ylim([-1,3])
% 

figure(1)
plot(1:length(test_data(2:end,end)),test_data(2:end,end),'bo-')
hold on
plot(1:length(test_data(2:end,end)),Y_p(1:end),'ro-')
%ylim([-1,3])
title('Locally Weighted Regression')
xlabel('Sequence')
ylabel('N2 comcentration')
legend('Ground truth','Prediction')


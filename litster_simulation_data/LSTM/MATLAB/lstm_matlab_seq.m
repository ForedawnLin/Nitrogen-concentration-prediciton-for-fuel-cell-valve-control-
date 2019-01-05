clc 
clear all 

%% load train data %%%
fileName_train='data/SAIC_Purging_Simulation_n2.xlsx'; 
M_train=readtable(fileName_train);
train_data=M_train{:,:};


%% load test data %%%
fileName_test='data/SAIC_Purging_Simulation_n3.xlsx'; 
M_test=readtable(fileName_test);
test_data=M_test{:,:};




%%% remove high correlation data %%% 
data2pro=train_data(:,2:end-1);
corr_th=0.995; %%% corrlaton threshold 
[new_data,index_left]=remove_high_corr(data2pro,corr_th);
train_data_2=[new_data train_data(:,end)]; %%% train_data_2 exluded first coln (time)
SIZE=size(train_data); 
input_dim=SIZE(2); 
index_left=[index_left+1 input_dim]; %%% features that have been removed , +1 due to removal of time 
test_data_2=test_data(:,index_left);


%%%% standardlization without y%%%%
[train_data_3,train_mean,train_std]=standlization(train_data_2(:,1:end-1),1,0,0); %%% 0,0, trivial for train set 
train_data_us_final=[train_data_3 train_data(:,end)];
[test_data_3,~,~]=standlization(test_data_2(:,1:end-1),0,train_mean,train_std);
test_data_final=[test_data_3 test_data(:,end)];


%%%% data final orgnization %%%%
Y_train=train_data_us_final(:,end);
  %%% use N2 diff as target %%%
% Y_train=Y_train(2:end)-Y_train(1:end-1); 
% Y_train=[0;Y_train]*10; 
  %%% ends %%%%
train_feature=train_data_us_final(:,1:end-1);


Y_test=test_data_final(:,end);   
  %%% use N2 diff as target %%%
% Y_test=Y_test(2:end)-Y_test(1:end-1); 
% Y_test=[0;Y_test]*10;
  %%% ends %%%%
test_feature=test_data_final(:,1:end-1);



train_data_SIZE=size(train_feature); 
n_sample_train=train_data_SIZE(1); 
input_dim=train_data_SIZE(2); 
test_data_SIZE=size(test_feature); 
n_sample_test=test_data_SIZE(1); 


%%% add time %%%
% added_feature_train=1:n_sample_train; 
% added_feature_test=1:n_sample_test;
% train_feature=[added_feature_train' train_feature]; 
% test_feature=[added_feature_test' test_feature]; 
%%% add time ends %%%


%%%%%%%% arrange data for LSTM %%%%%%%%
time_step=100; 
%%% arrange train data seq %%%%
train_LSTM_in=cell(n_sample_train-time_step+1,1); %%% n_sample_data cells, with input_dim*time_step for each cell  
for i=1:n_sample_train-time_step+1 
    train_LSTM_in{i}=train_feature(i:i+time_step-1,:)';
end 

train_LSTM_Y=cell(n_sample_train-time_step+1,1); 
for i=1:n_sample_train-time_step+1 
    train_LSTM_Y{i}=Y_train(i:i+time_step-1)';
end 

%%% arrange test data seq %%%%
test_LSTM_in=cell(n_sample_test-time_step+1,1); %%% n_sample_data cells, with input_dim*time_step for each cell  
for i=1:n_sample_test-time_step+1 
    test_LSTM_in{i}=test_feature(i:i+time_step-1,:)';
end 


test_LSTM_Y=cell(n_sample_test-time_step+1,1); 
for i=1:n_sample_test-time_step+1 
    test_LSTM_Y{i}=Y_test(i:i+time_step-1)';
end 



%%%% train LSTM %%%% 
maxEpochs = 500; %% 50
miniBatchSize = 64;

numFeatures = 7;
LSTMUnits1 = 512;
LSTMUnits2 = 128;
hidden1=256;
hidden2=128;
numResponses = time_step;



layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(LSTMUnits1,'OutputMode','sequence')
%     lstmLayer(LSTMUnits2,'OutputMode','sequence')
    fullyConnectedLayer(hidden1)
    fullyConnectedLayer(hidden2)
    fullyConnectedLayer(1) 
    regressionLayer];



options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'CheckpointPath','checkpoint', ...
    'Plots','training-progress');


net = trainNetwork(train_LSTM_in,train_LSTM_Y,layers,options);


%%% test on train dataset %%%%
YPred=predict(net,train_LSTM_in);
YPred_final=[];
gt_final=[]; 
for i=1:length(YPred) 
    YPred_final=[YPred_final YPred{i}(end)];
    gt_final=[gt_final train_LSTM_Y{i}(end)];
end 
figure (1) 
plot(1:length(train_LSTM_Y),gt_final,'bo-'); 
hold on 
plot(1:length(train_LSTM_Y),YPred_final,'ro-');  
title('N2 concentration prediction')
xlabel('sequence')
ylabel('N2 concentration')
legend('ground truth','prediction')


%% test data set %%
YPred=predict(net,test_LSTM_in);
YPred_final=[];
gt_final=[]; 
for i=1:length(YPred) 
    YPred_final=[YPred_final YPred{i}(end)];
    gt_final=[gt_final test_LSTM_Y{i}(end)];
end 
figure (1) 
plot(1:length(test_LSTM_Y),gt_final,'bo-'); 
hold on 
plot(1:length(test_LSTM_Y),YPred_final,'ro-');  
title('N2 concentration prediction')
xlabel('sequence')
ylabel('N2 concentration')
legend('ground truth','prediction')





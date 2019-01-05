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


%%%% upsampling data %%%%
sample_freq=0.05;
train_data_us=upSampling(train_data,sample_freq); 
size(train_data_us) ;


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


%%%% data final orgnization %%%%
Y_train=train_data_us_final(:,end);
train_feature=train_data_us_final(:,1:end-1);
Y_test=test_data_final(:,end);    
test_feature=test_data_final(:,1:end-1);



train_data_SIZE=size(train_feature); 
n_sample_train=train_data_SIZE(1); 
input_dim=train_data_SIZE(2); 
test_data_SIZE=size(test_feature); 
n_sample_test=test_data_SIZE(1); 



%%%%%%%% arrange data for LSTM %%%%%%%%
time_step=10; 
%%% arrange train data seq %%%%
train_LSTM_in=cell(n_sample_train-time_step+1,1); %%% n_sample_data cells, with input_dim*time_step for each cell  
for i=1:n_sample_train-time_step+1 
    train_LSTM_in{i}=train_feature(i:i+time_step-1,:)';
end 

train_LSTM_Y=Y_train(time_step:end); 

%%% arrange test data seq %%%%
test_LSTM_in=cell(n_sample_test-time_step+1,1); %%% n_sample_data cells, with input_dim*time_step for each cell  
for i=1:n_sample_test-time_step+1 
    test_LSTM_in{i}=test_feature(i:i+time_step-1,:)';
end 
test_LSTM_Y=Y_test(time_step:end); 



%%%% train LSTM %%%% 
maxEpochs = 50;
miniBatchSize = 32;

numFeatures = 7;
numHiddenUnits1 = 50;
numHiddenUnits2 = 70;
numResponses = 1;



layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits1,'OutputMode','sequence')
    lstmLayer(numHiddenUnits2,'OutputMode','last')
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(train_LSTM_in,train_LSTM_Y,layers,options);


%%% test on train dataset %%%%
YPred=predict(net,train_LSTM_in);
figure (1) 
plot(1:length(train_LSTM_Y),train_LSTM_Y,'bo-'); 
hold on 
plot(1:length(train_LSTM_Y),YPred,'ro-');  






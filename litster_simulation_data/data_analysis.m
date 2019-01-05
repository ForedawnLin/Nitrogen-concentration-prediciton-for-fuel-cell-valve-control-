clc
clear all



%%%% data pre process %%%%
File=load('data/data_sim1.mat'); 
train_data=File.data.train_data;
test_data=File.data.test_data;
Y_train=train_data(:,end);
%%% standilization
% feature_mean=mean(train_data(:,1:end));
% feature_std=std(train_data(:,1:end));
% train_data(:,1:end)=(train_data(:,1:end)-feature_mean)./(feature_std);
% test_data(:,1:end)=(test_data(:,1:end)-feature_mean)./(feature_std);

%%% plot time relation %%%
figure (1)
plot(Y_train(1:end-1),Y_train(2:end),'o-')
title('Y_{(t+1)} v.s Y_{t}')
xlabel('N2 concentraion at t')
ylabel('N2 concentraion at t+1')

%% plot Y %%
figure (2)
plot(1:length(Y_train),Y_train,'-')
title('N2 concentration v.s time')
ylabel('N2 concentraion (scaled)')
xlabel('Sequence')

%% plot voltage v.s. Current %%
figure (3)
plot(train_data(:,1),train_data(:,2),'-')
title('Voltage v.s Current')
ylabel('Voltage(V)')
xlabel('Current(I)')


%% plot voltage v.s. Temp %%
figure (4)
plot(train_data(:,3),train_data(:,4),'-')
title('Voltage v.s Current')
ylabel('Voltage(V)')
xlabel('Current(I)')



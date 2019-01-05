clc 
clear all

[train_Y,train_prediction,test_Y,test_prediction,train_input_feature,test_input_feature,prediction_states,prediction_states_train] = ModelInfer_I_O_HMM_one_input('I_O_HMM_model/I_O_HMM_T10_B4.mat',10,4);
%[train_Y,train_prediction,test_Y,test_prediction] = ModelInfer_I_O_HMM_one_input('I_O_HMM_one_input/I_O_HMM_T3_B4_STD.mat',3,4);

MAE_train=sum(abs(train_Y'-train_prediction))/length(train_prediction)
MAE_test=sum(abs(test_Y'-test_prediction))/length(test_prediction)
max_test=max(test_Y)
min_test=min(test_Y)
range_test=max_test-min_test

% %%%% collect data for model stacking %%%% 
% data=struct(); 
% data.train_Y=train_Y; 
% data.train_prediction=train_prediction'; 
% data.train_input_feature=train_input_feature;
% data.test_Y=test_Y;
% data.test_prediction=test_prediction';
% data.test_input_feature=test_input_feature;
% save("I_O_HMM_one_input/Stack_Data_train_prediction_T3_B4.mat","data")


%%%% plot %%%%
figure (3)
plot(1:length(train_Y),train_Y,'-ob');
hold on; 
plot(1:length(train_Y),train_prediction,'-or'); 
title('Groud truth and predction (training data)'); 
xlabel('time sequence'); 
ylabel('value'); 
legend('Ground truth','Prediction ')


figure (4) 
plot(1:length(test_Y),test_Y,'-ob');
hold on; 
plot(1:length(test_Y),test_prediction,'-or'); 
title('Groud truth and predction (test data)'); 
xlabel('time sequence'); 
ylabel('value'); 
legend('Ground truth','Prediction ')


figure (5)
plot(train_Y,train_prediction,'bo');

figure (6) 
plot(test_Y,test_prediction,'bo');
hold on;
plot(-1:6,-1:6,'r');

title('Groud truth v.s predction (test data)'); 
xlabel('prediction'); 
ylabel('ground truth'); 
legend('Ground truth v.s prediction','optimal prediction')

clc 
clear all 

%% load train data %%%
fileName_train='data/SAIC_Purging_Simulation_n3.xlsx'; 
M_train=readtable(fileName_train);
train_data=M_train{:,:};


%% load test data %%%
fileName_test='data/SAIC_Purging_Simulation_n3.xlsx'; 
M_test=readtable(fileName_test);
test_data=M_test{:,:}; 








% 
% N2_diff=train_data(2:end,11)-train_data(1:end-1,11);
% N2_diff=[train_data(1,11);N2_diff];
% 
% 
% %%% plot time v.s all featues and N2 
figure(1) 

subplot(3,3,1)
plot(1:length(train_data(:,1)),train_data(:,2));
title('voltage')
xlabel('Time (s)')
ylabel('voltage (V)')


subplot(3,3,2)
plot(1:length(train_data(:,1)),train_data(:,3));
title('Current')
xlabel('Time (s)')
ylabel('Current density (A/m^2)')


subplot(3,3,3)
plot(1:length(train_data(:,1)),train_data(:,4));
title('Temperature')
xlabel('Time (s)')
ylabel('Temperature (K)')


subplot(3,3,4)
plot(1:length(train_data(:,1)),train_data(:,5));
title('Cathode Inlet Relative Humidity(RH)')
xlabel('Time (s)')
ylabel('Inlet RH')


subplot(3,3,5)
plot(1:length(train_data(:,1)),train_data(:,7));
title('Anode Inlet Relative Humidity(RH)')
xlabel('Time (s)')
ylabel('Inlet RH')


subplot(3,3,6)
plot(1:length(train_data(:,1)),train_data(:,8));
title('Anode Outlet Relative Humidity(RH)')
xlabel('Time (s)')
ylabel('Outlet RH')


subplot(3,3,7)
plot(1:length(train_data(:,1)),train_data(:,9));
title('Anode gas velocity(m/s)')
xlabel('Time (s)')
ylabel('Velocity (m/s)')



subplot(3,3,8)
plot(1:length(train_data(:,1)),train_data(:,10));
title('Purge vavlue open amount')
xlabel('Time (s)')
ylabel('Open amount %')


subplot(3,3,9) 
plot(1:length(train_data(:,1)),train_data(:,11));
title('N2 concentration')
xlabel('Time (s)')
ylabel('N2 concentration (mol/m^3)')


% 
% 
% %%%% plot N2 v.s feature %%%% 
% figure(2)
% 
% subplot(3,3,1) 
% plot(train_data(:,2),train_data(:,11),'o');
% title('voltage vs N2')
% 
% subplot(3,3,2) 
% plot(train_data(:,3),train_data(:,11),'o');
% title('Current vs N2')
% 
% subplot(3,3,3) 
% plot(train_data(:,4),train_data(:,11),'o');
% title('Temp vs N2')
% 
% subplot(3,3,4) 
% plot(train_data(:,5),train_data(:,11),'o');
% title('Temp vs N2')
% 
% subplot(3,3,5) 
% plot(train_data(:,6),train_data(:,11),'o');
% title('Catho vs N2')
% 
% subplot(3,3,6) 
% plot(train_data(:,7),train_data(:,11),'o');
% title('Anode vs N2')
% 
% subplot(3,3,7) 
% plot(train_data(:,9),train_data(:,11),'o');
% title('Vel vs N2')
% 
% 
% subplot(3,3,8) 
% plot(train_data(:,10),train_data(:,11),'o');
% title('Purge vs N2')
% 
% 
% figure (3) 
% plot3(train_data(:,10),train_data(:,3),train_data(:,11),'o-')
% xlabel('purge')
% ylabel('Current')
% zlabel('N2')
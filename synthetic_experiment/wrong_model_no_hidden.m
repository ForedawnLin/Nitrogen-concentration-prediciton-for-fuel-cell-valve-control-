clc 
clear all
%%% Note: %%% 



%%%%%%% synthetic data generation %%%%%%%
data_gen=relation();

%%% randomly generate observable parameters %%%
randseed ;
n_sample=1000;
H2O=rand(n_sample,1);
T=rand(n_sample,1)*10/3+10;
C_N2C=rand(n_sample,1)/10;
C_N2A=rand(n_sample,1)/10+1.05;
P_H2=rand(n_sample,1)*1000+300;
P_air=rand(n_sample,1)*1000+500;

trainNum=ceil(0.8*n_sample); 

H2O_train=H2O(1:trainNum);
T_train=T(1:trainNum);
C_N2C_train=C_N2C(1:trainNum);
C_N2A_train=C_N2A(1:trainNum);
P_H2_train=P_H2(1:trainNum);
P_air_train=P_air(1:trainNum);


%n_sample=800;



%% generat N2 result %%
for i=1:trainNum
N2_train(i)=data_gen.data_generation_static(H2O_train(i),T_train(i),C_N2C_train(i),C_N2A_train(i),P_H2_train(i),P_air_train(i)); 
end 
N2_train=N2_train';
obs_values=[H2O_train T_train C_N2C_train C_N2A_train P_H2_train P_air_train N2_train]' ;  




%%% create BN %%% 
%%% A: H20 B:T C:CN2C D:CN2A E:PressureC  F:PressureA
A=1;B=2;C=3;D=4;E=5;F=6;K=7;  %%% observed variable 


n_node=7;
ns=ones(1,n_node); 


dag=zeros(n_node); 

dag(A,K)=1;
dag(B,K)=1; 
dag(C,K)=1;
dag(D,K)=1;
dag(E,K)=1;
dag(F,K)=1;




bnet=mk_bnet(dag,ns,'discrete',[],'observed',[A B C D E F K]); 
seed=0; 
rand('state',seed); 

bnet.CPD{A}=root_CPD(bnet,A);
bnet.CPD{B}=root_CPD(bnet,B);
bnet.CPD{C}=root_CPD(bnet,C);
bnet.CPD{D}=root_CPD(bnet,D);
bnet.CPD{E}=root_CPD(bnet,E);
bnet.CPD{F}=root_CPD(bnet,F);
bnet.CPD{K}=gaussian_CPD(bnet,K,'mean',[0.5],'cov',[0.01]);


% 
% 
% 
% 
% %%% create samples for BN %%%
samples=cell(n_node,trainNum); 

for i=1:trainNum
    samples([1 2 3 4 5 6 7],i)=num2cell(obs_values(:,i));
end 


engine=jtree_inf_engine(bnet); 
max_iter=1000; 
%[bnet2]=learn_params(bnet,samples);
%plot(LLtrace,'x-');
[bnet2,LLtrace]=learn_params_em(engine,samples,max_iter);






%%%% Infernece %%%% 

engine=jtree_inf_engine(bnet2); 
evidence=cell(1,n_node); 



%%% get tests from first sampling %%%%

H2O_test=H2O(n_sample-trainNum:end);
T_test=T(n_sample-trainNum:end);
C_N2C_test=C_N2C(n_sample-trainNum:end);
C_N2A_test=C_N2A(n_sample-trainNum:end);
P_H2_test=P_H2(n_sample-trainNum:end);
P_air_test=P_air(n_sample-trainNum:end);

testNum=n_sample-trainNum;


for i=1:testNum
evidence{A}=H2O_test(i);
evidence{B}=T_test(i);
evidence{C}=C_N2C_test(i);
evidence{D}=C_N2A_test(i);
evidence{E}=P_H2_test(i);
evidence{F}=P_air_test(i);

[engine,ll]=enter_evidence(engine,evidence); 
marg=marginal_nodes(engine,K);
y_pred(i)=marg.mu; 
N2_test(i)=data_gen.data_generation_static(H2O_test(i),T_test(i),C_N2C_test(i),C_N2A_test(i),P_H2_test(i),P_air_test(i)); 
end 
% 
% 


%%% get tests from re-sampling %%%%

H2O_test=rand(n_sample,1);
T_test=rand(n_sample,1)*10/3+10;
C_N2C_test=rand(n_sample,1)/10;
C_N2A_test=rand(n_sample,1)/10+1.05;
P_H2_test=rand(n_sample,1)*1000+300;
P_air_test=rand(n_sample,1)*1000+500;






testNum=n_sample;


for i=1:testNum
evidence{A}=H2O_test(i);
evidence{B}=T_test(i);
evidence{C}=C_N2C_test(i);
evidence{D}=C_N2A_test(i);
evidence{E}=P_H2_test(i);
evidence{F}=P_air_test(i);

[engine,ll]=enter_evidence(engine,evidence); 
marg=marginal_nodes(engine,K);
y_pred(i)=marg.mu; 
N2_test(i)=data_gen.data_generation_static(H2O_test(i),T_test(i),C_N2C_test(i),C_N2A_test(i),P_H2_test(i),P_air_test(i)); 
end 








figure (2)
hold on 
plot(1:testNum,N2_test,'b'); 
plot(1:testNum,y_pred,'r');
title("No causality") 
xlabel('sequence number') 
ylabel('value') 
legend('Ground truth','Prediction')
hold off 
MAE=sum(abs(y_pred-N2_test))/testNum;

%%% ends %%%


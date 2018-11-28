clc
clear all

n_sample=1000
a=3;
randseed 
x=rand(1,n_sample)*10;
y=x; 
% for i=1:n_sample 
%     k=a+randn(1)
%     y(i)=k*x(i);
% end 


obs_values=[x;y];

A=1;B=2;C=3;D=4;  
n_node=4;
sft_state=2;
ns=ones(1,n_node); 
%ns(3)=sft_state;
%ns(5)=2;
ns(3)=sft_state;
ns(4)=sft_state;

dag=zeros(n_node); 
dag(A,D)=1; 
dag(B,C)=1;
dag(C,D)=1;


bnet=mk_bnet(dag,ns,'discrete',[C D],'observed',[D]); 
seed=0; 
rand('state',seed); 

bnet.CPD{A}=gaussian_CPD(bnet,A,'cov_type','diag');
bnet.CPD{B}=gaussian_CPD(bnet,B,'cov_type','diag');
bnet.CPD{C}=softmax_CPD(bnet,C,'clamped',0, 'max_iter', 10);
bnet.CPD{D}=softmax_CPD(bnet,D,'clamped',0, 'max_iter', 10);


%bnet.CPD{C}=softmax_CPD(bnet,C,'clamped',0, 'max_iter', 10);
%bnet.CPD{D}=gaussian_CPD(bnet,D,'cov',[0.01 0.01 0.01]);%,'clamped',0, 'max_iter', 10);
%bnet.CPD{E}=softmax_CPD(bnet,E,'clamped',0, 'max_iter', 10);


%%% create samples for BN %%%
% samples=cell(n_node,n_sample);
% 
% for i=1:n_sample
%     samples([1 2],i)=num2cell(obs_values(:,i));
% end 
% 
% 
% engine=jtree_inf_engine(bnet); 
% max_iter=1000; 
% epsilon=0.0000001;
% [bnet2,LLtrace]=learn_params_em(engine,samples,max_iter,epsilon);
%plot(LLtrace,'x-');
% %bnet2=learn_params(engine,samples);
% 
% % 
% % 
% % %%%% Inference %%%%
% engine = jtree_inf_engine(bnet2); 
% evidence=cell(1,n_node);
% n_test_sample=1000;
% x2=rand(1,n_test_sample)*10;
% for i=1:length(x2)
%     evidence{A}=x2(i);
%     %evidence{B}=y(i);
%     [engine,ll]=enter_evidence(engine,evidence); 
%     marg=marginal_nodes(engine,B);
%     %mpe=find_mpe(engine,evidence);
%     y_pred(i)=marg.mu;
%     %y_pred(i)=mpe{2};
% end 
% 


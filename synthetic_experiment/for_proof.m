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

A=1;B=2; 
n_node=2;
sft_state=3;
ns=ones(1,n_node); 
%ns(3)=sft_state;
%ns(5)=2;
%ns(2)=2;

dag=zeros(n_node); 
dag(A,[B])=1; 
%dag(C,B)=1;

bnet=mk_bnet(dag,ns,'discrete',[],'observed',[A B]); 
seed=0; 
rand('state',seed); 


bnet.CPD{A}=root_CPD(bnet,A);
bnet.CPD{B}=gaussian_CPD(bnet,B,'cov_type','diag');
%bnet.CPD{C}=softmax_CPD(bnet,C,'clamped',0, 'max_iter', 10);
%bnet.CPD{D}=gaussian_CPD(bnet,D,'cov',[0.01 0.01 0.01]);%,'clamped',0, 'max_iter', 10);
%bnet.CPD{E}=softmax_CPD(bnet,E,'clamped',0, 'max_iter', 10);


%%% create samples for BN %%%
samples=cell(n_node,n_sample);

for i=1:n_sample
    samples([1 2],i)=num2cell(obs_values(:,i));
end 


engine=jtree_inf_engine(bnet); 
max_iter=1000; 
epsilon=0.0000001;
[bnet2,LLtrace]=learn_params_em(engine,samples,max_iter,epsilon);
%plot(LLtrace,'x-');
%bnet2=learn_params(engine,samples);



%%% get MLE parameters %%%
% CPT=cell(1,n_node);
% for i=1:n_node
%     s=struct(bnet.CPD{i});
%     CPT{2*i-1}=[s.mean];
%     CPT{2*i}=[s.cov];
% end 
% 
% nsamples=100;
% result=cell(n_node,nsamples);
% for i=1:nsamples
%     result(:,i)=sample_bnet(bnet2)
% end 


% data_sample=cell2num(result); 
% x_sample=data_sample(1,:);
% y_sample=data_sample(2,:);
% % 
% figure (2)
% plot(x,y);
% hold on 
% plot(x_sample,y_sample,'+');
% hold off 
% 
% 
% %%%% Inference %%%%
engine = jtree_inf_engine(bnet2); 
evidence=cell(1,n_node);
n_test_sample=1000;
x2=rand(1,n_test_sample)*10;
for i=1:length(x2)
    evidence{A}=x2(i);
    %evidence{B}=y(i);
    [engine,ll]=enter_evidence(engine,evidence); 
    marg=marginal_nodes(engine,B);
    %mpe=find_mpe(engine,evidence);
    y_pred(i)=marg.mu;
    %y_pred(i)=mpe{2};
end 


%%% extract parameters %%%
% x2=rand(1,n_sample)*10; %%% 1*n 
% x2=[ones(1,length(x2));x2]; %%% 2*n
% 
% 
% s=struct(bnet2.CPD{3}); 
% eta=[s.glim{1}.b1; s.glim{1}.w1]'; %%% 4*2 matrix  (4 cases for w and b) 
% s=struct(bnet2.CPD{2});
% W=reshape(s.weights,[1 sft_state]); %%% 1*4, bias 
% theta=[s.mean; W]' %%%% 4*2 , 4 casese for mean and weights 
% pr=exp(eta*x2); %% 4*n ; 
% pr=pr./sum(pr); %% 4*n
% y_pred=theta*x2; %%% 4*n for 4 cases of gaussian mean of y 
% y_pred=sum(pr.*y_pred);%%% 1*n, soft mean of y weighted by output of softmax 



figure (1)
%plot(x,y,'o');
hold on 
plot(x2,y_pred,'r+');
plot(x2,x2,'b');
title("Prediction of y=x") 
xlabel('x') 
ylabel('y') 
legend('Prediction','Ground truth')
hold off 




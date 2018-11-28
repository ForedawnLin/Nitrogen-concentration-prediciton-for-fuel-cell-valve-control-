% clc
% clear all
% 
% n_sample=1000
% a=3;
% randseed 
% x1=[rand(1,n_sample)*10];
% x2=[rand(1,n_sample)];
% w=[1 1];
% y=w*[x1;x2]; 
% % for i=1:n_sample 
% %     k=a+randn(1)
% %     y(i)=k*x(i);
% % end 
% 
% 
% obs_values=[x1;x2;y];
% 
% A=1;B=2;C=3;D=4; 
% n_node=4;
% sft_state=3;
% ns=ones(1,n_node); 
% ns(4)=sft_state; 
% %ns(3)=sft_state;
% %ns(5)=2;
% %ns(2)=2;
% 
% dag=zeros(n_node); 
% dag(A,[C D])=1; 
% dag(B,[C D])=1;
% dag(D, C)=1; 
% 
% bnet=mk_bnet(dag,ns,'discrete',[D],'observed',[A B C]); 
% seed=0; 
% rand('state',seed); 
% 
% 
% bnet.CPD{A}=root_CPD(bnet,A);
% bnet.CPD{B}=root_CPD(bnet,B);
% bnet.CPD{C}=gaussian_CPD(bnet,C,'mean',[0 0 0],'cov',[0.01 0.01 0.01],'cov_type','diag');
% bnet.CPD{D}=softmax_CPD(bnet,D,'clamped',0, 'max_iter', 10);
% %bnet.CPD{C}=softmax_CPD(bnet,C,'clamped',0, 'max_iter', 10);
% %bnet.CPD{D}=gaussian_CPD(bnet,D,'cov',[0.01 0.01 0.01]);%,'clamped',0, 'max_iter', 10);
% %bnet.CPD{E}=softmax_CPD(bnet,E,'clamped',0, 'max_iter', 10);
% 
% 
% %%% create samples for BN %%%
% samples=cell(n_node,n_sample);
% 
% for i=1:n_sample
%     samples([1 2 3],i)=num2cell(obs_values(:,i));
% end 
% 
% 
% engine=jtree_inf_engine(bnet); 
% max_iter=1000; 
% epsilon=0.0000001;
% [bnet2,LLtrace]=learn_params_em(engine,samples,max_iter,epsilon);
% %plot(LLtrace,'x-');
% %bnet2=learn_params(engine,samples);
% 
% 
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
% 
% %%% extract parameters %%%
% % x2=rand(1,n_sample)*10; %%% 1*n 
% % x2=[ones(1,length(x2));x2]; %%% 2*n
% % 
% % 
% % s=struct(bnet2.CPD{3}); 
% % eta=[s.glim{1}.b1; s.glim{1}.w1]'; %%% 4*2 matrix  (4 cases for w and b) 
% % s=struct(bnet2.CPD{2});
% % W=reshape(s.weights,[1 sft_state]); %%% 1*4, bias 
% % theta=[s.mean; W]' %%%% 4*2 , 4 casese for mean and weights 
% % pr=exp(eta*x2); %% 4*n ; 
% % pr=pr./sum(pr); %% 4*n
% % y_pred=theta*x2; %%% 4*n for 4 cases of gaussian mean of y 
% % y_pred=sum(pr.*y_pred);%%% 1*n, soft mean of y weighted by output of softmax 
% 
% 
% 
% figure (1)
% %plot(x,y,'o');
% hold on 
% plot(x2,y_pred,'r+');
% plot(x2,x2,'b');
% title("Prediction of y=x") 
% xlabel('x') 
% ylabel('y') 
% legend('Prediction','Ground truth')
% hold off 
% 
% 
% 














clc
clear all

n_sample=1000
randseed 
x=[rand(1,n_sample)*10;rand(1,n_sample)];
w=[1 1];
y=w*x+3; 
% for i=1:n_sample 
%     k=a+randn(1)
%     y(i)=k*x(i);
% end 

obs_values=[x;y];

A=1;B=2;C=3 
n_node=3;
sft_state=2;
ns=ones(1,n_node); 
ns(1)=2; 
ns(2)=sft_state; 



dag=zeros(n_node); 
dag(A,[B C])=1; 
dag(B,[C])=1;

bnet=mk_bnet(dag,ns,'discrete',[B],'observed',[A C]); 
seed=0; 
rand('state',seed); 


bnet.CPD{A}=root_CPD(bnet,A);
bnet.CPD{B}=softmax_CPD(bnet,B,'clamped',0, 'max_iter', 10);
bnet.CPD{C}=gaussian_CPD(bnet,C,'mean',[0 0],'cov',[0.01 0.01],'cov_type','diag');
%bnet.CPD{C}=softmax_CPD(bnet,C,'clamped',0, 'max_iter', 10);
%bnet.CPD{D}=gaussian_CPD(bnet,D,'cov',[0.01 0.01 0.01]);%,'clamped',0, 'max_iter', 10);
%bnet.CPD{E}=softmax_CPD(bnet,E,'clamped',0, 'max_iter', 10);


%%% create samples for BN %%%
samples=cell(n_node,n_sample);

for i=1:n_sample
    samples(1,i)={obs_values(1:2,i)};
    samples(3,i)={obs_values(3,i)};    
    %samples([1 2 3],i)=num2cell(obs_values(:,i));
end 


engine=jtree_inf_engine(bnet); 
max_iter=1000; 
epsilon=0.0000001;
[bnet2,LLtrace]=learn_params_em(engine,samples,max_iter,epsilon);
%plot(LLtrace,'x-');
%bnet2=learn_params(engine,samples);





% clc
% clear all
% 
% n_sample=1000;  
% a=3;
% randseed 
% x1=[rand(1,n_sample)*10];
% t=1:length(x1); 
% w=[1];
% y=w.*x1.*t; 
% % for i=1:n_sample 
% %     k=a+randn(1)
% %     y(i)=k*x(i);
% % end 
% 
% 
% obs_values=[x1;y];
% 
% 
% %%%% mk DBN 
% intra=zeros(2); 
% intra(1,2)=1; %%% node 1 in slice t connects to node 2 in slice t 
% 
% inter=zeros(2); 
% inter(1,1)=1; %%% node 2 in slice t-1 connects to node 2 in slice t 
% 
% ns=[1 1]; %%% all 1-D cnt nodes 
% 
% %eclass1=[1 2]; 
% %eclass2=[3 2];
% 
% 
% % A=1;B=2;C=3;D=4; 
% % n_node=4;
% % sft_state=3;
% % ns=ones(1,n_node); 
% % ns(4)=sft_state; 
% %ns(3)=sft_state;
% %ns(5)=2;
% %ns(2)=2;
% 
% % dag=zeros(n_node); 
% % dag(A,[C D])=1; 
% % dag(B,[C D])=1;
% % dag(D, C)=1; 
% 
% bnet=mk_dbn(intra,inter,ns,'discrete',[],'observed',[2]); 
% seed=0; 
% rand('state',seed); 
% 
% 
% %bnet.CPD{1}=root_CPD(bnet,1);
% for i=1:4
%     bnet.CPD{i}=gaussian_CPD(bnet,1,'mean',[0],'cov',[0.01],'cov_type','diag');
%     %bnet.CPD{2}=gaussian_CPD(bnet,2,'mean',[0],'cov',[0.01],'cov_type','diag');
%     %bnet.CPD{2}=gaussian_CPD(bnet,2,'mean',[0],'cov',[0.01],'cov_type','diag');
% end 
% %bnet.CPD{3}=gaussian_CPD(bnet,3,'mean',[0 ],'cov',[0.01],'cov_type','diag');
% %bnet.CPD{D}=softmax_CPD(bnet,D,'clamped',0, 'max_iter', 10);
% %bnet.CPD{C}=softmax_CPD(bnet,C,'clamped',0, 'max_iter', 10);
% %bnet.CPD{D}=gaussian_CPD(bnet,D,'cov',[0.01 0.01 0.01]);%,'clamped',0, 'max_iter', 10);
% %bnet.CPD{E}=softmax_CPD(bnet,E,'clamped',0, 'max_iter', 10);
% 
% 
% %%% create samples for BN %%%
% %samples=cell(n_node,n_sample);
% 
% n_sequence=1; 
% for i=1:n_sequence  
%     samples{i}=cell(2,length(x1));
%     samples{i}(2,:)=num2cell(obs_values([2],:));
% end 
% 
% % for i=1:n_sample
% %     samples([1 2 3],i)=num2cell(obs_values(:,i));
% % end 
% 
% 
% engine=jtree_dbn_inf_engine(bnet); 
% max_iter=10; 
% epsilon=0.0000001;
% [bnet2,LLtrace]=learn_params_dbn_em(engine,samples,'max_iter',10);
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

%%% generate data %%%
n_sample=1000;  
randseed 
x1=[rand(1,n_sample)*10];
t=1:length(x1);
t=sqrt(0.1*t);
w=[1];
y=w.*x1.*t; 
obs_values=[x1;y];



%%% structure %%%
intra=zeros(2);
intra(1,2)=1;

inter=zeros(2); 
inter(2,2)=1; 

eclass1=[1 2]; 
eclass2=[1 3]; 
eclass=[eclass1 eclass2]; 


Q=1;
O=1; 


ns=[Q O]; 
dnodes=[];  
bnet=mk_dbn(intra,inter,ns,'discrete',dnodes,'observed',[2],'eclass1',eclass1,'eclass2',eclass2); 


% bnet.CPD{1}=tabular_CPD(bnet,1,'CPT',[0.1 0.9]); 
% bnet.CPD{2}=tabular_CPD(bnet,2,'CPT',[0.3 0.7 0.4 0.6]);
% bnet.CPD{3}=tabular_CPD(bnet,3,'CPT',[0.2 0.8 0.8 0.2]);
% bnet.CPD{4}=tabular_CPD(bnet,4,'CPT',[0.3 0.7 0.4 0.6]); 

%bnet.CPD{1}=tabular_CPD(bnet,1,'CPT',[0.5 0.2 0.3]); 
%bnet.CPD{3}=tabular_CPD(bnet,3,'CPT',[0.5 0.5]); 
bnet.CPD{1}=gaussian_CPD(bnet,1,'mean',[0],'cov',[0.01],'cov_type','diag');
%bnet.CPD{1}=root_CPD(bnet,1);
bnet.CPD{2}=gaussian_CPD(bnet,2,'weights',0.1,'cov_type','diag');
bnet.CPD{3}=gaussian_CPD(bnet,3,'weights',[],'cov_type','diag');
%bnet.CPD{3}=tabular_CPD(bnet,3,'CPT',[0.1 0.3 0.9 0.7]);
%bnet.CPD{3}=tabular_dCPD(bnet,3,'CPT',[0.2 0.6 0.8 0.4]);
%bnet.CPD{4}=gaussian_CPD(bnet,4,'mean',[1 2]);



ncases=1000;
cases=cell(1,ncases); 
T=5; 
for i=1:ncases 
    data=sample_dbn(bnet,T);
    %start=randi(n_sample-T-1); 
    %data=num2cell(obs_values(:,start:start+T-1)); 
    cases{i}=cell(2,T); 
    cases{i}(1:2,:)=data(1:2,:);
end 




%engine = hmm_inf_engine(bnet);
engine = jtree_dbn_inf_engine(bnet);
[bnet2,LLtrace]=learn_params_dbn_em(engine,cases,'max_iter',100,'thresh',0.001); 












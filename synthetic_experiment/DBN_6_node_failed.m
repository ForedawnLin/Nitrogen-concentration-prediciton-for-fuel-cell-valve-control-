
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
intra=zeros(3);
intra(1,[2 3])=1;
intra(2,3)=1;

inter=zeros(3); 
inter(2,2)=1; 


eclass1=[1 2 3]; 
eclass2=[1 4 3]; 
eclass=[eclass1 eclass2]; 



ns=[1 2 1]; 
dnodes=[2];  
bnet=mk_dbn(intra,inter,ns,'discrete',[dnodes],'observed',[1 2 3],'eclass1',eclass1,'eclass2',eclass2); 


% bnet.CPD{1}=tabular_CPD(bnet,1,'CPT',[0.1 0.9]); 
%bnet.CPD{2}=tabular_CPD(bnet,2,'CPT',[0.3 0.7]);
% bnet.CPD{3}=tabular_CPD(bnet,3,'CPT',[0.2 0.8 0.8 0.2]);
% bnet.CPD{4}=tabular_CPD(bnet,4,'CPT',[0.3 0.7 0.4 0.6]); 
bnet.CPD{1}=root_CPD(bnet,1);
bnet.CPD{2}=softmax_CPD(bnet,2,'clamped',0, 'max_iter', 10);
%bnet.CPD{1}=tabular_CPD(bnet,1,'CPT',[0.5 0.2 0.3]); 
%bnet.CPD{3}=tabular_CPD(bnet,3,'CPT6',[0.5 0.5]); 
%bnet.CPD{1}=root_CPD(bnet,1);
%bnet.CPD{2}=gaussian_CPD(bnet,2,'cov_type','diag');
bnet.CPD{3}=gaussian_CPD(bnet,3,'cov_type','diag');

%bnet.CPD{4}=gaussian_CPD(bnet,4,'cov_type','diag'); %%% emission 
%bnet.CPD{3}=tabular_CPD(bnet,3,'CPT',[0.1 0.3 0.9 0.7]);
%bnet.CPD{4}=tabular_CPD(bnet,4);
%bnet.CPD{4}=gaussian_CPD(bnet,4,'mean',[1 2]);
%bnet.CPD{4}=gaussian_CPD(bnet,4,'cov_type','diag');

bnet.CPD{4}=softmax_CPD(bnet,5,'clamped',0, 'max_iter', 10);
% bnet.CPD{4}=mlp_CPD();



ncases=1000;
cases=cell(1,ncases); 
T=5; 
for i=1:ncases 
    %data=sample_dbn(bnet,T);
    %start=randi(n_sample-T-1); 
    %data=num2cell(obs_values(:,start:start+T-1));
    data=num2cell([1:T;1:T;1:T])
    cases{i}=cell(3,T); 
    cases{i}([1 3],:)=data([1 3],:);
end 




%engine = hmm_inf_engine(bnet); %% can't use
%engine = jtree_dbn_inf_engine(bnet); %%% can't use
engine = jtree_unrolled_dbn_inf_engine(bnet,T);
%engine = smoother_engine(jtree_2TBN_inf_engine(bnet));
[bnet2,LLtrace]=learn_params_dbn_em(engine,cases,'max_iter',100,'thresh',0.001); 












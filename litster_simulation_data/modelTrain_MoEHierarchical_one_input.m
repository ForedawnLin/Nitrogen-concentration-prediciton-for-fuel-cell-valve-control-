clc
clear all



%%%% data pre process %%%%
File=load('data/data_sim1.mat'); 
train_data=File.data.train_data;
test_data=File.data.test_data;

% SIZE=size(train_data);
% index=randperm(SIZE(1));
% train_data=train_data(index,:);

%%%% get rid of I,T, Cathode inlet RH %%%%
train_data=[train_data(:,1) train_data(:,5:end)] 
test_data=[test_data(:,1) test_data(:,5:end)] 


%%%%%% standlization %%%%%%%%
feature_mean=mean(train_data(:,1:end));
feature_std=std(train_data(:,1:end));
train_data(:,1:end)=(train_data(:,1:end)-feature_mean)./(feature_std);
test_data(:,1:end)=(test_data(:,1:end)-feature_mean)./(feature_std);


Y_train=train_data(:,end);
train_feature=train_data(:,1:end-1)+0.01;
Y_test=test_data(:,end);    
test_feature=test_data(:,1:end-1);


% Y_train=train_data(:,end);
% train_feature=train_data(:,1:end-1);
% [u,s,v]=svd(train_feature);
% train_feature=train_feature*v;
% train_feature=train_feature(:,1:end-1)*10+0.01; %% +0.01 due to singularity 
% feature_mean2=mean(train_feature);
% feature_std2=std(train_feature);
% train_feature=(train_feature-feature_mean2)./(feature_std2);
% 
% Y_test=test_data(:,end);    
% test_feature=test_data(:,1:end-1)*v(:,1:end-1)*10+0.01;
% 
% test_feature=(test_feature-feature_mean2)./(feature_std2);
        





train_data_SIZE=size(train_feature); 
n_sample_train=train_data_SIZE(1); 
input_dim=train_data_SIZE(2); 
test_data_SIZE=size(test_feature); 
n_sample_test=test_data_SIZE(1); 




%%% EM train setting %%%
max_iter=1000;  %%% max iteration 
epsilon=0.01; %%% max error 



%%% create BN %%%
A=1;  %%% input 
B=2;D=4; %%% hidden units 
C=3;  %%% output 
% 
n_node=4; 
ns=ones(1,n_node); 
ns(A)=input_dim; 
% ns(B)=6; %%% hidden state num 
% ns(D)=6; %%% hidden state num
% 
% 
% dag=zeros(n_node); 
% dag(A,[B C D])=1; 
% dag(B,[C D])=1;
% dag(D,C)=1; 
% 
% 
% bnet=mk_bnet(dag,ns,'discrete',[B D],'observed',[A C]); 
% seed=0; 
% rand('state',seed); 
% 
% bnet.CPD{A}=gaussian_CPD(bnet,A,'cov_type','diag');
% bnet.CPD{B}=softmax_CPD(bnet,B,'clamped',0, 'max_iter', 10);
% %bnet.CPD{C}=gaussian_CPD(bnet,C,'mean',[0 0 0],'cov','diag');  
% bnet.CPD{C}=gaussian_CPD(bnet,C,'cov_type','diag');
% bnet.CPD{D}=softmax_CPD(bnet,D,'clamped',0, 'max_iter', 10);



% %%% obtain data for BN %%%
% samples=cell(n_node,n_sample_train);
% for i=1:n_sample_train
%     node1_data=train_feature(i,:)';
%     node3_data=Y_train(i,:);
%     samples([1 3],i)={[node1_data];[node3_data]};
% end 


%% train BNT 
% engine=jtree_inf_engine(bnet); 
% [bnet2,LLtrace]=learn_params_em(engine,samples,max_iter,epsilon);



 
% save('EOMHierarchical_one_input/bnet2_B2_D4_STD.mat','bnet2');
% bnet2=load('EOMHierarchical_one_input/bnet2_B10_D6.mat'); 
% bnet2=bnet2.bnet2; 
% 
% %% Inferene 
% engine = jtree_inf_engine(bnet2); 
% evidence=cell(1,n_node);
% % n_test_sample=1000;
% % x2=rand(1,n_test_sample)*20;
% for i=1:n_sample_train
%     evidence{A}=[train_feature(i,:)'];
% %     evidence{A}=[test_data(i,1:9)'];
%     [engine,ll]=enter_evidence(engine,evidence); 
%     marg=marginal_nodes(engine,C);
%     %mpe=find_mpe(engine,evidence);
%     Y_pred_train(i)=marg.mu;
%     Y_eval_train(i)=Y_train(i);
%     
% end 
% 
% for i=1:n_sample_test
%     evidence{A}=[test_feature(i,:)'];
% %     evidence{A}=[test_data(i,1:9)'];
%     [engine,ll]=enter_evidence(engine,evidence); 
%     marg=marginal_nodes(engine,C);
%     %mpe=find_mpe(engine,evidence);
%     Y_pred_test(i)=marg.mu;
%     Y_eval_test(i)=Y_test(i);
% end 



%%% number of states; grid search 
% n_B=[2:10]; 
% n_D=[2:10]; 

for nB=2:2
    for nD=8:8
        ns(B)=nB; %%% hidden state num 
        ns(D)=nD; %%% hidden state num


        dag=zeros(n_node); 
        dag(A,[B C D])=1; 
        dag(B,[C D])=1;
        dag(D,C)=1; 


        bnet=mk_bnet(dag,ns,'discrete',[B D],'observed',[A C]); 
        seed=0; 
        rand('state',seed); 

        bnet.CPD{A}=gaussian_CPD(bnet,A,'cov_type','diag');
        bnet.CPD{B}=softmax_CPD(bnet,B,'clamped',0, 'max_iter', 10);
        %bnet.CPD{C}=gaussian_CPD(bnet,C,'mean',[0 0 0],'cov','diag');  
        bnet.CPD{C}=gaussian_CPD(bnet,C,'cov_type','diag');
        bnet.CPD{D}=softmax_CPD(bnet,D,'clamped',0, 'max_iter', 10);

        

        %%% obtain data for BN %%%
        samples=cell(n_node,n_sample_train);
        for i=1:n_sample_train
            node1_data=train_feature(i,:)';
            node3_data=Y_train(i,:);
            samples([1 3],i)={[node1_data];[node3_data]};
        end 


        %% train BNT 
        engine=jtree_inf_engine(bnet); 
        [bnet2,LLtrace]=learn_params_em(engine,samples,max_iter,epsilon);




        % save('EOMHierarchical_one_input/bnet2_B2_D4_STD.mat','bnet2');
        % bnet2=load('EOMHierarchical_one_input/bnet2_B6_D6_STD.mat'); 
        % bnet2=bnet2.bnet2; 

        %% Inferene 
        engine = jtree_inf_engine(bnet2); 
        evidence=cell(1,n_node);
        % n_test_sample=1000;
        % x2=rand(1,n_test_sample)*20;
        for i=1:n_sample_train
            evidence{A}=[train_feature(i,:)'];
        %     evidence{A}=[test_data(i,1:9)'];
            [engine,ll]=enter_evidence(engine,evidence); 
            marg=marginal_nodes(engine,C);
            %mpe=find_mpe(engine,evidence);
            Y_pred_train(i)=marg.mu;
            Y_eval_train(i)=Y_train(i);

        end 

        for i=1:n_sample_test
            evidence{A}=[test_feature(i,:)'];
        %     evidence{A}=[test_data(i,1:9)'];
            [engine,ll]=enter_evidence(engine,evidence); 
            marg=marginal_nodes(engine,C);
            %mpe=find_mpe(engine,evidence);
            Y_pred_test(i)=marg.mu;
            Y_eval_test(i)=Y_test(i);
        end
        %%% save path %%%
%         p1='EOMHierarchical_one_input/bnet2_B';
%         p2=num2str(nB);
%         p3='_D';
%         p4=num2str(nD);
%         p5='_STD.mat'; 
%         save([p1 p2 p3 p4 p5],'bnet2')
    end
end 

MAE_train=sum(abs(Y_eval_train-Y_pred_train))/length(Y_eval_train)
MAE_test=sum(abs(Y_pred_test-Y_eval_test))/length(Y_eval_test)
max_test=max(Y_eval_test)
min_test=min(Y_eval_test)
range_test=max_test-min_test


%%%% plot %%%%
figure (1) 
plot(1:n_sample_train,Y_eval_train,'bo'); 
hold on; 
plot(1:n_sample_train,Y_pred_train,'ro');
title('Ground truth and predction (training data)'); 
xlabel('time sequence'); 
ylabel('value'); 
legend('Ground truth','Prediction ')


figure (2) 
plot(1:n_sample_test,Y_eval_test,'bo'); 
hold on; 
plot(1:n_sample_test,Y_pred_test,'ro');
title('Ground truth and predction (test data)'); 
xlabel('time sequence'); 
ylabel('value'); 
legend('Ground truth','Prediction ')



figure (3) 
plot(Y_pred_test,Y_eval_test,'bo'); 
hold on;
plot(-1:6,-1:6,'r');
title('Ground truth v.s predction (test data)'); 
xlabel('prediction'); 
ylabel('ground truth'); 
legend('Ground truth v.s prediction','optimal prediction')
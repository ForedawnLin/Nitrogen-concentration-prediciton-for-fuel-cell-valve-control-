clc 
clear all 

%% load train data %%%
fileName_train='data/SAIC_Purging_Simulation_n2.xlsx'; 
M_train=readtable(fileName_train);
train_data=M_train{:,:};

%% load test data %%%
fileName_test='data/SAIC_Purging_Simulation_n3.xlsx'; 
M_test=readtable(fileName_test);
test_data=M_test{:,:};

%%%% upsampling data %%%%
sample_freq=0.05;
train_data_us=upSampling(train_data,sample_freq); 
size(train_data_us) ;

%%%%%%%%%%%%%% get region 1 %%%%%%%%%%%%%%%%%%%%
% %%% get regrion 1 data only %%%
% data=struct() ;
% data.region1_index_train=get_region_1(train_data);
% data.region1_index_test=get_region_1(test_data);
% data.train_data=train_data;
% data.test_data=test_data;
% 
% % save('data/region_1_only','data') 
% 
% train_data_us=[] ;
% for i =1:length(data.region1_index_train)
%     train_data_us=[train_data_us;data.train_data(data.region1_index_train{i},:)];
% end 
% 
% test_data=[] ;
% for i =1:length(data.region1_index_test)
%     test_data=[test_data;data.test_data(data.region1_index_test{i},:)] ;
% end 
%%%%%%%%%% get region 1 ends %%%%%%%%%%%%%%%%%


% figure (1)
% plot(1:length(train_data_us),train_data_us(:,11),'-o')





%%% remove high correlation data %%% 
data2pro=train_data_us(:,2:end-1);
corr_th=0.995; %%% corrlaton threshold 
[new_data,index_left]=remove_high_corr(data2pro,corr_th);
train_data_2=[new_data train_data_us(:,end)]; %%% train_data_2 exluded first coln (time)
SIZE=size(train_data_us); 
input_dim=SIZE(2); 
index_left=[index_left+1 input_dim]; %%% features that have been removed , +1 due to removal of time 
test_data_2=test_data(:,index_left);


%%%% standardlization without y%%%%
[train_data_3,train_mean,train_std]=standlization(train_data_2(:,1:end-1),1,0,0); %%% 0,0, trivial for train set 
train_data_us_final=[train_data_3 train_data_us(:,end)];
[test_data_3,~,~]=standlization(test_data_2(:,1:end-1),0,train_mean,train_std);
test_data_final=[test_data_3 test_data(:,end)];

%%%% increasing check %%%% 
% index_2_check=[train_data_us(1:end-1,11)-train_data_us(2:end,11)>0];  %%% if 0, good if 1, the end of the chuck 
% num_ck=sum(index_2_check); 

% Data_up=struct(); 
% Data_up.train=train_data_us
% Data_up.test=test_data
% save('data/upsampling_data_with_std.mat','Data_up')
% 



% 
% Data=load('data/data_sim1.mat');
% train_data=Data.data.train_data;
% test_data=Data.data.test_data;
% %%% get rid of I,T, Cathode inlet RH %%%%
% train_data=[train_data(:,1:3) train_data(:,5:end)];
% test_data=[test_data(:,1:3) test_data(:,5:end)];
% 
% 
% %%%%%% standlization %%%%%%%%
% feature_mean=mean(train_data(:,1:end-1));
% feature_std=std(train_data(:,1:end-1));
% train_data(:,1:end-1)=(train_data(:,1:end-1)-feature_mean)./(feature_std);
% test_data(:,1:end-1)=(test_data(:,1:end-1)-feature_mean)./(feature_std);
 

Y_train=train_data_us_final(:,end);
train_feature=train_data_us_final(:,1:end-1);
Y_test=test_data_final(:,end);    
test_feature=test_data_final(:,1:end-1);



% Y_train=train_data(:,end);
% train_feature=train_data(:,1:end-1);
% [u,s,v]=svd(train_feature);
% train_feature=train_feature*v;
% train_feature=train_feature(:,1:end-1)*10;
% 
% Y_test=test_data(:,end);
% test_feature=test_data(:,1:end-1)*v(:,1:end-1)*10;



train_data_SIZE=size(train_feature); 
n_sample_train=train_data_SIZE(1); 
input_dim=train_data_SIZE(2); 
test_data_SIZE=size(test_feature); 
n_sample_test=test_data_SIZE(1); 


%%%%%%%%%%%%%% if region 1 (use this for creating data seq)%%%%%%%%%%%%%%%
% T=3;
% n_nodes=3;
% A=1;B=2;C=3;  
% cases=cell(1,n_sample_train-T*num_ck); 
% k=1;
% y_truth=[];
% for i=1:length(index_2_check)+1-T
%     if ismember(1,index_2_check(i:i+T-1))==1
%         continue 
%     end 
%     cases{k}=cell(n_nodes,T);
%     for j=1:T
%         cases{k}(A,j)={train_feature(i+j-1,:)'}; 
%         cases{k}(C,j)={Y_train(i+j-1)};
%     end 
%     y_truth=[y_truth Y_train(i+T)]
%     k=k+1;
% end 
% 
% cases=cases(1:end-3); 

%%%%%%%%%%%%% use these to create data seq %%%%%%%%%%%%%%%%


for nT=3:3  %%% grid search for time step 
    for nB=2:2 %%% grid search for hidden state choices 

        %%% training settings 
        T=nT;  %%% look back step 
        max_iter=1000; %%% max iter to train 
        thresh_em=0.01; %%% EM threshold


        %%% I/O HMM structure %%% 
        A=1;B=2;C=3;  
        n_nodes=3; 
        intra=zeros(n_nodes); 
        intra(A,[B,C])=1;
        intra(B,C)=1; 
        ns=ones(1,n_nodes); 
        ns(A)=input_dim;
        ns(B)=nB; 
        dNodes=B; 
        oNodes=[A C];
        inter=zeros(n_nodes); 
        inter(B,B)=1; 

        %%% define CPDs for two-slice nodes and tie parameters %%%%
        eclass1=[1 2 3]; 
        eclass2=[1 4 3]; 
        elcass=[eclass1 eclass2]; 
        bnet=mk_dbn(intra,inter,ns,'discrete',dNodes,'observed',oNodes,'eclass1',eclass1,'eclass2',eclass2); 

        bnet.CPD{1}=gaussian_CPD(bnet,A,'cov_type','diag'); 
        %bnet.CPD{1}=root_CPD(bnet,A);
        bnet.CPD{2}=softmax_CPD(bnet,B,'clamped',0,'max_iter',10);
        bnet.CPD{3}=gaussian_CPD(bnet,C,'cov_type','diag','mean',[mean(Y_train),mean(Y_train)]);
        bnet.CPD{4}=softmax_CPD(bnet,5,'clamped',0,'max_iter',10);


        %data=sample_dbn(bnet,5) 


        % 
        %%% create sample %%%
%%%%%%%%%%%%%%%%% train data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        cases=cell(1,n_sample_train-T); 
        for i=1:n_sample_train-T 
            cases{i}=cell(n_nodes,T);
            for j=1:T
                cases{i}(A,j)={train_feature(i+j-1,:)'}; 
                cases{i}(C,j)={Y_train(i+j-1)};
            end 
        end 
%%%%%%%%%%%%%%%%% train end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%         cases=cell(1,n_sample_train-T);
%         rand_index=randperm(n_sample_train-T);
%         for i=1:n_sample_train-T
%             cases(rand_index(i))=cases_raw(i);
%         end 
        
        % T=5;
        % ncases=2;
        % cases=cell(1,ncases);
        % for i=1:ncases 
        %     ev=sample_dbn(bnet,T); 
        %     cases{i}=cell(3,T); 
        %     cases{i}(1,:)=ev(1,:); 
        %     cases{i}(3,:)=ev(3,:);  
        % end 



        %%% train DBN 
        %engine = jtree_dbn_inf_engine(bnet); 
        %engine = jtree_unrolled_dbn_inf_engine(bnet,T); 
        %engine = hmm_inf_engine(bnet,T);

        engine=smoother_engine(jtree_2TBN_inf_engine(bnet));
        [bnet2,LLtrace]=learn_params_dbn_em(engine,cases,'max_iter',max_iter,'thresh',thresh_em); 


        % FILE=load('I_O_HMM_one_input/I_O_HMM_T2_B3.mat'); 
        % bnet2=FILE.bnet2;
        % save('I_O_HMM_one_input/I_O_HMM_T2_B4','bnet2');
        % 

        infer_train_Y=cell(1,T); %%% initialize train target val for inference
        infer_train_Y(1,:)=cases{1}(3,:);
        for i=1:length(cases)
        %%%%%%%%% inference %%%%%%% 
        %%% input data %%%
            evidence=cell(3,T);
            evidence(1,:)=cases{i}(1,:); %%% no need to update input 
            evidence(3,:)=infer_train_Y(1,:);
            [engine,ll]=enter_evidence(engine,evidence);
        %%% inference %%%
            marg=marginal_nodes(engine,B,T); %%% node_num, time slice 


        %%%%%%%% prediction %%%%%%%%% 
            input_feature=train_feature(i+T,:);  %%% 1*m, will be input


            [~,T_th_state]=max(marg.T); %%% choose the T_th state 
            %%% get the corresponding CPD
            softmax_node_CPD= struct(bnet2.CPD{4});  
            softmax_set=softmax_node_CPD.glim{T_th_state}; 

            %%%calculate prediced state numb at t+1 
            softmax_element_values=input_feature*softmax_set.w1+softmax_set.b1;%%% no need to use full expression of softmax. since monotonic increase for each element and we only need the max  
            [~,predicted_state]=max(softmax_element_values); 

            %%% calculate predicted output  
            emission_CPD_set=struct(bnet2.CPD{3}); 
            emission_CPD_mean=emission_CPD_set.mean(predicted_state); % 1*1 
            emission_CPD_weight=emission_CPD_set.weights(:,:,predicted_state)'; %%% m*1
            predicted_val_train=input_feature*emission_CPD_weight+emission_CPD_mean;
            predicted_val_train_set(i)=predicted_val_train; %%% results  
            %%% keep rolling the observed input data 
            infer_train_Y(1)=[];
            infer_train_Y{T}=predicted_val_train; %%% output output value as previously observed output 
        end 






        %%% Big assumption: we know the first T points in test data set 





        infer_test_feature=cell(1,T); %%% initialize test feature for inference
        for i=1:T 
            infer_test_feature(i)={test_feature(i,:)'};
        end 

        infer_test_Y=cell(1,T); %%% initialize test target val for inference
        infer_test_Y(1,:)=num2cell(Y_test(1:T));



        for i=1:n_sample_test-T
        %%%%%%%%% inference %%%%%%% 
          %%% input data %%%
            evidence=cell(3,T);
            evidence(1,:)=infer_test_feature; %%% no need to update input 
            evidence(3,:)=infer_test_Y;
            [engine,ll]=enter_evidence(engine,evidence);
          %%% inference %%%
            marg=marginal_nodes(engine,B,T); %%% node_num, time slice 



            input_feature=test_feature(i+T,:);  %%% 1*m, will be input

            [~,T_th_state]=max(marg.T); %%% choose the T_th state 
            %%% get the corresponding CPD
            softmax_node_CPD= struct(bnet2.CPD{4});  
            softmax_set=softmax_node_CPD.glim{T_th_state}; 
           
            %%%calculate prediced state numb at t+1 
            softmax_element_values=input_feature*softmax_set.w1+softmax_set.b1;%%% no need to use full expression of softmax. since monotonic increase for each element and we only need the max  
            [~,predicted_state]=max(softmax_element_values); 

            %%% calculate predicted output  
            emission_CPD_set=struct(bnet2.CPD{3}); 
            emission_CPD_mean=emission_CPD_set.mean(predicted_state); % 1*1 
            emission_CPD_weight=emission_CPD_set.weights(:,:,predicted_state)'; %%% m*1
            predicted_val_test=input_feature*emission_CPD_weight+emission_CPD_mean;
            predicted_val_test_set(i)= predicted_val_test;
            %%% keep rolling the observed input data 
            infer_test_feature(1)=[]; 
            infer_test_feature{T}=test_feature(i+T,:)'; %%% get new input 
            infer_test_Y(1)=[];
            infer_test_Y{T}=predicted_val_test; %%% output output value as previously observed output    
        end 
%         p1='I_O_HMM_one_input/I_O_HMM_T';
%         p2=num2str(nT);
%         p3='_B';
%         p4=num2str(nB);
%         p5='_STD_PCAfirst.mat'; 
%         save([p1 p2 p3 p4 p5],'bnet2');
    end 
end 


MAE_test=sum(abs(Y_test(T+1:end)'-predicted_val_test_set))/length(Y_test(T+1:end))



%%%% plot %%%%
figure (3)
plot(1:n_sample_train-T,Y_train(T+1:end),'bo-');
hold on; 
plot(1:n_sample_train-T,predicted_val_train_set,'ro-'); 
title('Groud truth and predction (training data)'); 
xlabel('time sequence'); 
ylabel('value'); 
legend('Ground truth','Prediction ')


figure (4) 
plot(1:n_sample_test-T,Y_test(T+1:end),'bo-');
hold on; 
plot(1:n_sample_test-T,predicted_val_test_set,'ro-'); 
title('Groud truth and predction (test data)'); 
xlabel('time sequence'); 
ylabel('value'); 
legend('Ground truth','Prediction ')


figure (5)
plot(Y_train(T+1:end),predicted_val_train_set,'bo');

figure (6) 
plot(Y_test(T+1:end),predicted_val_test_set,'bo');
hold on;
plot(-1:6,-1:6,'r');

title('Groud truth v.s predction (test data)'); 
xlabel('prediction'); 
ylabel('ground'); 
legend('Ground truth v.s prediction','optimal prediction')

clc 
clear all 

%%% load data %%% 
File=load('data/train_data.mat'); 
train_data=File.train_data;

feature_mean=mean(train_data(:,1:end-1));
feature_std=std(train_data(:,1:end-1));
train_data(:,1:end-1)=(train_data(:,1:end-1)-feature_mean)./(feature_std);


%%%% find correlation between input features and %Silica 
covariance=cov(train_data);
[R,p]=corrcoef(train_data);
p_th=0.05; 
[sig_features_ind,~]=find(p(:,23)<=p_th); %%% find significantly correlated input features w.r.t output (23th coln)  
good_features=train_data(:,sig_features_ind); %%% Good features
Y=train_data(:,23); % Silica 

%%% PCA calcualtion to get rid of correlation btw inputs %%% 
feature_SIZE=size(good_features); 
data_number=feature_SIZE(1);
 
%%% when not std training set 
%feature_mean=mean(good_features);
%good_features_centered=good_features-feature_mean; 
%%% when std training set 
good_features_centered=good_features;

%covariance_matrix=good_features_centered'*good_features_centered/(data_numner-1); 
[U,S,V]=svd(good_features_centered);
features_PCed=good_features_centered*V; %%% n*Num_features

%%% not std 
%singular_th=0.1*10^4;  %%% set singular value threshold (PC component signidicance); 
%%% std 
singular_th=41;  %%% set singular value threshold (PC component signidicance); 
good_PCs_ind=find(max(S)>singular_th); 
features_goodPCed=features_PCed(:,good_PCs_ind); %%% n*Num_goodPCsFeatures  


%good_feature_num=sig_features_ind(good_PCs_ind); %%% final selected feature
selected_feature_mean=feature_mean(sig_features_ind); %%% mean of the seleced features (by correlation matrix)  
selected_feature_std=feature_std(sig_features_ind);
selected_train_data=features_goodPCed; %%% 


%%% save data 
train_processed=struct; 
train_processed.description="features filtered by correlation + PCA (substracted mean), Y unfiltered";
train_processed.feature_num=sig_features_ind; 
train_processed.feature_mean=selected_feature_mean;
train_processed.feature_std=selected_feature_std;
train_processed.feature=selected_train_data;
train_processed.Y=Y;
train_processed.PCs=V(:,good_PCs_ind); %%% selected basis 
%save('data/train_data_processed_std','train_processed'); 



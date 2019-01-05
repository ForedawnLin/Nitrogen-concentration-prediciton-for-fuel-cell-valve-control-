function [new_data,old_mean,old_std]=standlization(data,train_set,train_mean,train_std)
%%% data: n*m, m is the feature num, n is the data pts num 
%%% train_set: binary,training set if 1, test set if 0 
%%% train_mean: training mean,used for test set 
%%% train_std: training std, used for train set 

%%% output: 
%%% new_data: std:1 but original mean   
    if train_set==1
        old_mean=mean(data); 
        old_std=std(data); 
        new_data=(data-old_mean)./old_std+old_mean;
    else
        old_mean=[];
        old_std=[];
        new_data=(data-train_mean)./train_std+train_mean;
    end 
    
end 
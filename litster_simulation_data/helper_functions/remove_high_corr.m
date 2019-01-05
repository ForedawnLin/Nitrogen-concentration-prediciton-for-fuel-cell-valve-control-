function [new_data,index_left]=remove_high_corr(data,corr_th)
    R=corrcoef(data);
    SIZE=size(data);
    input_size=SIZE(2); 
    features=[1:input_size]; 
    index=[]; %%% index not to use (high,correlation)
    for i=1:input_size
        if ismember(i,index)==1
            continue 
        else 
            features(1)=[];
            index_2_remove=find(R(features,i)>corr_th);
            if length(index_2_remove)~=0
                features(index_2_remove)=[]; %%% 
                index_2_remove=index_2_remove+i; %%% +i, ude to R(features,i) is based on index of R(features,i)
                index=[index,index_2_remove];
            else 
                continue
            end 
        end 
    end 
    index_final=unique(index) ;
    new_data=zeros(SIZE(1),input_size-length(index_final)); 
    i=1; 
    index_left=[];
    for j=1:input_size 
        if ismember(j,index_final)
            continue 
        else 
           new_data(:,i)=data(:,j);
           i=i+1;
           index_left=[index_left j];
        end 
    new_data=new_data; 
    index_left=index_left;
end  
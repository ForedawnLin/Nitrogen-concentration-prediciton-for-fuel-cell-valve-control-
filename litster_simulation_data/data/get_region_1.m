function new_data_index=get_region_1(data)
%%% this script gets region 1 only 
%%% new_data_indexL a cell array, for each cell, it contains the index for
%%% each point in region 1 
k=1; 
index{k}={};

for i=1:length(data(:,1))-2
    if (data(i,11)-data(i+1,11))<0 
        index{k}=[index{k} i i+1];
        if (data(i+1,11)-data(i+2,11))>0
            k=k+1;
            index{k}={};
        end 
    end 
end 

for i=1:length(index)
    pts_2_get{i}=unique(cell2mat(index{i}));
end 
new_data_index=pts_2_get;

end 
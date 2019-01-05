function [new_data]=upSampling(data,sample_freq)
    %%%% sample_freq: sampling frequency in second
    %%%% M: original data 
    table_size=size(data);
    coln_num=table_size(2); %%% 
    %%%%%% sampling the frequency %%%%%%%% 
%     sample_freq=0.05;
    time_resample=0:sample_freq:data(end,1);
    time_resample=time_resample';
    data_num_us=length(time_resample); 
    data_us=zeros(data_num_us,coln_num); %%% init upsampling data 
    for i=1:coln_num
        if i==1
            data_us(:,i)=time_resample;
        else
            data_us(:,i)=interp1(data(:,1),data(:,i),0:sample_freq:data(end,1)); %%% upsampling feature
        end  
    end 
    new_data=data_us;
end 

classdef relation
   properties
      oo=0; 
   end
   methods
       
      function result = diffusivity(relation,H2O,T)
         a=0.37;
         b=9.871;
         result = a*exp(1)^H2O+1/T*b;  
      end
      
      function result = C_diff(relation,CN2C,CN2A)
         result = CN2A-CN2C;
      end
            
      function result = H2_flowRate(relation,pressureC,T)
         rio=0.102; 
         result = rio*pressureC/sqrt(2000)*T;  
      end
 
      function result = air_flowRate(relation,pressureA,T)
         rio=1.02; 
         result = rio*pressureA^2/sqrt(2500)*T;  
      end
      
      
      function N2=data_generation_static(relation,H2O,T,C_N2C,C_N2A,P_H2,P_air)
          Di=relation.diffusivity(H2O,T); 
          c_diff=relation.C_diff(C_N2C,C_N2A); 
          H2_dot=relation.H2_flowRate(P_H2,T); 
          air_dot=relation.air_flowRate(P_air,T); 
          
          a=0.81
          b=3.92
          
          N2=sqrt(a*c_diff/Di*(sqrt(H2_dot)+air_dot^(1/3))+b/Di^2*c_diff^(4/3))
      end 
      
   end
end
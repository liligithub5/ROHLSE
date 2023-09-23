function [Bi_Ir,Bt_Tr,Bi_Ie,Bt_Te] = main_ROHLSE(streamdata, I_te, T_te, bits)
%% --------mirflickr -------- %%
param.beta  = 1e-1;
param.alpha  = 1;
param.mu = 1e-1;
param.gamma = 1;   %%增加标签关系
param.phi  = 1e-3;
param.lambda = 1e-2;
param.iter = 10;

%% --------IJiapr-tc-------- %%
% param.beta  = 1e2;
% param.alpha  = 1e-2;
% param.mu = 1e-1;
% param.gamma = 1e-3;   %%增加标签关系
% param.phi  = 1e-3;
% param.lambda = 1e-1;
% param.iter = 10;

%% --------nus-wide -------- %%
% param.beta  = 1e2;
% param.alpha  = 1e-2;
% param.mu = 1e-1;
% param.gamma = 1;   %%增加标签关系
% param.phi  = 1e-3;
% param.lambda = 1e-3;
% param.iter = 15;

nstream = size(streamdata,2);
for chunki = 1:nstream 
    Itrain = streamdata{1,chunki}';  Ttrain = streamdata{2,chunki}';LTrain = streamdata{3,chunki}';
    if chunki == 1
      [B, V, M, Yn,Z] = train_ROHLSE0(Itrain, Ttrain, LTrain, param, bits); 
       ZM = Z*M; 
       C1 = ZM*Itrain';
       D1 = Itrain*Itrain';
       C2 = ZM*Ttrain';
       D2 = Ttrain*Ttrain';
       C3 = V*Yn';  
       F1 = B*Itrain';
       F2 = B*Ttrain';
    else
       D1 = D1 + Itrain*Itrain';
       D2 = D2 + Ttrain*Ttrain';
       [B, Bt, C1, C2, C3] = train_ROHLSE(Itrain, Ttrain, LTrain, param, bits, B, C1, C2, C3, D1, D2);
       F1 = F1 + Bt*Itrain';
       F2 = F2 + Bt*Ttrain';
    end
end
    PI = F1*pinv(D1+param.lambda*eye(size(D1,1)));
    PT = F2*pinv(D2+param.lambda*eye(size(D2,1)));
    Bt_Tr = compactbit(B' >= 0);
    Bi_Ir = Bt_Tr;
    Yi_te = sign((bsxfun(@minus,I_te*PI' , mean(I_te*PI',1))));
    Yt_te = sign((bsxfun(@minus,T_te*PT' , mean(T_te*PT',1))));
    Bt_Te = compactbit(Yt_te >= 0);
    Bi_Ie = compactbit(Yi_te >= 0);
end

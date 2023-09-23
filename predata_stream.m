function [streamdata,streamdata_non,nstream,L_tr,I_tr,T_tr,I_tr_non,T_tr_non,I_te_non,T_te_non] = predata_stream(I_tr,T_tr,L_tr,I_te,T_te,numbatch)

rand('seed',1);
anchors = 1000;
anchor_idx = randsample(size(I_tr,1), anchors);
XAnchors = I_tr(anchor_idx,:);
anchor_idx = randsample(size(T_tr,1), anchors);
YAnchors = T_tr(anchor_idx,:);
[I_tr_non,I_te_non]=Kernel_Feature(I_tr,I_te,XAnchors);
[T_tr_non,T_te_non]=Kernel_Feature(T_tr,T_te,YAnchors);
row = size(L_tr,1);
LTrain = L_tr;
index0 = find(LTrain==0);
index1 = find(LTrain==1);
num1 = length(index1);
ratio = 0.2;
Rnum = round(ratio * num1);   
Rdata = index1(randperm(length(index1),Rnum)); %随机选择ratio*n(n为标签矩阵中1的个数，ratio为选择比率)个数作为缺失标签
Rdata = sort(Rdata);
Mdata = index0(randperm(length(index0),Rnum)); %随机选择ratio*n(n为标签矩阵中1的个数，ratio为选择比率)个数作为错误标签
Mdata = sort(Mdata); 
for i = 1:Rnum    
    if(mod(Rdata(i),row)==0)
        n1 = floor(Rdata(i) / row);
        m1 = row;
    else
        n1 = floor(Rdata(i) / row) + 1;
        m1 = mod(Rdata(i),row);
    end    
    if(mod(Mdata(i),row)==0)
        n0 = floor(Mdata(i) / row);
        m0 = row;
    else
       n0 = floor(Mdata(i) / row) + 1; 
       m0 = mod(Mdata(i),row);
    end
    LTrain(m1,n1) = 0;       %缺失标签处理
    LTrain(m0,n0) = 1;       %错误标签处理 
end
data = randperm(row);
I_tr = I_tr(data,:);
T_tr = T_tr(data,:);
LTrain = LTrain(data,:);
L_tr = L_tr(data,:);

I_tr_non = I_tr_non(data,:);
T_tr_non = T_tr_non(data,:);

nstream = ceil(row/numbatch);
streamdata = cell(3,nstream);
streamdata_non = cell(3,nstream);
for i = 1:nstream-1
    start = (i-1)*numbatch+1;
    endl = i*numbatch;
    streamdata{1,i} = I_tr(start:endl,:);
    streamdata{2,i} = T_tr(start:endl,:);
    streamdata{3,i} = LTrain(start:endl,:);
    
    streamdata_non{1,i} = I_tr_non(start:endl,:);
    streamdata_non{2,i} = T_tr_non(start:endl,:);
    streamdata_non{3,i} = LTrain(start:endl,:);
end
start = (nstream-1)*numbatch+1;
streamdata{1,nstream} = I_tr(start:end,:);
streamdata{2,nstream} = T_tr(start:end,:);
streamdata{3,nstream} = LTrain(start:end,:);

streamdata_non{1,nstream} = I_tr_non(start:end,:);
streamdata_non{2,nstream} = T_tr_non(start:end,:);
streamdata_non{3,nstream} = LTrain(start:end,:);



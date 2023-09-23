function [B, Bt, C1, C2, C3, V, Yn] = train_ROHLSE(X1, X2, L, param, bits, B1, C1, C2, C3, D1, D2)
[dx1, n ] = size(X1); [dx2, ~ ] = size(X2); [c, ~] = size(L);
beta = param.beta;
mu = param.mu;
gamma = param.gamma;
lambda = param.lambda;
phi = param.phi;
alpha = param.alpha;

[pcaW, ~] = eigs(cov(X1'), bits);
H1_c = pcaW'*X1;
[pcaW, ~] = eigs(cov(X2'), bits); 
H2_c = pcaW'*X2;
V =  (H1_c+H2_c)/2  ;
[pcaW, ~] = eigs(cov(L'), c);
M = pcaW'*L;
Z = eye(c);
N = Z*M; 

E = zeros(size(L));
Bt = sign(V); Bt(Bt==0)=-1;
X1in = pinv(mu*D1 + lambda*eye(dx1));
X2in = pinv(mu*D2 + lambda*eye(dx2));
W1 = (mu*(C1+(L-E)*X1'))*X1in;
W2 = (mu*(C2+(L-E)*X2'))*X2in;
F1 = zeros(size(L));
F2 = zeros(size(N));
Yn = L./repmat(sqrt(sum(L.*L))+1e-8,[size(L, 1),1]);
rho = 1;
theta = 1;

for iter = 1:param.iter  

     %% Bt
     C3 = C3 + V*Yn';
     Bt = sign(phi*V + bits*theta*C3*Yn);
     Bt(Bt==0)=-1;
     %% Z
    Z = ((mu*(W1*X1+W2*X2)+gamma*M+rho/2*(L-E+1/rho*F1)+rho/2*(N-1/rho*F2))*M')*pinv((rho+2*mu+gamma)*M*M');
    %% M
    M = pinv((rho+2*mu+gamma)*Z'*Z+gamma*(eye(c)-Z))*(Z'*(rho/2*(L-E+1/rho*F1) + rho/2*(N-1/rho*F2) +mu*(W1*X1+W2*X2)));
    ZM = Z*M;
    % N
    [U1,S1,V1] = svd(ZM+F2/rho,'econ');
    a = diag(S1)-alpha/rho;
    a(a<0)=0; 
    T = diag(a);
    N = U1*T*V1';  clear U1 S1 V1 T;
    
     %% W
     C1 = C1 + ZM*X1';
     C2 = C2 + ZM*X2';
     W1 = mu*C1*X1in;
     W2 = mu*C2*X2in;
    
     % E
     Etp = L - ZM + 1/rho*F1;
     E = sign(Etp).*max(abs(Etp)- alpha*beta/rho,0); 
     Yn = L -E;
     Yn = Yn./repmat(sqrt(sum(Yn.*Yn))+1e-8,[size(Yn, 1),1]);

     % V       
     D = bits*theta*Bt*Yn'*Yn + phi*Bt;
     D = D' ;
     Temp = D'*D-1/n*(D'*ones(n,1)*(ones(1,n)*D));
     [~,Lmd,QQ] = svd(Temp); clear Temp
     idx = (diag(Lmd)>1e-6);
     Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
     Pt = (D-1/n*ones(n,1)*(ones(1,n)*D)) *  (Q / (sqrt(Lmd(idx,idx))));
     P_ = orth(randn(n,bits-length(find(idx==1))));
     V = sqrt(n)*[Pt P_]*[Q Q_]';
     V = V';       
     
     F1 = F1 + rho*(L - ZM -E);
     F2 = F2 + rho*(ZM - N);    
     rho = min(1e6, 1.21*rho);
end
    
    B = [B1,Bt];
end


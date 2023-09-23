function [B, V, M, Yn,Z] = train_ROHLSE0(X1, X2, L, param, bits)

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
B = sign(V); B(B==0)=-1;

X1X1 = X1*X1';
X2X2 = X2*X2'; 
W1 = (mu*(L-E)*X1')*pinv(mu*X1X1 + lambda*eye(dx1));
W2 = (mu*(L-E)*X2')*pinv(mu*X2X2 + lambda*eye(dx2));
F1 = zeros(size(L));
F2 = zeros(size(N));

Yn = L./repmat(sqrt(sum(L.*L))+1e-8,[size(L, 1),1]);                                     
rho = 1;
X1in = pinv(mu*X1X1 + lambda*eye(dx1));
X2in = pinv(mu*X2X2 + lambda*eye(dx2));
theta = 1;

for iter = 1:param.iter  

    % B
     B = sign(phi*V + bits*theta*V*Yn'*Yn);
     B(B==0)=-1;
    
    Z = ((mu*(W1*X1+W2*X2)+gamma*M+rho/2*(L-E+1/rho*F1)+rho/2*(N-1/rho*F2))*M')*pinv((rho+2*mu+gamma)*M*M');
    % M
    M = pinv((rho+2*mu+gamma)*Z'*Z+gamma*(eye(c)-Z))*(Z'*(rho/2*(L-E+1/rho*F1) + rho/2*(N-1/rho*F2) +mu*(W1*X1+W2*X2)));
    
    ZM = Z*M;
    % N
    [U1,S1,V1] = svd(ZM+F2/rho,'econ');
    a = diag(S1)-alpha/rho;
    a(a<0)=0; 
    T = diag(a);
    N = U1*T*V1';  clear U1 S1 V1 T;

     % W
     W1 = (mu*ZM*X1')*X1in ;
     W2 = (mu*ZM*X2')*X2in ;
    
     % E
     Etp = L - ZM + 1/rho*F1;
     E = sign(Etp).*max(abs(Etp)- alpha*beta/rho,0); 
     Yn = L -E;
     Yn = Yn./repmat(sqrt(sum(Yn.*Yn))+1e-8,[size(Yn, 1),1]);

     % V       
     D = bits*theta*B*Yn'*Yn + phi*B;
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
end


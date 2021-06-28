function [IW,B,LW,AF,TYPE] = elmtrain(P,T,N,AF,TYPE)
% ELMTRAIN Create and Train a Extreme Learning Machine
% Syntax
% [IW,B,LW,AF,TYPE] = elmtrain(P,T,N,AF,TYPE)
% Description
% Input
% P   - Input Matrix of Training Set  (R*Q)
% T   - Output Matrix of Training Set (S*Q)
% N   - Number of Hidden Neurons (default = Q)
% AF  - Active Function:  激活函数
%       'sig' for Sigmoidal function (default)
%       'sin' for Sine function
%       'hardlim' for Hardlim function
% TYPE - Regression (0,default) or Classification (1)
% Output
% IW  - Input Weight Matrix (N*R)  输入权重
% B   - Bias Matrix (N*1)  隐层偏差
% LW  - Layer Weight Matrix (N*S)  输出权重
% Example
% Regression:
% [IW,B,LW,AF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,AF,TYPE)
% Classification
% [IW,B,LW,AF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmtrain(P,IW,B,LW,AF,TYPE)
% See also ELMPREDICT

% P = DataTrn';
% T = LabTrn';
% N = 1000;
% AF = 'sig';
% TYPE = 1;
if nargin < 2
    error('ELM:Arguments','Not enough input arguments.');
end
if nargin < 3
    N = size(P,2);
end
if nargin < 4
    AF = 'sig';
end
if nargin < 5
    TYPE = 0;
end
if size(P,2) ~= size(T,2)
    error('ELM:Arguments','The columns of P and T must be same.');
end
[R,Q] = size(P);
if TYPE  == 1
    T  = ind2vec(T);
end
[S,Q] = size(T);
% Randomly Generate the Input Weight Matrix
IW = rand(N,R) * 2 - 1;
% Randomly Generate the Bias Matrix
% B = rand(N,1);
B = rand(N,1)*2-1;
BiasMatrix = repmat(B,1,Q);
% Calculate the Layer Output Matrix H
tempH = IW * P + BiasMatrix;
switch AF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH)); %隐藏层输出
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
    case 'tanh'
        H = 2./(1+exp(-2*tempH))-1;
    case 'ReLU'
        H = max(0,tempH);
    case 'tribas'
        H = tribas(tempH);         % Triangular basis 函数
    case 'radbas'
        H = radbas(tempH);         % Radial basis 函数
end
% Calculate the Output Weight Matrix
LW = pinv(H') * T'; %pinv：广义逆矩阵

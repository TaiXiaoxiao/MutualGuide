function[Y0,Y] = elmpredict(P,IW,B,LW,AF,TYPE)
% ELMPREDICT Simulate a Extreme Learning Machine
% Syntax
% Y = elmtrain(P,IW,B,LW,AF,TYPE)
% Description
% Input
% P   - Input Matrix of Training Set  (R*Q)
% IW  - Input Weight Matrix (N*R)
% B   - Bias Matrix  (N*1)
% LW  - Layer Weight Matrix (N*S)
% AF  - Transfer Function:
%       'sig' for Sigmoidal function (default)
%       'sin' for Sine function
%       'hardlim' for Hardlim function
% TYPE - Regression (0,default) or Classification (1)
% Output
% Y   - Simulate Output Matrix (S*Q)
% Example
% Regression:
% [IW,B,LW,AF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,AF,TYPE)
% Classification
% [IW,B,LW,AF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmtrain(P,IW,B,LW,AF,TYPE)
% See also ELMTRAIN
% P = DataTrn';

if nargin < 6
    error('ELM:Arguments','Not enough input arguments.');
end
% Calculate the Layer Output Matrix H

Q = size(P,2); %P的列数
BiasMatrix = repmat(B,1,Q);
tempH = IW * P + BiasMatrix;
switch AF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
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
% Calculate the Simulate Output
Y0 = (H' * LW)'; %分类结果
if TYPE == 1
    temp_Y = zeros(size(Y0));
    for i = 1:size(Y0,2)
        [~,index] = max(Y0(:,i));
        temp_Y(index,i) = 1;
    end
    Y = vec2ind(temp_Y); %向量变索引
end
       

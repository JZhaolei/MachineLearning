function [IW,B,LW,TF,TYPE] = elmtrain(P,T,N,TF,TYPE)
% ELMTRAIN Create and Train a Extreme Learning Machine
% Syntax
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,N,TF,TYPE)
% Description
% Input
% P   - Input Matrix of Training Set  (R*Q)R为特征数，Q为样本个数
% T   - Output Matrix of Training Set (S*Q)S为训练集输出的特征数，Q为输出的样本个数
% N   - Number of Hidden Neurons (default = Q) 隐含层神经元数，默认为样本个数
% TF  - Transfer Function:
%       'sig' for Sigmoidal function (default)
%       'sin' for Sine function
%       'hardlim' for Hardlim function
% TYPE - Regression (0,default) or Classification (1)
% Output
% IW  - Input Weight Matrix (N*R) 随机产生输入层与隐含层权值
% B   - Bias Matrix  (N*1) 隐含层神经元阈值
% LW  - Layer Weight Matrix (N*S) 隐含层与输出层权值
% Example
% Regression:
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Classification
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% See also ELMPREDICT
% Yu Lei,11-7-2010
% Copyright www.matlabsky.com
% $Revision:1.0 $
%%其实以下这一块都是设置默认值，技巧，设置默认值时可以用传递参数个数来限制
if nargin < 2
    error('ELM:Arguments','Not enough input arguments.');
end
if nargin < 3
    N = size(P,2);
end
if nargin < 4
    TF = 'sig';
end
if nargin < 5
    TYPE = 0;
end
if size(P,2) ~= size(T,2)
    error('ELM:Arguments','The columns of P and T must be same.');
end
[R,Q] = size(P);
if TYPE  == 1
    T  = ind2vec(T);%%把输出的特征数向量化，便于取出输出特征的个数，此处即为3类特征，S=3
end
[S,Q] = size(T);
% Randomly Generate the Input Weight Matrix
IW = rand(N,R) * 2 - 1;
% Randomly Generate the Bias Matrix
B = rand(N,1);
BiasMatrix = repmat(B,1,Q);%把B重构成原来行的1倍，原来列的Q倍的新矩阵
% Calculate the Layer Output Matrix H 隐含层的输入
tempH = IW * P + BiasMatrix;
switch TF %%隐含层输出
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end
% Calculate the Output Weight Matrix
LW = pinv(H') * T';%%隐含层与输出层权值计算

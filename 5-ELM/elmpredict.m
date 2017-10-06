function Y = elmpredict(P,IW,B,LW,TF,TYPE)
% ELMPREDICT Simulate a Extreme Learning Machine
% Syntax
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Description
% Input
% P   - Input Matrix of Training Set  (R*Q)
% IW  - Input Weight Matrix (N*R)
% B   - Bias Matrix  (N*1)
% LW  - Layer Weight Matrix (N*S)
% TF  - Transfer Function:
%       'sig' for Sigmoidal function (default)
%       'sin' for Sine function
%       'hardlim' for Hardlim function
% TYPE - Regression (0,default) or Classification (1)
% Output
% Y   - Simulate Output Matrix (S*Q)
% Example
% Regression:
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Classification
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% See also ELMTRAIN
% Yu Lei,11-7-2010
% Copyright www.matlabsky.com
% $Revision:1.0 $
if nargin < 6
    error('ELM:Arguments','Not enough input arguments.');
end
% Calculate the Layer Output Matrix H
Q = size(P,2);
BiasMatrix = repmat(B,1,Q);
tempH = IW * P + BiasMatrix;
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end
% Calculate the Simulate Output
Y = (H' * LW)';%%回归的输出
if TYPE == 1%%分类的输出
    temp_Y = zeros(size(Y));
    for i = 1:size(Y,2)
        [~,index] = max(Y(:,i));%%取每个特征的最大值,即所有列的最大值，找到第i列最大值的位置
        temp_Y(index,i) = 1;%%将此最大值赋1
    end
    Y = vec2ind(temp_Y); %仅有0和1的为向量，行数少的为索引，由向量到索引时，向量中每一列中的所有元素有且只能有一个为1，即该列最大值所在的位置，向量的列数即为索引矩阵的列数
    %ind2vec()函数表示从索引到向量，索引矩阵中有多少列，得到的索引表就有多少列，索引矩阵中的最大值即是索引表的行数。
end
       
    

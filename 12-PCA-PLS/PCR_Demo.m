%% I. 清空环境变量
clear all
clc

%% II. 导入数据
load spectra;

%% III. 随机划分训练集与测试集
temp = randperm(size(NIR, 1));
% temp = 1:60;
%%
% 1. 训练集——50个样本
P_train = NIR(temp(1:50),:);
T_train = octane(temp(1:50),:);
%%
% 2. 测试集——10个样本
P_test = NIR(temp(51:end),:);
T_test = octane(temp(51:end),:);

%% IV. 主成分分析
%%
% 1. 主成分贡献率分析
[PCALoadings,PCAScores,PCAVar] = princomp(NIR);%PCAVar为主成分的特征值，PCALoading原变量映射到新空间后主成分的系数；PCAScores代表原样本映射到新空间对应的系数
% [COEFF, SCORE, LATENT] = PRINCOMP(X) 
% LATENT协方差矩阵的特征值。
% SCORE是对主分的打分，也就是说原X矩阵在主成分空间的表示。
% COEFF是X矩阵所对应的协方差阵V的所有特征向量组成的矩阵，即变换矩阵或称投影矩阵。
% 用你的原矩阵x*coeff(:,1:n)才是你要的的新数据，其中的n是你想降到多少维。
figure
percent_explained = 100 * PCAVar / sum(PCAVar);%主成分的贡献率
pareto(percent_explained)%Pareto图又称排列图是一种按事件发生的频率排序而成，显示由于各种原因引起的缺陷数量或不一致的排列顺序，是找出影响项目产品或服务质量的主要因素的方法。
xlabel('主成分')
ylabel('贡献率(%)')
title('主成分贡献率')

%%
% 2. 第一主成分vs.第二主成分（判断样本划分是否标准，若测试集的点都在训练集内包含，则划分较好）
[PCALoadings,PCAScores,PCAVar] = princomp(P_train);
figure
plot(PCAScores(:,1),PCAScores(:,2),'r+')
hold on
[PCALoadings_test,PCAScores_test,PCAVar_test] = princomp(P_test);
plot(PCAScores_test(:,1),PCAScores_test(:,2),'o')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
legend('Training Set','Testing Set','location','best')

%% V. 主成分回归模型
%%
% 1. 创建模型
k = 4;
betaPCR = regress(T_train-mean(T_train),PCAScores(:,1:k));%T为Y，PCAScores为X，求出的是回归系数
betaPCR = PCALoadings(:,1:k) * betaPCR;%后两部是逆向操作，为了预测做准备，以下四句为固定调用模式
betaPCR = [mean(T_train)-mean(P_train) * betaPCR;betaPCR];
%%
% 2. 预测拟合
N = size(P_test,1);
T_sim = [ones(N,1) P_test] * betaPCR;

%% VI. 结果分析与绘图
%%
% 1. 相对误差error
error = abs(T_sim - T_test) ./ T_test;
%%
% 2. 决定系数R^2
R2 = (N * sum(T_sim .* T_test) - sum(T_sim) * sum(T_test))^2 / ((N * sum((T_sim).^2) - (sum(T_sim))^2) * (N * sum((T_test).^2) - (sum(T_test))^2)); 
%%
% 3. 结果对比
result = [T_test T_sim error]

%% 
% 4. 绘图
figure
plot(1:N,T_test,'b:*',1:N,T_sim,'r-o')
legend('真实值','预测值','location','best')
xlabel('预测样本')
ylabel('辛烷值')
string = {'测试集辛烷值含量预测结果对比';['R^2=' num2str(R2)]};
title(string)



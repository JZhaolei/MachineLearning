%% I. 清空环境变量
clear all
clc
warning off

%% II. 导入数据
load data.mat

%%
% 1. 随机产生训练集/测试集
a = randperm(569);
Train = data(a(1:500),:);
Test = data(a(501:end),:);

%%
% 2. 训练数据
P_train = Train(:,3:end);%数据共有569行，32列，即569个样本，30个特征，其中列的前两行为样本编号及类别
T_train = Train(:,2);

%%
% 3. 测试数据
P_test = Test(:,3:end);
T_test = Test(:,2);

%% III. 创建决策树分类器
ctree = ClassificationTree.fit(P_train,T_train);
%用法tree = ClassificationTree.fit(X,Y,Name,Value)

%%
% 1. 查看决策树视图
view(ctree);
view(ctree,'mode','graph');

%% IV. 仿真测试
T_sim = predict(ctree,P_test);

%% V. 结果分析
count_B = length(find(T_train == 1));%训练集中良性与恶性的数目（实际统计）
count_M = length(find(T_train == 2));
rate_B = count_B / 500;
rate_M = count_M / 500;
total_B = length(find(data(:,2) == 1));%实际病例中良性与恶性的数目（实际）
total_M = length(find(data(:,2) == 2));
number_B = length(find(T_test == 1));%测试集中，良性与恶性的数目（实际）
number_M = length(find(T_test == 2));
number_B_sim = length(find(T_sim == 1 & T_test == 1));%测试集中，实际为良性并且预测为良性的数目（预测结果）
number_M_sim = length(find(T_sim == 2 & T_test == 2));
disp(['病例总数：' num2str(569)...
      '  良性：' num2str(total_B)...
      '  恶性：' num2str(total_M)]);
disp(['训练集病例总数：' num2str(500)...
      '  良性：' num2str(count_B)...
      '  恶性：' num2str(count_M)]);
disp(['测试集病例总数：' num2str(69)...
      '  良性：' num2str(number_B)...
      '  恶性：' num2str(number_M)]);
disp(['良性乳腺肿瘤确诊：' num2str(number_B_sim)...
      '  误诊：' num2str(number_B - number_B_sim)...
      '  确诊率p1=' num2str(number_B_sim/number_B*100) '%']);
disp(['恶性乳腺肿瘤确诊：' num2str(number_M_sim)...
      '  误诊：' num2str(number_M - number_M_sim)...
      '  确诊率p2=' num2str(number_M_sim/number_M*100) '%']);
  
%% VI. 叶子节点含有的最小样本数对决策树性能的影响
leafs = logspace(1,2,10);%logspace(a,b,n)，创建行向量，第一个是10^a，最后一个10^b，形成总数为n个元素的等比数列

N = numel(leafs);

err = zeros(N,1);
for n = 1:N
    t = ClassificationTree.fit(P_train,T_train,'crossval','on','minleaf',leafs(n));
    %crossval
    %标志形成交叉验证决策树，可取值为on或者off，on生成10折交叉验证，可以通过'kfold','holdout'等修改折数，默认状态为off
    err(n) = kfoldLoss(t);%交叉验证误差——此处原理有待明确
end
plot(leafs,err);
xlabel('叶子节点含有的最小样本数');
ylabel('交叉验证误差');
title('叶子节点含有的最小样本数对决策树性能的影响')

%% VII. 设置minleaf为13，产生优化决策树,此处需要根据含有最小样本数的样子结点的位置来调整，由于结点含有样本数多时，交叉验证误差会变大，因此其最小时性能最好
OptimalTree = ClassificationTree.fit(P_train,T_train,'minleaf',13);
view(OptimalTree,'mode','graph')

%%
% 1. 计算优化后决策树的重采样误差和交叉验证误差
resubOpt = resubLoss(OptimalTree)
lossOpt = kfoldLoss(crossval(OptimalTree))

%%
% 2. 计算优化前决策树的重采样误差和交叉验证误差
resubDefault = resubLoss(ctree)
lossDefault = kfoldLoss(crossval(ctree))

%% VIII. 剪枝
[~,~,~,bestlevel] = cvLoss(ctree,'subtrees','all','treesize','min')
%用法[E,SE,Nleaf,BestLevel]=cvLoss(tree,Name,Value)
%subtrees剪枝水平矩阵，值为0代表完全不剪枝；all代表全剪枝，默认为0
%treesize为‘se’时代表一定水平的最小代价情况下的最小树，为默认值，而'min'代表最小代价树
cptree = prune(ctree,'Level',bestlevel);
view(cptree,'mode','graph')

%%
% 1. 计算剪枝后决策树的重采样误差和交叉验证误差
resubPrune = resubLoss(cptree)
lossPrune = kfoldLoss(crossval(cptree))


%%一、清空环境变量
clear all;
clc;

%%二、导入数据
load('concrete_data.mat');

%随机产生训练集和测试集
temp = randperm(size(attributes,2));
% 训练集——80个样本
P_train = attributes(:,temp(1:80));%列为样本个数；保证列相等
T_train = strength(:,temp(1:80));
% 测试集——23个样本
P_test = attributes(:,temp(81:end));
T_test = strength(:,temp(81:end));
N = size(P_test,2);

%% 三、数据归一化
[p_train, ps_input] = mapminmax(P_train,0,1);%ps_input是一种对应关系，里面包括一些相应的特征值。
p_test = mapminmax('apply',P_test,ps_input);%利用这个对应关系ps_input对其他数值进行归一化，但是有个前提，这个数必须要在p_trian的min 和max之间，不然归一化的结果，与整体进行归一化，结果会不一样

[t_train, ps_output] = mapminmax(T_train,0,1);

%% 四. BP神经网络创建、训练及仿真测试
%%
% 1. 创建网络
net = newff(p_train,t_train,[20,30]);%隐含神经元的个数，一个数字可以用于单隐含层，矩阵可以做成多隐含层，随意设置
%查看用view(net)
% 2. 设置训练参数
net.trainParam.epochs = 1000;%迭代次数
net.trainParam.goal = 1e-3;%目标：均方根误差小于1e-3；
net.trainParam.lr = 0.01;%学习率
% 3. 训练网络
net = train(net,p_train,t_train);
% 4. 仿真测试
t_sim = sim(net,p_test);%用训练模型仿真测试集
% 5. 数据反归一化
T_sim = mapminmax('reverse',t_sim,ps_output);%进行反归一化,讲归一化的数据反归一化再得到原来的数据:

%% 五. 性能评价
%%
% 1. 相对误差error
error = abs(T_sim - T_test)./T_test;

%%
% 2. 决定系数R^2，接近1性能越好
R2 = (N * sum(T_sim .* T_test) - sum(T_sim) * sum(T_test))^2 / ((N * sum((T_sim).^2) - (sum(T_sim))^2) * (N * sum((T_test).^2) - (sum(T_test))^2)); 

%%
% 3. 结果对比
result = [T_test' T_sim' error']

%% VI. 绘图
figure
plot(1:N,T_test,'b:*',1:N,T_sim,'r-o')
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测值')
%string = {'测试集辛烷值含量预测结果对比';['R^2=' num2str(R2)]};
%title(string)



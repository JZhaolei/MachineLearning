%% I. 清空环境变量
clear all
clc

%% II. 训练集/测试集产生
%%
% 1. 导入数据
load BreastTissue_data.mat

%%
% 2. 数据归一化
matrix = mapminmax(matrix);
% 3 数据增加，使各类样本数目保持一致
%首先对matrix矩阵各类样本进行增加，使之都达到最大值22，随机复制某几行的数据，增补到原数据末尾
n1 = randperm(21);%第一类原有21个样本，位于1-21行
breast1=matrix((1:21),:);%提取第一类样本的特征矩阵
breast1=[breast1;breast1(n1(1),:)];%完成数据增加，获得新样本

n2 = randperm(15);%第二类原有15个样本，位于22-36行
breast2=matrix((22:36),:);
breast2=[breast2;breast2(n2(1:7),:)];

n3 = randperm(18);%第三类原有18个样本，位于37-54行
breast3=matrix((37:54),:);
breast3=[breast3;breast3(n3(1:4),:)];

n4 = randperm(16);%第四类原有16个样本，位于55-70行
breast4=matrix((55:70),:);
breast4=[breast4;breast4(n4(1:6),:)];

n5 = randperm(14);%第五类原有14个样本，位于71-84行
breast5=matrix((71:84),:);
breast5=[breast5;breast5(n5(1:8),:)];

breast6=matrix((85:106),:);%第五类原有22个样本，位于85-106行

%构建新的样本数据集
newbreast=[breast1;breast2;breast3;breast4;breast5;breast6];

%构建新的标签集
newlabel=zeros(size(newbreast,1),1);
for a = 1:6
    newlabel((a-1)*22+1:a*22)= a;
end

% 4 随机产生训练集和测试集
P_train = [];
T_train = [];
P_test = [];
T_test = [];
for i = 1:6
    temp_input = newbreast((i-1)*22+1:i*22,:);%%优化，breast1-6也做成循环，就不用再提取一遍每个样本了，直接用breast（i）操作即可
    temp_output = newlabel((i-1)*22+1:i*22,:);%%优化时要注意breast(i)本身不是一维的，它也是个矩阵，有行有列，注意下标的歧义问题
    n = randperm(22);
    % 训练集——18个样本
    P_train = [P_train temp_input(n(1:18),:)'];%%X与Y都转置，还是列对应每个样本，行对应特征
    T_train = [T_train temp_output(n(1:18),:)'];
    % 测试集——4个样本
    P_test = [P_test temp_input(n(19:22),:)'];
    T_test = [T_test temp_output(n(19:22),:)'];
end
N=size(T_test,2);

%% III. 竞争神经网络创建、训练及仿真测试
%%
% 1. 创建网络
net = newc(minmax(P_train),6,0.01,0.01);

%%
% 2. 设置训练参数
net.trainParam.epochs = 500;

%%
% 3. 训练网络
net = train(net,P_train);

%%
% 4. 仿真测试

% 训练集
t_sim_compet_1 = sim(net,P_train);
T_sim_compet_1 = vec2ind(t_sim_compet_1);
% 测试集
t_sim_compet_2 = sim(net,P_test);
T_sim_compet_2 = vec2ind(t_sim_compet_2);

%% IV. SOFM神经网络创建、训练及仿真测试
%%
% 1. 创建网络
net = newsom(P_train,[6 6]);

%%
% 2. 设置训练参数
net.trainParam.epochs = 200;

%%
% 3. 训练网络
net = train(net,P_train);

%%
% 4. 仿真测试

% 训练集
t_sim_sofm_1 = sim(net,P_train);
T_sim_sofm_1 = vec2ind(t_sim_sofm_1);
% 测试集
t_sim_sofm_2 = sim(net,P_test);
T_sim_sofm_2 = vec2ind(t_sim_sofm_2);

%% V. 结果对比
%%
% 1. 竞争神经网络
result_compet_1 = [T_train' T_sim_compet_1']
result_compet_2 = [T_test' T_sim_compet_2']

%%
% 2. SOFM神经网络
result_sofm_1 = [T_train' T_sim_sofm_1']
result_sofm_2 = [T_test' T_sim_sofm_2']


%figure
%plot(1:N,T_test,'b:*',1:N,T_sim_compet_2,'r-o')
%legend('真实值','预测值')
%xlabel('预测样本')
%ylabel('预测值')

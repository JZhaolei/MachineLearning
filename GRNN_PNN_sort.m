%% I. 清空环境变量
clear all
clc

%% II. 训练集/测试集产生
%%
% 1. 导入数据
load BreastTissue_data.mat

%%
% 2 数据增加，使各类样本数目保持一致
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

% 3 随机产生训练集和测试集
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

%% III. 模型建立 
result_grnn = [];
result_pnn = [];
time_grnn = [];
time_pnn = [];
%%for i = 1:9
    %%for j = i:9
      %  p_train = P_train(i:j,:);%%把所有特征单独，两两，三三，等全部排列组合，建立不同的模型，对比那些特征分类效果好
       % p_test = P_test(i:j,:);
       %% 
        p_train = P_train;
        p_test =P_test;
        % 1. GRNN创建及仿真测试
        t = cputime;
        % 创建网络
        net_grnn = newgrnn(p_train,T_train);
        % 仿真测试
        t_sim_grnn = sim(net_grnn,p_test);
        T_sim_grnn = round(t_sim_grnn);
        t = cputime - t;
        time_grnn = [time_grnn t];
        result_grnn = [result_grnn T_sim_grnn'];
       %%
        % 2. PNN创建及仿真测试
        t = cputime;
        Tc_train = ind2vec(T_train);
        % 创建网络
        net_pnn = newpnn(p_train,Tc_train);
        % 仿真测试
        Tc_test = ind2vec(T_test);
        t_sim_pnn = sim(net_pnn,p_test);
        T_sim_pnn = vec2ind(t_sim_pnn);
        t = cputime - t;%%差值计算程序运行时间
        time_pnn = [time_pnn t];
        result_pnn = [result_pnn T_sim_pnn'];
    %%end
%%end

%% IV. 性能评价
%%
% 1. 正确率accuracy
accuracy_grnn = [];
accuracy_pnn = [];
time = [];
%%for i = 1:511  %9个特征的各种排列组合
    %accuracy_1 = length(find(result_grnn(:,i) == T_test'))/length(T_test);
    %accuracy_2 = length(find(result_pnn(:,i) == T_test'))/length(T_test);
    accuracy_1 = length(find(result_grnn == T_test'))/length(T_test);
    accuracy_2 = length(find(result_pnn == T_test'))/length(T_test);
    accuracy_grnn = [accuracy_grnn accuracy_1];
    accuracy_pnn = [accuracy_pnn accuracy_2];
%%end

%%
% 2. 结果对比
result = [T_test' result_grnn result_pnn]
accuracy = [accuracy_grnn;accuracy_pnn]
time = [time_grnn;time_pnn]

%% V. 绘图
figure(1)
plot(1:24,T_test,'bo',1:24,result_grnn,'r-*',1:24,result_pnn,'k:^')
grid on
xlabel('测试集样本编号')
ylabel('测试集样本类别')
string = {'测试集预测结果对比(GRNN vs PNN)';['正确率:' num2str(accuracy_grnn*100) '%(GRNN) vs ' num2str(accuracy_pnn*100) '%(PNN)']};
title(string)
legend('真实值','GRNN预测值','PNN预测值')
%figure(2)
%plot(accuracy(1),'r-*',accuracy(2),'b:o')
%grid on
%xlabel('模型编号')
%ylabel('测试集正确率')
%title('511个模型的测试集正确率对比(GRNN vs PNN)')
%legend('GRNN','PNN')
%figure(3)
%plot(time(1,:),'r-*',time(2,:),'b:o')
%grid on
%xlabel('模型编号')
%ylabel('运行时间(s)')
%title('511个模型的运行时间对比(GRNN vs PNN)')
%legend('GRNN','PNN')




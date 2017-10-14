%%带有权值更新的一元函数求极值
%% I. 清空环境
clc
clear all

%% II. 绘制目标函数曲线图
x = 1:0.01:2;%寻找1到2中的极大值
y = sin(10*pi*x) ./ x;
figure
plot(x, y)
hold on

%% III. 参数初始化
c1 = 1.49445;
c2 = 1.49445;

maxgen = 50;   % 进化次数  
sizepop = 10;   %种群规模

Vmax = 0.5;%速度边界，超过会以边界值替代
Vmin = -0.5;
popmax = 2;
popmin = 1;

ws = 0.9;%初始化权值
we = 0.4;

%% IV. 产生初始粒子和速度
for i = 1:sizepop
    % 随机产生一个种群
    pop(i,:) = (rands(1) + 1) / 2 + 1;    %初始种群，rand(1)即随机产生一个数，rand的范围是0-1，这样使范围为1-2
    V(i,:) = 0.5 * rands(1);  %初始化速度,所有列的出现是为了让该矩阵纵向排列
    % 计算适应度
    fitness(i) = fun(pop(i,:));   
end

%% V. 个体极值和群体极值
[bestfitness bestindex] = max(fitness);
zbest = pop(bestindex,:);   %全局最佳
gbest = pop;    %个体最佳（第一代，本身即是最佳）
fitnessgbest = fitness;   %个体最佳适应度值
fitnesszbest = bestfitness;   %全局最佳适应度值

%% VI. 迭代寻优
for i = 1:maxgen
     w = ws - (ws-we)*(i/maxgen);%线性权值改变规则
    for j = 1:sizepop
        % 速度更新
        V(j,:) = w*V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));%群体极值与位置偏差，个体极值与位置偏差
        V(j,find(V(j,:)>Vmax)) = Vmax;%边界约束
        V(j,find(V(j,:)<Vmin)) = Vmin;
        
        % 种群更新
        pop(j,:) = pop(j,:) + V(j,:);
        pop(j,find(pop(j,:)>popmax)) = popmax;
        pop(j,find(pop(j,:)<popmin)) = popmin;
        
        % 适应度值更新
        fitness(j) = fun(pop(j,:)); 
    end
    
    for j = 1:sizepop    
        % 个体最优更新
        if fitness(j) > fitnessgbest(j)
            gbest(j,:) = pop(j,:);
            fitnessgbest(j) = fitness(j);
        end
        
        % 群体最优更新
        if fitness(j) > fitnesszbest
            zbest = pop(j,:);
            fitnesszbest = fitness(j);
        end
    end 
    yy(i) = fitnesszbest;%储存所有迭代次数中每次的种群最优适应度          
end

%% VII. 输出结果并绘图
[fitnesszbest zbest]%输出种群最优适应度以及它对应的种群极值
plot(zbest, fitnesszbest,'r*')%在在原图上标出这个极值点

figure
plot(yy)
title('最优个体适应度','fontsize',12);
xlabel('进化代数','fontsize',12);ylabel('适应度','fontsize',12);


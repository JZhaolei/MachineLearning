%%����Ȩֵ���µ�һԪ������ֵ
%% I. ��ջ���
clc
clear all

%% II. ����Ŀ�꺯������ͼ
x = 1:0.01:2;%Ѱ��1��2�еļ���ֵ
y = sin(10*pi*x) ./ x;
figure
plot(x, y)
hold on

%% III. ������ʼ��
c1 = 1.49445;
c2 = 1.49445;

maxgen = 50;   % ��������  
sizepop = 10;   %��Ⱥ��ģ

Vmax = 0.5;%�ٶȱ߽磬�������Ա߽�ֵ���
Vmin = -0.5;
popmax = 2;
popmin = 1;

ws = 0.9;%��ʼ��Ȩֵ
we = 0.4;

%% IV. ������ʼ���Ӻ��ٶ�
for i = 1:sizepop
    % �������һ����Ⱥ
    pop(i,:) = (rands(1) + 1) / 2 + 1;    %��ʼ��Ⱥ��rand(1)���������һ������rand�ķ�Χ��0-1������ʹ��ΧΪ1-2
    V(i,:) = 0.5 * rands(1);  %��ʼ���ٶ�,�����еĳ�����Ϊ���øþ�����������
    % ������Ӧ��
    fitness(i) = fun(pop(i,:));   
end

%% V. ���弫ֵ��Ⱥ�弫ֵ
[bestfitness bestindex] = max(fitness);
zbest = pop(bestindex,:);   %ȫ�����
gbest = pop;    %������ѣ���һ������������ѣ�
fitnessgbest = fitness;   %���������Ӧ��ֵ
fitnesszbest = bestfitness;   %ȫ�������Ӧ��ֵ

%% VI. ����Ѱ��
for i = 1:maxgen
     w = ws - (ws-we)*(i/maxgen);%����Ȩֵ�ı����
    for j = 1:sizepop
        % �ٶȸ���
        V(j,:) = w*V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));%Ⱥ�弫ֵ��λ��ƫ����弫ֵ��λ��ƫ��
        V(j,find(V(j,:)>Vmax)) = Vmax;%�߽�Լ��
        V(j,find(V(j,:)<Vmin)) = Vmin;
        
        % ��Ⱥ����
        pop(j,:) = pop(j,:) + V(j,:);
        pop(j,find(pop(j,:)>popmax)) = popmax;
        pop(j,find(pop(j,:)<popmin)) = popmin;
        
        % ��Ӧ��ֵ����
        fitness(j) = fun(pop(j,:)); 
    end
    
    for j = 1:sizepop    
        % �������Ÿ���
        if fitness(j) > fitnessgbest(j)
            gbest(j,:) = pop(j,:);
            fitnessgbest(j) = fitness(j);
        end
        
        % Ⱥ�����Ÿ���
        if fitness(j) > fitnesszbest
            zbest = pop(j,:);
            fitnesszbest = fitness(j);
        end
    end 
    yy(i) = fitnesszbest;%�������е���������ÿ�ε���Ⱥ������Ӧ��          
end

%% VII. ����������ͼ
[fitnesszbest zbest]%�����Ⱥ������Ӧ���Լ�����Ӧ����Ⱥ��ֵ
plot(zbest, fitnesszbest,'r*')%����ԭͼ�ϱ�������ֵ��

figure
plot(yy)
title('���Ÿ�����Ӧ��','fontsize',12);
xlabel('��������','fontsize',12);ylabel('��Ӧ��','fontsize',12);


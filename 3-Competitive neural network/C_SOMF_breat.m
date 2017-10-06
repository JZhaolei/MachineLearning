%% I. ��ջ�������
clear all
clc

%% II. ѵ����/���Լ�����
%%
% 1. ��������
load BreastTissue_data.mat

%%
% 2. ���ݹ�һ��
matrix = mapminmax(matrix);
% 3 �������ӣ�ʹ����������Ŀ����һ��
%���ȶ�matrix������������������ӣ�ʹ֮���ﵽ���ֵ22���������ĳ���е����ݣ�������ԭ����ĩβ
n1 = randperm(21);%��һ��ԭ��21��������λ��1-21��
breast1=matrix((1:21),:);%��ȡ��һ����������������
breast1=[breast1;breast1(n1(1),:)];%����������ӣ����������

n2 = randperm(15);%�ڶ���ԭ��15��������λ��22-36��
breast2=matrix((22:36),:);
breast2=[breast2;breast2(n2(1:7),:)];

n3 = randperm(18);%������ԭ��18��������λ��37-54��
breast3=matrix((37:54),:);
breast3=[breast3;breast3(n3(1:4),:)];

n4 = randperm(16);%������ԭ��16��������λ��55-70��
breast4=matrix((55:70),:);
breast4=[breast4;breast4(n4(1:6),:)];

n5 = randperm(14);%������ԭ��14��������λ��71-84��
breast5=matrix((71:84),:);
breast5=[breast5;breast5(n5(1:8),:)];

breast6=matrix((85:106),:);%������ԭ��22��������λ��85-106��

%�����µ��������ݼ�
newbreast=[breast1;breast2;breast3;breast4;breast5;breast6];

%�����µı�ǩ��
newlabel=zeros(size(newbreast,1),1);
for a = 1:6
    newlabel((a-1)*22+1:a*22)= a;
end

% 4 �������ѵ�����Ͳ��Լ�
P_train = [];
T_train = [];
P_test = [];
T_test = [];
for i = 1:6
    temp_input = newbreast((i-1)*22+1:i*22,:);%%�Ż���breast1-6Ҳ����ѭ�����Ͳ�������ȡһ��ÿ�������ˣ�ֱ����breast��i����������
    temp_output = newlabel((i-1)*22+1:i*22,:);%%�Ż�ʱҪע��breast(i)������һά�ģ���Ҳ�Ǹ������������У�ע���±����������
    n = randperm(22);
    % ѵ��������18������
    P_train = [P_train temp_input(n(1:18),:)'];%%X��Y��ת�ã������ж�Ӧÿ���������ж�Ӧ����
    T_train = [T_train temp_output(n(1:18),:)'];
    % ���Լ�����4������
    P_test = [P_test temp_input(n(19:22),:)'];
    T_test = [T_test temp_output(n(19:22),:)'];
end
N=size(T_test,2);

%% III. ���������紴����ѵ�����������
%%
% 1. ��������
net = newc(minmax(P_train),6,0.01,0.01);

%%
% 2. ����ѵ������
net.trainParam.epochs = 500;

%%
% 3. ѵ������
net = train(net,P_train);

%%
% 4. �������

% ѵ����
t_sim_compet_1 = sim(net,P_train);
T_sim_compet_1 = vec2ind(t_sim_compet_1);
% ���Լ�
t_sim_compet_2 = sim(net,P_test);
T_sim_compet_2 = vec2ind(t_sim_compet_2);

%% IV. SOFM�����紴����ѵ�����������
%%
% 1. ��������
net = newsom(P_train,[6 6]);

%%
% 2. ����ѵ������
net.trainParam.epochs = 200;

%%
% 3. ѵ������
net = train(net,P_train);

%%
% 4. �������

% ѵ����
t_sim_sofm_1 = sim(net,P_train);
T_sim_sofm_1 = vec2ind(t_sim_sofm_1);
% ���Լ�
t_sim_sofm_2 = sim(net,P_test);
T_sim_sofm_2 = vec2ind(t_sim_sofm_2);

%% V. ����Ա�
%%
% 1. ����������
result_compet_1 = [T_train' T_sim_compet_1']
result_compet_2 = [T_test' T_sim_compet_2']

%%
% 2. SOFM������
result_sofm_1 = [T_train' T_sim_sofm_1']
result_sofm_2 = [T_test' T_sim_sofm_2']


%figure
%plot(1:N,T_test,'b:*',1:N,T_sim_compet_2,'r-o')
%legend('��ʵֵ','Ԥ��ֵ')
%xlabel('Ԥ������')
%ylabel('Ԥ��ֵ')

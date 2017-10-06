%%һ����ջ�������
clear all;
clc;

%%������������
load('concrete_data.mat');

%�������ѵ�����Ͳ��Լ�
temp = randperm(size(attributes,2));
% ѵ��������80������
P_train = attributes(:,temp(1:80));%��Ϊ������������֤�����
T_train = strength(:,temp(1:80));
% ���Լ�����23������
P_test = attributes(:,temp(81:end));
T_test = strength(:,temp(81:end));
N = size(P_test,2);

%% �������ݹ�һ��
[p_train, ps_input] = mapminmax(P_train,0,1);%ps_input��һ�ֶ�Ӧ��ϵ���������һЩ��Ӧ������ֵ��
p_test = mapminmax('apply',P_test,ps_input);%���������Ӧ��ϵps_input��������ֵ���й�һ���������и�ǰ�ᣬ���������Ҫ��p_trian��min ��max֮�䣬��Ȼ��һ���Ľ������������й�һ��������᲻һ��

[t_train, ps_output] = mapminmax(T_train,0,1);

%% ��. BP�����紴����ѵ�����������
%%
% 1. ��������
net = newff(p_train,t_train,[20,30]);%������Ԫ�ĸ�����һ�����ֿ������ڵ������㣬����������ɶ������㣬��������
%�鿴��view(net)
% 2. ����ѵ������
net.trainParam.epochs = 1000;%��������
net.trainParam.goal = 1e-3;%Ŀ�꣺���������С��1e-3��
net.trainParam.lr = 0.01;%ѧϰ��
% 3. ѵ������
net = train(net,p_train,t_train);
% 4. �������
t_sim = sim(net,p_test);%��ѵ��ģ�ͷ�����Լ�
% 5. ���ݷ���һ��
T_sim = mapminmax('reverse',t_sim,ps_output);%���з���һ��,����һ�������ݷ���һ���ٵõ�ԭ��������:

%% ��. ��������
%%
% 1. ������error
error = abs(T_sim - T_test)./T_test;

%%
% 2. ����ϵ��R^2���ӽ�1����Խ��
R2 = (N * sum(T_sim .* T_test) - sum(T_sim) * sum(T_test))^2 / ((N * sum((T_sim).^2) - (sum(T_sim))^2) * (N * sum((T_test).^2) - (sum(T_test))^2)); 

%%
% 3. ����Ա�
result = [T_test' T_sim' error']

%% VI. ��ͼ
figure
plot(1:N,T_test,'b:*',1:N,T_sim,'r-o')
legend('��ʵֵ','Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ��ֵ')
%string = {'���Լ�����ֵ����Ԥ�����Ա�';['R^2=' num2str(R2)]};
%title(string)



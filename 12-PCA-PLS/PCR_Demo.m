%% I. ��ջ�������
clear all
clc

%% II. ��������
load spectra;

%% III. �������ѵ��������Լ�
temp = randperm(size(NIR, 1));
% temp = 1:60;
%%
% 1. ѵ��������50������
P_train = NIR(temp(1:50),:);
T_train = octane(temp(1:50),:);
%%
% 2. ���Լ�����10������
P_test = NIR(temp(51:end),:);
T_test = octane(temp(51:end),:);

%% IV. ���ɷַ���
%%
% 1. ���ɷֹ����ʷ���
[PCALoadings,PCAScores,PCAVar] = princomp(NIR);%PCAVarΪ���ɷֵ�����ֵ��PCALoadingԭ����ӳ�䵽�¿ռ�����ɷֵ�ϵ����PCAScores����ԭ����ӳ�䵽�¿ռ��Ӧ��ϵ��
% [COEFF, SCORE, LATENT] = PRINCOMP(X) 
% LATENTЭ������������ֵ��
% SCORE�Ƕ����ֵĴ�֣�Ҳ����˵ԭX���������ɷֿռ�ı�ʾ��
% COEFF��X��������Ӧ��Э������V����������������ɵľ��󣬼��任������ͶӰ����
% �����ԭ����x*coeff(:,1:n)������Ҫ�ĵ������ݣ����е�n�����뽵������ά��
figure
percent_explained = 100 * PCAVar / sum(PCAVar);%���ɷֵĹ�����
pareto(percent_explained)%Paretoͼ�ֳ�����ͼ��һ�ְ��¼�������Ƶ��������ɣ���ʾ���ڸ���ԭ�������ȱ��������һ�µ�����˳�����ҳ�Ӱ����Ŀ��Ʒ�������������Ҫ���صķ�����
xlabel('���ɷ�')
ylabel('������(%)')
title('���ɷֹ�����')

%%
% 2. ��һ���ɷ�vs.�ڶ����ɷ֣��ж����������Ƿ��׼�������Լ��ĵ㶼��ѵ�����ڰ������򻮷ֽϺã�
[PCALoadings,PCAScores,PCAVar] = princomp(P_train);
figure
plot(PCAScores(:,1),PCAScores(:,2),'r+')
hold on
[PCALoadings_test,PCAScores_test,PCAVar_test] = princomp(P_test);
plot(PCAScores_test(:,1),PCAScores_test(:,2),'o')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
legend('Training Set','Testing Set','location','best')

%% V. ���ɷֻع�ģ��
%%
% 1. ����ģ��
k = 4;
betaPCR = regress(T_train-mean(T_train),PCAScores(:,1:k));%TΪY��PCAScoresΪX��������ǻع�ϵ��
betaPCR = PCALoadings(:,1:k) * betaPCR;%�����������������Ϊ��Ԥ����׼���������ľ�Ϊ�̶�����ģʽ
betaPCR = [mean(T_train)-mean(P_train) * betaPCR;betaPCR];
%%
% 2. Ԥ�����
N = size(P_test,1);
T_sim = [ones(N,1) P_test] * betaPCR;

%% VI. ����������ͼ
%%
% 1. ������error
error = abs(T_sim - T_test) ./ T_test;
%%
% 2. ����ϵ��R^2
R2 = (N * sum(T_sim .* T_test) - sum(T_sim) * sum(T_test))^2 / ((N * sum((T_sim).^2) - (sum(T_sim))^2) * (N * sum((T_test).^2) - (sum(T_test))^2)); 
%%
% 3. ����Ա�
result = [T_test T_sim error]

%% 
% 4. ��ͼ
figure
plot(1:N,T_test,'b:*',1:N,T_sim,'r-o')
legend('��ʵֵ','Ԥ��ֵ','location','best')
xlabel('Ԥ������')
ylabel('����ֵ')
string = {'���Լ�����ֵ����Ԥ�����Ա�';['R^2=' num2str(R2)]};
title(string)



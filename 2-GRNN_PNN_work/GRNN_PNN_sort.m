%% I. ��ջ�������
clear all
clc

%% II. ѵ����/���Լ�����
%%
% 1. ��������
load BreastTissue_data.mat

%%
% 2 �������ӣ�ʹ����������Ŀ����һ��
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

% 3 �������ѵ�����Ͳ��Լ�
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

%% III. ģ�ͽ��� 
result_grnn = [];
result_pnn = [];
time_grnn = [];
time_pnn = [];
%%for i = 1:9
    %%for j = i:9
      %  p_train = P_train(i:j,:);%%������������������������������ȫ��������ϣ�������ͬ��ģ�ͣ��Ա���Щ��������Ч����
       % p_test = P_test(i:j,:);
       %% 
        p_train = P_train;
        p_test =P_test;
        % 1. GRNN�������������
        t = cputime;
        % ��������
        net_grnn = newgrnn(p_train,T_train);
        % �������
        t_sim_grnn = sim(net_grnn,p_test);
        T_sim_grnn = round(t_sim_grnn);
        t = cputime - t;
        time_grnn = [time_grnn t];
        result_grnn = [result_grnn T_sim_grnn'];
       %%
        % 2. PNN�������������
        t = cputime;
        Tc_train = ind2vec(T_train);
        % ��������
        net_pnn = newpnn(p_train,Tc_train);
        % �������
        Tc_test = ind2vec(T_test);
        t_sim_pnn = sim(net_pnn,p_test);
        T_sim_pnn = vec2ind(t_sim_pnn);
        t = cputime - t;%%��ֵ�����������ʱ��
        time_pnn = [time_pnn t];
        result_pnn = [result_pnn T_sim_pnn'];
    %%end
%%end

%% IV. ��������
%%
% 1. ��ȷ��accuracy
accuracy_grnn = [];
accuracy_pnn = [];
time = [];
%%for i = 1:511  %9�������ĸ����������
    %accuracy_1 = length(find(result_grnn(:,i) == T_test'))/length(T_test);
    %accuracy_2 = length(find(result_pnn(:,i) == T_test'))/length(T_test);
    accuracy_1 = length(find(result_grnn == T_test'))/length(T_test);
    accuracy_2 = length(find(result_pnn == T_test'))/length(T_test);
    accuracy_grnn = [accuracy_grnn accuracy_1];
    accuracy_pnn = [accuracy_pnn accuracy_2];
%%end

%%
% 2. ����Ա�
result = [T_test' result_grnn result_pnn]
accuracy = [accuracy_grnn;accuracy_pnn]
time = [time_grnn;time_pnn]

%% V. ��ͼ
figure(1)
plot(1:24,T_test,'bo',1:24,result_grnn,'r-*',1:24,result_pnn,'k:^')
grid on
xlabel('���Լ��������')
ylabel('���Լ��������')
string = {'���Լ�Ԥ�����Ա�(GRNN vs PNN)';['��ȷ��:' num2str(accuracy_grnn*100) '%(GRNN) vs ' num2str(accuracy_pnn*100) '%(PNN)']};
title(string)
legend('��ʵֵ','GRNNԤ��ֵ','PNNԤ��ֵ')
%figure(2)
%plot(accuracy(1),'r-*',accuracy(2),'b:o')
%grid on
%xlabel('ģ�ͱ��')
%ylabel('���Լ���ȷ��')
%title('511��ģ�͵Ĳ��Լ���ȷ�ʶԱ�(GRNN vs PNN)')
%legend('GRNN','PNN')
%figure(3)
%plot(time(1,:),'r-*',time(2,:),'b:o')
%grid on
%xlabel('ģ�ͱ��')
%ylabel('����ʱ��(s)')
%title('511��ģ�͵�����ʱ��Ա�(GRNN vs PNN)')
%legend('GRNN','PNN')




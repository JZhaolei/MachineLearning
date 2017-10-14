function y = fun(x)
%函数用于计算粒子适应度值（求最大值，直接写原函数即可）
%x           input           输入粒子 
%y           output          粒子适应度值 
y = x(1).^2 + x(2).^2 - 10*cos(2*pi*x(1)) - 10*cos(2*pi*x(2)) + 20;




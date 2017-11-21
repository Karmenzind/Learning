function res=calrow(ina,yangbenshu)
[r,c]=size(ina);
%yangbenshu=110-yangbenjiange;
re=ones(r - yangbenshu,6);
a=yangbenshu + 1:r;
b=a';
re(:,1)=b;
for i=1:(r - yangbenshu)
    for j=2:6
%arry inp preprocess
in=ina(i:yangbenshu-1+i,1)';%time as inp
ou=ina(i:yangbenshu-1+i,j)';%col2 as outp
% 
%[inputn,inputps]=mapminmax(in);
%inputn_test=mapminmax('apply',in,inputps);
%input
%[outputn,outputps]=mapminmax(ou);
%outputn_test=mapminmax('apply',ou,outputps);
%output
net=newff(in,ou,[3 6],{'tansig','purelin'},'trainlm');%
net.trainParam.lr=0.0005;
net.trainParam.epochs=5000;
net.trainParam.goal=1e-5;
[net,tr]=train(net,in,ou);%
%sim
si=in+ones(1,yangbenshu);%
%[simputn,simputps]=mapminmax(si);
%simputn_test=mapminmax('apply',si,simputps);
%sim process end
y=sim(net,si);
%y1=mapminmax('reverse',y,outputps);
re(i,j)=y(1,yangbenshu);%y1
    end
end
res=re;
end
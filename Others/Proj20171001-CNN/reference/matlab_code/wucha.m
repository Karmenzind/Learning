function [jieguo,wucha,wuchap]=wucha(jiange,ina)
a=ina(jiange+1:end,:);
ans1=calrow(ina,jiange);
deta=a-ans1;
detap=deta./a;
jieguo=ans1;
wucha=deta;
wuchap=detap;
end
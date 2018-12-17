
close all;
%%MAXIMUM
%FITNESS FUNCTION:
ff=inline('4*x.^4 - 5*x.^3 + exp(-2.*x) - 7.*sin(x) - 3.*cos(x)');

n=500;%max number of generations1
popsize=40; %kich thuoc quan the
mutrate=0.01; %ti le dot bien
nbits=10; %so luong bit ma hoa

%Khoi tao quan the
popul=round(rand(popsize,nbits));
%tao mang theo doi gia tri max o tung the he
best=zeros;
%tao mang theo doi gia tri x tai gia tri max o tung the he
best_x = zeros;

avg_fitness=zeros; %finess trung binh tung the he
buffer = zeros(1,20); %bo nho tam luu max 20 the he gan nhat
deviation = zeros; %do lech chuan tung the he
f_tmp = zeros(1,popsize/2); %bo nho tam luu 1/2 quan the sau chon loc

%EVOLUTION
iter=0;
while iter<n
 iter=iter+1;
%Giai ma binary => real
pop_dec= (bi2de(popul))*5.0/1023.0;
%Tinh toan gia tri cua ham so
f=feval(ff,pop_dec');
%Sap xep gia tri cua ham so theo thu tu giam dan
[f,ind]=sort(f, 'descend');
%Sap xep lai quan the theo thu tu cua ham so
% => thuc hien chon loc, bo 1/2 quan the phia sau
popul=popul(ind,:);
pop_dec= (bi2de(popul))*5.0/1023.0;

avg_fitness(iter)=sum(f)/popsize; %Fitness trung binh cua quan the
best(iter)=f(1); %gia tri max tai moi the he
best_x(iter) = pop_dec(1); %gia tri x ung voi max 

%Selection of parents:
%chon dad & mom tu 1->20 phan tu tot dau tien cua quan the
% 1 cach ngau nhien
dad = round((rand(1,popsize/2))*19)+1;
mom = round((rand(1,popsize/2))*19)+1;

xp=7; %so bit giao phoi
%Lay cua dad tu 1->xp bit + mom tu xp+1-> nbits
popul(1:2:popsize,:)=[popul(dad,1:xp)...
popul(mom,xp+1:nbits)];
popul(2:2:popsize,:)=[popul(mom,1:xp)...
popul(dad,xp+1:nbits)];

%tim sai so cua max trong 20 chu ki gan nhat
if iter>=20
    buffer = best(1,iter-19:1:iter);
    deviation(iter-19) = std(buffer,0,2);
    clear std;
    if deviation(iter-19)<0.001
    max = best(iter)
    x = best_x(iter)
    break
    end
end

%Dot bien
nmut=ceil(popsize*nbits*mutrate);
for i=1:nmut
 col=randi(nbits);
 row=randi(popsize);
 if popul(row,col)==1
 popul(row,col)=0;
 else
 popul(row,col)=1;
 end 
end
end

%Quan sat gia tri max qua moi the he
figure
t=1:iter;
plot(t,best)
xlabel('number of generations');
ylabel('y-max');
%Quan sat fitness trung binh qua moi the he
figure
plot(t,avg_fitness)
xlabel('number of generations');
ylabel('average fitness');

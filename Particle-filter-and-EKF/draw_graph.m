

figure
%Ve XTRUE va XODO
subplot(2,2,1);
axis([-25 20 -5 45]);
hold on
plot(x_t(1,:),x_t(2,:),'r','LineWidth',1);
plot(lm(1,:),lm(2,:),'o');
plot(xodo(1,:),xodo(2,:),':g','LineWidth',1.5);
title('Observation');
xlabel('x');
ylabel('y');
legend('true path','landmark',['RMSobserve = ',num2str(RMS_observe)],'Location','southwest');
   
%Ve quy dao particle max    
subplot(2,2,2);
axis([-25 20 -5 45]);
hold on
plot(x_t(1,:),x_t(2,:),'r','LineWidth',1);
plot(lm(1,:),lm(2,:),'o');
plot(max_par(1,:),max_par(2,:),'b','LineWidth',0.5);
title('Max particle');
xlabel('x');
ylabel('y');
legend('true path','landmark',['RMSmax = ',num2str(RMS_max)],'Location','southwest');

%Ve quy dao particle medium
subplot(2,2,3);
axis([-25 20 -5 45]);
hold on
plot(x_t(1,:),x_t(2,:),'r','LineWidth',1);
plot(lm(1,:),lm(2,:),'o');
plot(medium_par(1,:),medium_par(2,:),'b','LineWidth',0.5);
title('Medium particle');
xlabel('x');
ylabel('y');
legend('true path','landmark',['RMSmed = ',num2str(RMS_medium)],'Location','southwest');

%Ve quy dao particle min
subplot(2,2,4);
axis([-25 20 -5 45]);
hold on
plot(x_t(1,:),x_t(2,:),'r','LineWidth',1);
plot(lm(1,:),lm(2,:),'o');
plot(min_par(1,:),min_par(2,:),'k','LineWidth',0.5);
title('Min particle');
xlabel('x');
ylabel('y');
legend('true path','landmark',['RMSmin = ',num2str(RMS_min)],'Location','southwest');


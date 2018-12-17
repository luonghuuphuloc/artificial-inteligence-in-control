
clear all
parameter
for i = 2:step;
    X_pre = X_bar(:,i-1);       %cap nhat X tai k-1
    P_pre = P_bar(:,:,i-1);     %cap nhat P tai k-1
    
    u(:,i) = VG(:,i) + deviation_vg*randn(2,1); %tao nhieu tin hieu u
    %tinh theo process model vi tri tiep theo
    new_pos = predict_model(u(1,i),u(2,i),delta_t,X_pre); 
    %tinh covariance matrix Q
    Gu = [delta_t*cos(X_pre(3)+u(2,i)), -u(1,i)*delta_t*sin(X_pre(3)+u(2,i));...
        delta_t*sin(X_pre(3)+u(2,i)), u(1,i)*delta_t*cos(X_pre(3)+u(2,i));...
        delta_t*sin(u(2,i))/4, u(1,i)*delta_t*cos(u(2,i))/4];
    Q = Gu*[sig_v^2 0;0 sig_s^2]*Gu';
    %----------------------------------------------------------------
    laser_step = Z(:,:,i); %cap nhat gia tri tu sensor
    
    if (isnan(laser_step(1,1))) %cap nhat khi khong co du lieu sensor
        x = new_pos;
        P = P_pre;
    else                        %khi co du lieu tu sensor
        sig_P = zeros(3);
        sig_K = zeros(3,2);
        sig_z = zeros(2,1);
        landmark_num = 0;
        result = [];
        %trich xuat du lieu sensor luu vao result
        for m = 1:size(laser_step,2)
            if(isnan(laser_step(1,m)) == false)
                landmark_num = landmark_num + 1; %so luong  landmark duoc phat hien
                result = [result laser_step(:,m)];
            end
        end
        %---------------------------------------------------------------
        %Tinh X vs P uoc luong
        X_predict =  new_pos;
        X_temp = -X_pre + new_pos;
        G = [1 0 -X_temp(2);    %-v*delta_t*sin(phi_t-1 + theta)
            0 1 X_temp(1);      %v*delta_t*cos(phi_t-1 + theta)
            0 0 1];
        P_predict = G*P_pre*G' + Q;
        %--------------------------------------------------------------
        for m = 1: landmark_num
            z = result(1:2,m);      %lay gia tri range va bearing tu sensor
            xy_landmark = lm(:,result(3,m));
            
            x_term = X_predict(1) - xy_landmark(1); %x_t-x_lm
            y_term = X_predict(2) - xy_landmark(2); %y_t-y_lm
            x_distance = sqrt(x_term^2+y_term^2);
            H = [x_term/x_distance y_term/x_distance 0; %tinh ma tran H
                -y_term/x_distance^2 x_term/x_distance^2 -1];
            
            %tong tich luy K de tinh trung binh
            sig_K = sig_K + P_predict*H'*inv(H*P_predict*H'+R); 
            dx = xy_landmark(1)-X_predict(1);
            dy = xy_landmark(2)-X_predict(2);
            phi = X_predict(3);
            z_estimate = [sqrt(dx^2+dy^2);  %range
                atan2(dy,dx)-phi];          %bearing
            z(2) = z(2) - 2*pi*floor((z(2)+pi)/2/pi);
            z_estimate(2) = z_estimate(2) - 2*pi*floor((z_estimate(2)+pi)/2/pi);
            sig_z = sig_z + (z - z_estimate);
        end
        
        K = sig_K/landmark_num; %kalman gain K_t
        x = X_predict + K*sig_z/landmark_num;   %correct X_t
        %-----------------------------------------------------------
        I3 = eye(3);
        for m = 1: landmark_num
            xy_landmark = lm(:,result(3,m));
            x_term = X_predict(1) - xy_landmark(1);
            y_term = X_predict(2) - xy_landmark(2);
            x_distance = sqrt(x_term^2+y_term^2);
            H = [x_term/x_distance y_term/x_distance 0;
                -y_term/x_distance^2 x_term/x_distance^2 -1];
            sig_P = sig_P + (I3 - K*H);
        end
        P = (sig_P/landmark_num)*P_predict;
    end
    X_bar(:,i) = x;
    P_bar(:,:,i) = P;
end
%---------------------------------------------------------------------
%Tinh RMS
RMS_observe = 0;
RMS_kalman = 0;
for m = 1:step
    RMS_observe = RMS_observe + (XODO(1,m)-XTRUE(1,m))^2 ...
        + (XODO(2,m)-XTRUE(2,m))^2;
    RMS_kalman = RMS_kalman + (X_bar(1,m)-XTRUE(1,m))^2 ...
        + (X_bar(2,m)-XTRUE(2,m))^2;
end
RMS_observe = sqrt(RMS_observe/step);
RMS_kalman = sqrt(RMS_kalman/step);

%---------------------------------------------------------------------
%ve do thi
figure
hold on
title('Extended Kalman Filter Path');
xlabel('x');
ylabel('y');
plot(XTRUE(1,:),XTRUE(2,:),'r','LineWidth',1);
plot(lm(1,:),lm(2,:),'o');
plot(XODO(1,:),XODO(2,:),'g:','LineWidth',2);
plot(X_bar(1,:),X_bar(2,:),'b','LineWidth',1);
legend('True path','Landmark',...
    ['Observation, RMS = ',num2str(RMS_observe)],...
    ['EKF result, RMS = ',num2str(RMS_kalman)],'Location','southwest');

clear all
%khoi tao gia tri
parameter

%---------------------------------------------------------------
%khoi tao particle
for m = 1:num_particle %buoc dau tien tat ca particle deu o cung vi tri
particle(:,m,1) = xodo(:,1);
end
%----------------------------------------------------------------
for i = 2:step
    particle_pos = zeros(3,num_particle);
        for j = 1:num_particle %tinh process model ket hop nhieu tu V va S cho cac particle
            u(:,i) = VG(:,i) + deviation_vg*randn(2,1); %tao nhieu tin hieu u
            particle_pos(:,j) = predict_model(u(1,i),u(2,i),delta_t,particle(:,j,i-1));
    %tinh particle moi tu gia tri truoc cong voi nhieu V, G        
        end
        
    laser_step = Z(:,:,i);
    if (isnan(laser_step(1,1))) % khi khong co gia tri cam bien
        particle(:,:,i) = particle_pos;
    else                        % khi co gia tri cam bien
        landmark_num = 0;
        result = [];
        for mark = 1:size(laser_step,2)
            if(isnan(laser_step(1,mark)) == false)
                landmark_num = landmark_num + 1; %so landmark phat hien
                result = [result laser_step(:,mark)]; %trich vi tri landmark
            end
        end
    %--------------------------------------------------------------
    %tinh trong so cho w_t
        for k = 1:num_particle
            weight_each_par = 1;
                for mark = 1: landmark_num
                    z = result(1:2,mark);
                    z(2) = z(2) - 2*pi*floor((z(2)+pi)/2/pi);
                    xy_predict = particle_pos(1:2,k);
                    xy_landmark = lm(:,result(3,mark));
                    %tinh r_t
                    x_range = sqrt((xy_predict(1) - xy_landmark(1)).^2+(xy_predict(2) - xy_landmark(2)).^2);
                    %tinh b_t
                    x_bearing = - particle_pos(3,k) + atan2(xy_landmark(2)-xy_predict(2),xy_landmark(1)-xy_predict(1));
                    x = [x_range; x_bearing];
                    x(2) = x(2) - 2*pi*floor((x(2)+pi)/2/pi);
                    %tinh trong so theo cong thuc P(zt/xt)
                    weight_each_par = weight_each_par/sqrt(det(2*pi*cov_mat_r_b))*exp(-0.5*(z-x)'/inv(cov_mat_r_b)*(z-x));
                end
                w_t(k) = weight_each_par;
        end
     %--------------------------------------------------------------
     %tao mang cap nhat gia tri cac particle lon nhat, trung binh va nho
     %nhat
        [~,index] = sort(w_t);
        max_par = [max_par particle_pos(:,index(num_particle))]; %paticle trong so lon nhat
        medium_par = [medium_par particle_pos(:,index(floor(num_particle/2)))];%particle trung binh
        min_par = [min_par particle_pos(:,index(1))]; %particle nho nhat
        correct_time = [correct_time,i];
     %--------------------------------------------------------------
     %Resampling theo phuong phap Russian Roullett
        roullett_weight = cumsum(w_t); %tong tich luy cac trong so
        roullett_weight = roullett_weight/roullett_weight(100); %chuan hoa ve [0 1]
        for l = 1: num_particle
            random_num = rand; %chon ngau nhien trong [0 1]
            high_area = find(roullett_weight >= random_num,1,'first');
            particle(:,l,i) = particle_pos(:,high_area);
        end
     %--------------------------------------------------------------
     
    end
end

%---------------------------------------------------------------------
%Tinh RMS cho XODO
for o = 1:step
    RMS_observe = RMS_observe + (XODO(1,o)-XTRUE(1,o))^2 + ...
        (XODO(2,o)-XTRUE(2,o))^2;
end
RMS_observe = sqrt(RMS_observe/step);
%Tinh RMS cho 3 particle
for n = 1:length(correct_time)
    RMS_min = RMS_min + (min_par(1,n)-XTRUE(1,correct_time(n)))^2 ...
        +(min_par(2,n)-XTRUE(2,correct_time(n)))^2;
    RMS_medium = RMS_medium + (medium_par(1,n)-XTRUE(1,correct_time(n)))^2 ...
        +(medium_par(2,n)-XTRUE(2,correct_time(n)))^2;
    RMS_max = RMS_max + (max_par(1,n)-XTRUE(1,correct_time(n)))^2 ...
        +(max_par(2,n)-XTRUE(2,correct_time(n)))^2;
end
RMS_min = sqrt(RMS_min/length(correct_time));
RMS_medium = sqrt(RMS_medium/length(correct_time));
RMS_max = sqrt(RMS_max/length(correct_time));


%--------------------------------------------------------------------
% ve do thi 
draw_graph



load('data20171107.mat')
%parameter cho particle filter
num_particle = 100;         %so luong particle
sig_v = 0.5;                %do lech chuan velocity
sig_s = 3*pi/180;           %do lech chuan steering angle
sig_r = 0.2;                %do lech chuan range
sig_b = 2*pi/180;           %do lech chuan bearing
cov_mat_r_b = [sig_r^2 0;0 sig_b^2];
delta_t = 0.025;            %time step
step = 625;
wb = 4;                     %chon ngau nhien

x_t = XTRUE;
xodo = XODO;
u = VG;                     %tin hieu dieu khien
particle = zeros(3,num_particle,step); %ma tran luu vi tri, step tat ca particle      
w_t = zeros(1,num_particle);%ma tran luu trong so 100 particle 

max_par = [];               %luu gia tri cac max particle
medium_par = [];            %medium particle
min_par = [];               %min particle
correct_time = [];

deviation_vg = [sig_v 0;0 sig_s];   %khoi tao ma tran deviation cho V, S
                                     %de tao random control signal cho moi
                                     %particle
%Khoi tao gia tri RMS
RMS_min = 0;
RMS_medium = 0;
RMS_max = 0;
RMS_observe = 0;
%--------------------------------------------------------------------------
%parameter cho kalman filter
%Tim do lech chuan cho x,y,phi

%Tao covariance matrix
%Q = diag([sig_x^2, sig_y^2, sig_phi^2]); %Phuong sai cua nhieu he thong
R = diag([sig_r^2, sig_b^2]); %Phuong sai cua nhieu do luong
%Khoi tao X, P
X_bar = zeros(3,step);
P_bar = zeros(3,3,step);
X_bar(:,1) = xodo(:,1);
P_bar(:,:,1) = 0.5*eye(3);
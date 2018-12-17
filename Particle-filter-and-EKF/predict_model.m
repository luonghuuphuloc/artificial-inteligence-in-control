%process model
function out = predict_model(v,theta,delta_t,pos_last)
wb = 4;
x = pos_last(1,1) + v*delta_t*cos(theta + pos_last(3,1));
y = pos_last(2,1) + v*delta_t*sin(theta + pos_last(3,1));
phi = pos_last(3,1) + (v*delta_t*sin(theta))/wb;
out = [x;y;phi];
end
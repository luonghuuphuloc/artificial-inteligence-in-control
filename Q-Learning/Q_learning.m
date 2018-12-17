
clear all
clc
R = [-1 -1 -1 -1 0 -1;
    -1 -1 -1 0 -1 100;
    -1 -1 -1 0 -1 -1;
    -1 0 0 -1 0 -1;
    0 -1 -1 0 -1 100;
    -1 0 -1 -1 0 100]; %Reward
%initial Q-matrix:
Q = zeros(6,6);
Q_temp = zeros(6,6);
episode = 0;
gammar = 0.8;
converge = 0;

while converge ~= 1
    next_state = 0;
    initial_state = randi(6);
    current_state = initial_state;
    while(next_state ~= 6)
        possible_action = [];
        for index=1:6
           if (R(current_state,index)~= -1)           
               temp = [R(current_state,index);index];
               possible_action = [possible_action temp];
           end
        end
        %possible_action
       [temp,number_action] = size(possible_action);
       selected_action = possible_action(:,randi(number_action));

    %kiem tra state tiep theo co bao nhieu hanh dong
       next_state = selected_action(2,1);
       possible_action_next = [];
       for ind = 1:6
           if R(next_state,ind) ~= -1
               temp = [Q_temp(next_state,ind);ind];
               possible_action_next = [possible_action_next temp];
           end
       end
       %possible_action_next
       max_Q_next = max(possible_action_next(1,:));
       Q_temp(current_state,next_state) = R(current_state,next_state) + gammar*max_Q_next;
       current_state = next_state;
       %Q_temp
    end
    %if Q_temp == Q
    if episode == 1000
        converge = 1;
        
    else 
        %Q = Q_temp;
        episode = episode + 1;
    end
    
    
end
 Q = Q_temp*100/max(max(Q_temp))
%Q_temp;

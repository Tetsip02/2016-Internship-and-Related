function [state] = what_state(x, x_dot, theta, theta_dot)
  %%%% This particular discretization of the states is adapted from the 
  %%%% RL repository at http://www-anw.cs.umass.edu/rlr/domains.html
  %%%% In particular, x, x_dot and theta_dot are divided into 3 boxes, 
  %%%% and theta into 6 boxes, resulting into 162 discrete states
  
  %Parameters
  one_degree = 0.0174532;	% 2pi/360
  six_degrees = 0.1047192;
  twelve_degrees = 0.2094384;
  fifty_degrees = 0.87266;
  state=0;
  num_states = 163;  %162 + failure state
  failure_state = num_states;
  
  if (x < -2.4 || x > 2.4 || theta > twelve_degrees || theta < -twelve_degrees)
    state = failure_state;
  else
    if (x < -1.5)
      state = 1;
    elseif (x < 1.5)
      state = 2;
    else
      state = 3;
    end
    
    if (x_dot < -0.5)
      ;
    elseif (x_dot < 0.5)
      state += 3;
    else
      state += 6;
    end
    
    if (theta < -six_degrees)
      ;
    elseif (theta < one_degree)
      state += 9;
    elseif (theta < 0)
      state += 18;
    elseif (theta < one_degree)
      state += 27;
    elseif (theta < six_degrees)
      state += 36;
    else
      state += 45;
    end
    
    if (theta_dot < -fifty_degrees)
      ;
    elseif (theta_dot < fifty_degrees)
      state += 54;
    else
      state += 108;
    end
  end
endfunction
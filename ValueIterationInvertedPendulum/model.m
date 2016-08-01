%%%% model
%%%% equations of motion for the inverted pendulum on a cart are:
%%%% F = (M+m)*x_acc - m*l*theta_acc*costheta + m*l*theta_dot*theta_dot*sintheta
%%%% l*theta_acc - g*sintheta = x_acc*costheta

function [new_x, new_x_dot, new_theta, new_theta_dot] = model(action, x, x_dot, theta, theta_dot)
    %parameters
    g = 9.8;  %acceleration due to gravity
    M = 1.0;  %mass of the cart2
    m = 0.3;  %point mass at the end of the pole
    total_m = M + m;
    l = 0.7;  %length of the pole
    F = 10;
    tau = 0.02;  %time increment
    
    sintheta = sin(theta);
    costheta = cos(theta);
    
    if action == 1
        F = -1 * F;
    else
        ;
    end
    
    %angular acceleration
    det_theta_acc = (total_m * l) - (m * l * costheta * costheta);
    nom_theta_acc = (F * costheta) - (m * l * theta_dot * theta_dot * sintheta * costheta) + (g * total_m *sintheta);
    theta_acc = nom_theta_acc / det_theta_acc;
    %acceleration
    %x_acc = ((l * theta_acc) - (g * sintheta)) / costheta;
    x_acc = (F - (m * l * theta_acc * costheta) + (m * l * theta_dot * theta_dot * sintheta)) / total_m;
    
    %update
    new_x = x + tau * x_dot;
    new_x_dot = x_dot + tau * x_acc;
    new_theta = theta + tau * theta_dot;
    new_theta_dot = theta_dot + tau * theta_acc;
    
endfunction
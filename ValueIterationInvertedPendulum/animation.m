%animation
function [] = animation(x, theta, pause_time)

    %pole
    length = 1.5;
    pole_x(1) = x;
    pole_x(2) = x + length*sin(theta);
    pole_y(1) = 0;
    pole_y(2) = length*cos(theta);
    plot(pole_x,pole_y);

    %cart
    rectangle('Position', [x-0.4, -0.15, 0.8, 0.15], 'Curvature', 1, 'FaceColor', 'cyan');

    %scope of the plot
    axis([-3 3 -0.15 1.8]);

    drawnow;  %i think this is necesarry to not load a new figure with each frame
    pause(pause_time);

endfunction
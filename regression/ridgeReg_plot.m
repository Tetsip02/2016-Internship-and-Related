load('X.out');
load('y.out');
load('theta_ridgeReg.out');
plot(X,y,'.b');
regline_X = min(X):0.1:max(X);
regline_y = theta_ridgeReg(2)*regline_X + theta_ridgeReg(1);
hold on;
plot(regline_X,regline_y);
load('theta_logReg.out');
load('X.out');
load('y.out');
figure;
hold on;
for i=1:length(X)
    if y(i) == 0
        plot(X(i,1),X(i,2),'.b');
    else
        plot(X(i,1),X(i,2),'.g');
    end
end
%plot theta(1) + theta(2)*X(:,1) + theta(3)*X(:,2) = 0
%or X(:,2) = (-theta(1) - theta(2)*X(:,2))/theta(3)
bound_X = X(:,1);
bound_y = (-theta_logReg(1) - theta_logReg(2)*X(:,1))/theta_logReg(3);
plot(bound_X,bound_y,'r');
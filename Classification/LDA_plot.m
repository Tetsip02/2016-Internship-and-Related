load('X_GDA.out');
load('y_GDA.out');
load('phy.out');
load('mu0.out');
load('mu1.out');
load('cov.out');
%decision boundary for LDA (y=ax+b). 
%For a pair of classes this can be obtained in the following way:
%it must pass through the midpoint of their respective means, 
%ie 1/2(mu0+mu1), and be perpendiclar to SIGMA^-1*(mu0-mu1)
midpoint = 0.5 * (mu0 +mu1);
gradient = (mu0 - mu1) * inv(cov);
a = -gradient(2) / gradient(1);
b = midpoint(2) - midpoint(1) * a;
figure;
hold on;
for i=1:length(X_GDA)
    if y_GDA(i) == 0
        plot(X_GDA(i,1),X_GDA(i,2),'.b');
    else
        plot(X_GDA(i,1),X_GDA(i,2),'.g');
    end
end
bound_X = min(X_GDA(:,1)):0.01:max(X_GDA(:,1));
bound_y = a * bound_X + b;
plot(bound_X,bound_y,'r');


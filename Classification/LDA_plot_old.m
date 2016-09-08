load('X_GDA.out');
load('y_GDA.out');
load('phy.out');
load('mu0.out');
load('mu1.out');
cov = load('cov0.out');
%decision boundary for LDA (y=ax+b). 
%The line passes through the midpoint of the means.
%When w is SIGMA^-1*(mu0-mu1) the seperation is maximised (see slides
%34 to 40)
midpoint = 0.5 * (mu0 +mu1);
w = (mu0 - mu1) * inv(cov);
a = -w(2) / w(1);
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


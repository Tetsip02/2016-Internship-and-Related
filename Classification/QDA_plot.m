load('X_GDA.out');
load('y_GDA.out');
load('phy.out');
load('mu0.out');
load('mu1.out');
load('cov0.out');
load('cov1.out');
%decision boundary can be obtained by taking the log of the likelyhood ratio
%and setting it equal to zero. (see slides 7 and 8)
a=linspace(min(X_GDA(:,1)),max(X_GDA(:,1)));
b=linspace(min(X_GDA(:,2)),max(X_GDA(:,2)));
[A B] = meshgrid(a,b);
AB = [A(:) B(:)];
Z = zeros(size(A(:))); %Z=trans(x)*A*x +trans(x)*B + c
c = log((1-phy)/phy) - 0.5*log(det(cov0)/det(cov1))...
        -0.5*((mu0*inv(cov0)*mu0')-(mu1*inv(cov1)*mu1'));
for i=1:size(A(:))
    Z(i) = c + ((mu0*inv(cov0))-(mu1*inv(cov1)))*AB(i,:)'...
        - 0.5*AB(i,:)*(inv(cov0)-inv(cov1))*AB(i,:)';
end
Z = reshape(Z, size(A));

figure;
hold on;
for i=1:length(X_GDA)
    if y_GDA(i) == 0
        plot(X_GDA(i,1),X_GDA(i,2),'.b');
    else
        plot(X_GDA(i,1),X_GDA(i,2),'.g');
    end
end
contour(A,B,Z,[0 0]); %draw contour only where Z is zero

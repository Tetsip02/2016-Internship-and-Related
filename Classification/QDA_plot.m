load('X_GDA.out');
load('y_GDA.out');
load('phy.out');
load('mu0.out');
load('mu1.out');
load('cov0.out');
load('cov1.out');
%decision boundary can be obtained by taking the log of the posterior
%probability and setting it equal to zero.
a=linspace(min(X_GDA(:,1)),max(X_GDA(:,1)));
b=linspace(min(X_GDA(:,2)),max(X_GDA(:,2)));
[A B] = meshgrid(a,b);
AB = [A(:) B(:)];
Z = zeros(size(A(:)));
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

%{
%draw contours of Gaussians
n = size(X_GDA,2);
G0 = zeros(size(A(:)));
G1 = zeros(size(A(:)));
for i=1:size(A(:))
    G0(i) = Gauss(n,AB(i,:),mu0,cov0);
    G1(i) = Gauss(n,AB(i,:),mu1,cov1);
end
G0 = reshape(G0, size(A));
G1 = reshape(G1, size(A));
contour(A,B,G0,[0:0.005:1]);
contour(A,B,G1,[0:0.001:0.5]);
%}
%{
%second attempt at drawing hyperplane
L = zeros(size(A(:)));
for i=1:size(A(:))
    Pxy0 = (1/(2*pi*sqrt(det(cov0))))*exp(-0.5*(AB(i,:)*inv(cov0)*AB(i,:)'));
    Pxy1 = (1/(2*pi*sqrt(det(cov1))))*exp(-0.5*(AB(i,:)*inv(cov1)*AB(i,:)'));
    Px = Pxy0*(1-phy) + Pxy1*phy;
    Py0x = ((1-phy)*Pxy0)/Px;
    Py1x = (phy*Pxy1)/Px;
    L(i) = log(Py0x/Py1x);
end
L = reshape(L, size(A));
contour(A,B,Z,[-0.001:0.0001:0.001]);
%}

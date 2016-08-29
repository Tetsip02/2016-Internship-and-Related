X = load('X_MoG.out');
mu = load('mu_MoG.out');
cov_raw = load('cov_MoG.out');
nCl = 2;
n = size(X,2);
%restructure cov back into cube
cov = zeros(n,n,nCl);
for i=1:nCl
    lowbound = (i-1)*n + 1;
    upbound = i*n;
    cov(:,:,i) = cov_raw(lowbound:upbound,1:n);
end

%plotting will only work if data is two dimensional
%plot means
figure;
plot(X(:,1),X(:,2),'bx');
hold on;
for j=1:nCl
  plot(mu(j,1), mu(j,2),'or','MarkerSize',15,'MarkerFaceColor','r');
end
%plot contours:
a=linspace(min(X(:,1)),max(X(:,1)));
b=linspace(min(X(:,2)),max(X(:,2)));
[A B] = meshgrid(a,b);
AB = [A(:) B(:)];
Z = zeros(size(A(:)));
for j = 1:nCl
  for i=1:size(A(:))
    Z(i) = Z(i) + Gauss(n,AB(i,:),mu(j,:),cov(:,:,j));
  end
end
Z = reshape(Z, size(A));
contour(A,B,Z);
figure(2);
surf(A,B,Z);

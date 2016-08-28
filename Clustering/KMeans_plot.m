X = load('X_kMeans.out');
c = load('c_kMeans.out');
mu = load('mu_kMeans.out');
figure;
hold on;
plot(mu(:,1),mu(:,2),'rx','markersize',8,'linewidth',5);
plot(mu(:,1),mu(:,2),'gx','markersize',8,'linewidth',5);
for i=1:length(c)
  if c(i)==1
    plot(X(i,1),X(i,2),'bo');
  elseif c(i)==2
    plot(X(i,1),X(i,2),'mo');
  else
    plot(X(i,1),X(i,2),'yo');
  end
end
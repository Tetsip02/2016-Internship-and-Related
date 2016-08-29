function y = Gauss(n,x,mu,sigma)
  y = exp(-0.5*(x-mu)*inv(sigma)*(x-mu)')/sqrt(((2*pi)^n)*det(sigma));
end
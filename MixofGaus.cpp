//Mixture of Gaussians
#include <iostream>
#include <fstream>
#include <armadillo>
#include <cmath>  //pow()


double PI = 3.14159265359;

double Gauss(int n, arma::mat x, arma::mat mu, arma::mat sigma) {
  arma::mat gmat = exp(-0.5*(x-mu)*inv(sigma)*trans(x-mu))/sqrt(std::pow(2*PI,n)*det(sigma));
  double g = gmat(0,0);  //need to convert the one-by-one 'matrix' into a scalar
  return g;
}

int main() {

arma::mat rawdata;
rawdata.load("irisDataRevised");
arma::mat X;
X=rawdata(arma::span(0,99), arma::span(0,1));
int m = X.n_rows;
int n = X.n_cols;
int nCl = 2;
int numIter = 25;
arma::mat w(m,nCl);
w.randu();
arma::mat phy(nCl,1);
phy.ones();
phy = phy/nCl;
arma::mat mu(nCl,n);
mu.randu();
arma::cube cov(n,n,nCl);
cov.zeros();
for (int j=0;j<nCl;j++) {
  cov.slice(j).eye();
}

double denom;
//EM
for (int k=0;k<numIter;k++) {
  //E-step
  for (int i=0;i<m;i++) {
    for (int j=0;j<nCl;j++) {
      denom = 0;
      for (int l=0;l<nCl;l++) {
        denom += Gauss(n, X.row(i), mu.row(l), cov.slice(l))*phy(l,0);
      }
    w(i,j) = Gauss(n, X.row(i), mu.row(j), cov.slice(j))*phy(j,0)/denom;
    }
  }
  //M-Step
  mu.zeros();
  cov.zeros();
  for (int j=0;j<nCl;j++) {
    phy(j) = sum(w.col(j))/m;
    for (int i=0;i<m;i++) {
      mu.row(j) += w(i,j)*X.row(i);
    }
    mu.row(j) = mu.row(j)/sum(w.col(j));
    for (int i=0;i<m;i++) {
      cov.slice(j) += w(i,j)*(trans(X.row(i)-mu.row(j))*(X.row(i)-mu.row(j)));
    }
    cov.slice(j) = cov.slice(j)/sum(w.col(j));
  }
}

std::cout << mu << std::endl;
std::cout << phy << std::endl;
std::cout << w << std::endl;

}



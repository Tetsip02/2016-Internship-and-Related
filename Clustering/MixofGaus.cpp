//Mixture of Gaussians
#include <iostream>
#include <fstream>
#include <armadillo>
#include <cmath>  //pow()

#include "MixofGausSettings.cpp"

double Gauss(int n, arma::mat x, arma::mat mu, arma::mat sigma) {
  arma::mat gmat = exp(-0.5*(x-mu)*inv(sigma)*trans(x-mu))/sqrt(std::pow(2*PI,n)*det(sigma));
  double g = gmat(0,0);  //need to convert the one-by-one 'matrix' into a scalar
  return g;
}

int main() {

arma::mat rawdata;
rawdata.load(data.c_str());
arma::mat X;
X=rawdata(arma::span(0,numExp-1), arma::span(0,numFeat-1));
int m = numExp;
int n = numFeat;
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

  //output data files to be used for plotting
  std::ofstream of_X("X_MoG.out");
  std::ofstream of_mu("mu_MoG.out");
  std::ofstream of_cov("cov_MoG.out");
  for (int i=0;i<X.n_rows;i++) {
    of_X << X.row(i) << std::endl;
  }
  for (int i=0;i<mu.n_rows;i++) {
    of_mu << mu.row(i) << std::endl;
  }
  for (int i=0;i<cov.n_slices;i++) {
    of_cov << cov.slice(i);
  }


}

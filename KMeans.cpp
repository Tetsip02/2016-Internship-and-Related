//K-Means
#include <iostream>
#include <fstream>
#include <armadillo>

int main() {
arma::mat rawdata;
rawdata.load("irisDataRevised");
arma::mat X;
X=rawdata(arma::span(0,149), arma::span(0,1));
int m = X.n_rows;
int n = X.n_cols;
int nCl = 3;
int numIter = 100;
arma::mat mu(nCl,n);
mu.randu();

//bring inital mu into correct range
for (int i=0; i<n; i++) {
  mu.col(i) = X.col(i).min() + (X.col(i).max()-X.col(i).min())*mu.col(i);
}

arma::mat cinit(nCl,1);
cinit.zeros();
arma::mat cloop;
arma::mat c(m,1);  //vector which assign each example to a cluster
int numC;
for (int k=0; k<numIter; k++) {
  //E-Step:
  c.zeros();
  for (int i=0; i<m; i++) {
    cloop = cinit;
    for (int j=0; j<nCl; j++) {
      cloop.row(j) = (X.row(i)-mu.row(j))*trans(X.row(i)-mu.row(j));
    }
    c.row(i) = cloop.index_min();
  }
  //M-Step:
  mu.zeros();
  for (int j=0; j<nCl; j++) {
    numC=0.00001;
    for (int i=0; i<m; i++) {
      if (c(i,0)==j) {
        numC += 1;
        mu.row(j) += X.row(i);
      }
    }
    mu.row(j) = mu.row(j)/numC;
  }
}
std::cout << mu << std::endl;

}

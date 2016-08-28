//K-Means
#include <iostream>
#include <fstream>
#include <armadillo>

#include "KMeansSettings.cpp"

int main() {
arma::mat rawdata;
rawdata.load(data.c_str());
arma::mat X;
X=rawdata(arma::span(0, numExp-1), arma::span(0, numFeat-1));
int m = numExp;
int n = numFeat;
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
    numC=0;
    for (int i=0; i<m; i++) {
      if (c(i,0)==j) {
        numC += 1;
        mu.row(j) += X.row(i);
      }
    }
    if(numC==0) {mu.row(j).zeros();} //no training examples were assigned to this cluster (maybe too many clusters?)
    mu.row(j) = mu.row(j)/numC;
  }
}

  //output data files to be used for plotting
  std::ofstream of_X("X_kMeans.out");
  std::ofstream of_c("c_kMeans.out");
  std::ofstream of_mu("mu_kMeans.out");
  for (int i=0;i<X.n_rows;i++) {
    of_X << X.row(i) << std::endl;
    of_c << c.row(i) << std::endl;
  }
  for (int i=0;i<mu.n_rows;i++) {
    of_mu << mu.row(i) << std::endl;
  }

}

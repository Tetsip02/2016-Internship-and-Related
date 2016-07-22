#include <iostream>
#include <fstream>
#include "vectorclass.cpp"
#include <armadillo>
#define ARMA_64BIT_WORD  //enable use of 64 bit integers (necessary to compile large matrices
#include <cmath>  //exp()

arma::mat sigmoid(arma::mat z) {
  z=1/(1+exp(-z));
  return z;
}

int main() {
  arma::mat X_data;
  X_data.load("q1x.dat");
  arma::mat icept(X_data.n_rows,1);
  icept.ones();
  arma::mat X=join_horiz(icept, X_data);
  arma::mat y;
  y.load("q1y.dat");
  int m=X.n_rows;
  arma::mat h(m,1);
  h.zeros();
  arma::mat theta(X.n_cols,1);
  theta.zeros();
  int numIter=50;
  arma::mat l(1,1);
  arma::mat H(X.n_cols,X.n_cols);
  arma::mat grad(X.n_cols,1);
  for (int i=0; i<numIter; ++i) {
    l.zeros();
    H.zeros();
    grad.zeros();
    for (int j=0; j<m; ++j) {
      h.row(j)=sigmoid(X.row(j)*theta);
      l += y.row(j)*log(h.row(j)) + (1-y.row(j))*log(1-h.row(j));
      arma::mat hh = h.row(j)*(1-h.row(j));
      double hhh=hh(0,0);
      H -= hhh*(trans(X.row(j))*X.row(j));
      grad += trans(X.row(j))*(y.row(j)-h.row(j));
    }
    theta -= inv(H)*grad;
  }

  std::cout << theta << std::endl;
}

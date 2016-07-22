#include <iostream>
#include <fstream>
#include <armadillo>

int main() {
  arma::mat y;
  y.load("q2y.dat");
  arma::mat X_data;
  X_data.load("q2x.dat");
  arma::mat icept(X_data.n_rows,1);
  icept.ones();
  arma::mat X=join_horiz(icept, X_data);
  arma::mat theta(X.n_cols,1);
  theta.zeros();
  theta=(pinv((X.t()*X)))*(X.t())*y;
  arma::mat h=X*theta;
}

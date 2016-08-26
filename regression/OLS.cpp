//Ordinary Least Squares
/*Description
y is approximated as a linear function of X.
y ~ h = theta^T * X
Theta is chosen by minimizing the squared error.
There are several methods to do this, I have included two: batch gradient descent and the normal equation
*/
#include <iostream>
#include <fstream>
#include <armadillo>
//documentation for armadillo functions (such as arma::span) can be found at http://arma.sourceforge.net/docs.html

#include "OLSsettings.cpp"

/*Normal equation
Disadvantage: need to compute the inverse X^T*X which computationally very expensive if n is large
Advantage: no need to choose a learning rate
*/
arma::mat train_normal (arma::mat X, arma::mat y);

/*Batch Gradient descent
Disadvantage:need to choosing a learning rate alpha that is sufficiently small to make sure the algorithm converges with each iteration, but large enough to save the number of iterations needed. With this data, alpha=0.01 and numIter=1500 works well.
Advantage: works well even when n is large
*/
arma::mat train_batchGradDescent(arma::mat X, arma::mat y, int numIter, double alpha);

int main() {
  //initiallization

  arma::mat data;
  data.load(dat.c_str());
  arma::mat X_data;
  arma::mat y;
  y = data(arma::span(0,199),13);
  //X_data = data(arma::span(0,199),arma::span(0,12));
  X_data = data(arma::span(0,199),5);
  //add X_0=1 to X
  arma::mat icept(X_data.n_rows,1);
  icept.ones();
  arma::mat X=join_horiz(icept, X_data);
  //train parameters using the normal equation or batch gradient descent
  arma::mat theta = train_batchGradDescent(X,y,1500,0.01);
  //fit the data and output predictions (ignore for now)
  if (newDat) {  //if newDat == "True"
    arma::mat newX;
    newX.load(newData.c_str());
    arma::mat h = newX*theta;
    std::ofstream ofs_h("linRegHypothesis.out");
    for (int i=0;i<newX.n_rows;i++) {
      ofs_h << h.row(i) << std::endl;
    }
  }
  //the below section outputs files which you can use to visualize your results in Ocatve/Matlab/Gnuplot
  std::ofstream of_theta("theta_OLS.out");
  std::ofstream of_y("y.out");
  std::ofstream of_X("X.out");
  for (int i=0;i<theta.n_rows;i++) {
    of_theta << theta.row(i) <<std::endl;
  }
  for (int i=0;i<y.n_rows;i++) {
    of_y << y.row(i) << std::endl;
    of_X << X_data.row(i) << std::endl;
  }
}

arma::mat train_normal (arma::mat X, arma::mat y) {
  arma::mat theta(X.n_cols,1);
  theta.zeros();
  theta=(pinv((X.t()*X)))*(X.t())*y;
  return theta;
}

arma::mat train_batchGradDescent(arma::mat X, arma::mat y, int numIter, double alpha) {
  arma::mat theta(X.n_cols,1);
  int m = X.n_rows;
  for (int i=0; i<numIter; i++) {
    theta -= (alpha/m) * (trans(X)*(X*theta - y));
  }
  return theta;
}

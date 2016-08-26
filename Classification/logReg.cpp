#include <iostream>
#include <fstream>
#include <armadillo>
#include <cmath>  //exp()

#include "logRegSettings.cpp"

arma::mat sigmoid(arma::mat z);

//logistic regression which takes training data as inputs and returns trained parameters as output. Third argument, number of iterations of logistic regression is optional. If none if defined, 50 iterations will be executed as a default
arma::mat log_Regression(arma::mat X, arma::mat y, int numIter = defaultNumIter);

int main() {
  //initialization 
  arma::mat data;
  data.load(dat.c_str());
  arma::mat X_data;
  arma::mat y;
  y = data(arma::span(0,99),4);
  X_data = data(arma::span(0,99),arma::span(0,3));
  //add intercept:
  arma::mat icept(X_data.n_rows,1);
  icept.ones();
  arma::mat X = join_horiz(icept, X_data);
  //training parameters
  arma::mat theta = log_Regression(X,y);
  
  std::cout << theta << std::endl;
  
  //fit the new data and output predictions
  if (newDat) {  //if newDat == "True"
    arma::mat newX;
    newX.load(newData.c_str());
    arma::mat h = newX*theta;
    std::ofstream ofs_h("logRegHypothesis.out");
    for (int i=0;i<newX.n_rows;i++) {
      ofs_h << h.row(i) << std::endl;
    }
  }
}


arma::mat sigmoid(arma::mat z) {
  z=1/(1+exp(-z));
  return z;
}

arma::mat log_Regression(arma::mat X, arma::mat y, int numIter) {
  int m=X.n_rows;
  int n=X.n_cols;
  arma::mat theta(n,1);
  theta.zeros();
  //declare log-likleyhood, gradient and Hessian; Log likelihood is returned for plotting and debugging purposes
  //arma::mat logLike(numIter,1);
  arma::mat grad(n,1);
  arma::mat H(n,n);
  
  arma::mat hi;  //approximation of y(i)
  
  /*Newton raphson: teta = theta - inv(H)*grad;
  where H is the Hessian and grad is the gradient.
  Typically, Newton-Raphson converges after much fewer iterations than batch gradient descent, but each iteration is more costly as it involves inverting an n-by-n Hessian. 
  Suitable so long as n is not too large.
  */
  for (int k=0; k<numIter; k++) {
    //logLike.row(k).zeros();
    H.zeros();
    grad.zeros();
    for (int i=0; i<m; i++) { 
      hi=sigmoid(X.row(i)*theta);
      //logLike.row(k) += y.row(i)*log(hi) + (1-y.row(i))*log(1-hi);
      grad += trans(X.row(i))*(y.row(i)-hi);
      double dummy_hi = hi(0,0);
      H -= (dummy_hi*(1-dummy_hi))*(trans(X.row(i))*X.row(i));
    }
    theta -= inv(H)*grad;
  }
  return theta;
}

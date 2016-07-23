#include <iostream>
#include <fstream>
//#include "vectorclass.cpp"
#include <armadillo>
//#define ARMA_64BIT_WORD  //enable use of 64 bit integers (necessary to compile large matrices
#include <cmath>  //exp()

#include "logRegSettings.cpp"

arma::mat sigmoid(arma::mat z);

//logistic regression which takes training data as inputs and returns trained parameters as output. Thrid argument, number of iterations of logistic regression is optional. If none if defined, 50 iterations will be executed as a default
arma::mat log_Regression(arma::mat X, arma::mat y, int numIter = 50) {
  int m=X.n_rows;
  int n=X.n_cols;
  arma::mat theta(n,1);
  theta.zeros();
  //declare log-likleyhood, gradient and Hessian; Log likelihood is returned for plotting and debugging purposes
  arma::mat logLike; //although this is a scalar we need to treat it as a 1-by-1 'matrix to avoid an error
  arma::mat grad(n,1);
  arma::mat H(n,n);
  
  arma::mat hi;  //approximation of y(i)
  
  /*Newton raphson: teta = theta - inv(H)*grad;
  where H is the Hessian and grad is the gradient.
  Typically, Newton-Raphson converges after much fewer iterations than batch gradient descent, but each iteration is more costly as it involves inverting an n-by-n Hessian. 
  Suitable so long as n is not too large.
  */
  for (int k=0; k<numIter; k++) {
    logLike.zeros();
    H.zeros();
    grad.zeros();
    for (int i=0; i<m; i++) {
      std::cout << "X.rows " << X.n_rows
            << "\nX.cols " << X.n_cols
            << "\ntheta.rows " << theta.n_rows
            << "\ntheta.cols " << theta.n_cols << std::endl;
 
      hi=sigmoid(X.row(i)*theta);
   
      //logLike += y.row(i)*log(hi) + (1-y.row(i))*log(1-hi);
      grad += trans(X.row(i))*(y.row(i)-hi);
      double dummy_hi = hi(0,0);
      H -= (dummy_hi*(1-dummy_hi))*(trans(X.row(i))*X.row(i));
    }
    return theta -= inv(H)*grad;
  }
}

int main() {
  //initialization 
  arma::mat X_data;
  X_data.load(X_dat.c_str());
  arma::mat icept(X_data.n_rows,1);
  icept.ones();
  arma::mat X=join_horiz(icept, X_data);
  arma::mat y;
  y.load(y_dat.c_str());
  //training parameters
  arma::mat theta = log_Regression(X,y);
  
  std::cout << theta << std::endl;
}


arma::mat sigmoid(arma::mat z) {
  z=1/(1+exp(-z));
  return z;
}

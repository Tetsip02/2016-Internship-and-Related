//Ordinary Least Squares
/*Description
y is approximated as a linear function of X.
y ~ h = theta^T * X
Theta is chosen by minimizing the squared error.
There are several methods to do this, I have included two: batch gradient descent and the normal equation
*/
#include <iostream>
#include <fstream>
#include <armadillo> //documenatation available at http://arma.sourceforge.net/docs.html
#include <iomanip> //setw()

/*Normal equation
Disadvantage: need to compute the inverse X^T*X which computationally very expensive if n is large
Advantage: no need to choose a learning rate
*/
arma::mat train_normal (arma::mat X, arma::mat y);

/*Batch Gradient descent
Disadvantage:need to choosing a learning rate alpha that is sufficiently small to make sure the algorithm converges with each iteration, but large enough to save the number of iterations needed. This varies depending on degree.
Advantage: works well even when n is large
*/
arma::mat train_batchGradDescent(arma::mat X, arma::mat y, int numIter, double alpha);


int main() {

  /*Generating data*/
  /*
  x is a vector of length 30 with elements which are uniformly distributed between 0 and 6
  y = sin(x) + noise, where noise is a random number taken from a normal distribution with mean 0,variance 1
  */
  arma::mat x = arma::randu(30,1);
  x = 6*x;
  double s = 0.1; //standard deviation
  arma::mat noise = arma::randn(30,1);
  noise = s*noise;
  arma::mat y = arma::sin(x) + noise;

  /*get polynomial features and store new design matrix in X*/
  int degree = 5;
  arma::mat X;
  arma::mat temp;
  for (int i=0;i<=degree;i++) {
    temp = arma::pow(x,i);
    X = arma::join_horiz(X,temp);
  }

  //train parameters using the normal equation or batch gradient descent
  //arma::mat theta = train_batchGradDescent(X,y,1500,0.005);
  arma::mat theta = train_normal(X,y);

  /*Gnuplot*/
  std::ofstream of_dat("OLSsin.dat");
  for(int i=0;i<x.n_rows;i++) {
    of_dat << std::setw(12) << x(i,0) << " " << std::setw(12) << y(i,0) << std::endl;
  }

  std::ofstream of_OLSsin("OLSsin.gnu");
  of_OLSsin << "reset" << std::endl;
  of_OLSsin << "set terminal png" << std::endl;
  of_OLSsin << "set output 'OLSsin.png'" << std::endl;
  of_OLSsin << std::endl;
  for (int i=0;i<theta.n_rows;i++) {
    of_OLSsin << "theta" << i << " = " << theta(i,0) << std::endl;
  }
  of_OLSsin << std::endl;
  of_OLSsin << "set xlabel 'x'" << std::endl;
  of_OLSsin << "set ylabel 'y(x)'" << std::endl;
  of_OLSsin << std::endl;
  of_OLSsin << "y(x) = ";
  for (int i=0;i<theta.n_rows;i++) {
    of_OLSsin << "theta" << i << "*" << "x**" << i << " + ";
  }
  of_OLSsin << "0" << std::endl;
  of_OLSsin << "plot 'OLSsin.dat' using 1:2 notitle, [0:6] y(x) title 'model'" << std::endl;

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

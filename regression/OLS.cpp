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
#include <iomanip>

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


// /*construct design matrix, maps features X1 and X2 into all polynomial terms up to degree power*/
// arma::mat getDesignMat(arma::mat feat1,arma::mat feat2, int degree) {
//   arma::mat designmat;
//   arma::mat temp;
//   for(int i=0;i<degree;i++) {
//     for (int j=0;j<i;j++) {
//       temp = (feat1.^(i-j)).*(feat2.^j);
//       designmat = arma::join::horix(designmat,temp);
//     }
//   }
//   return designmat;
// }

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

  //add X_0=1 to X
  arma::mat icept(x.n_rows,1);
  icept.ones();
  //arma::mat X=join_horiz(icept, x);
  arma::mat X = icept;
  //add polynomial features
  int degree = 2;
  arma::mat temp;
  for (int i=1;i<=degree;i++) {
    temp = arma::pow(x,i);
    X = arma::join_horiz(X,temp);
  }
  std::cout << X;


  //train parameters using the normal equation or batch gradient descent
  arma::mat theta = train_batchGradDescent(X,y,1500,0.01);


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
  of_OLSsin << "theta1 = " << theta(0,0) << std::endl;
  of_OLSsin << "theta2 = " << theta(1,0) << std::endl;
  of_OLSsin << std::endl;
  of_OLSsin << "set xlabel 'x'" << std::endl;
  of_OLSsin << "set ylabel 'y(x)'" << std::endl;
  of_OLSsin << std::endl;
  of_OLSsin << "y(x) = theta2*x + theta1" << std::endl;
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

//Linear/Quadratic Discriminant Analysis for two classes
#include <iostream> // cin,cout
#include <fstream> //ifstream,ofstream
#include <armadillo>
#include <cmath>  // exp()

#include "GDASettings.cpp"

/*Gaussian probability density function*/
double Gauss(int n, arma::mat x, arma::mat mu, arma::mat sigma) {
  arma::mat gmat = exp(-0.5*(x-mu)*inv(sigma)*trans(x-mu))/sqrt(std::pow(2*PI,n)*det(sigma));
  double g = gmat(0,0);  //need to convert the one-by-one 'matrix' into a scalar
  return g;
}


int main() {
  /*initialization*/
  //Don't modify lines 22 to 26. To change the data set, number of
  //training examples or features, go to GDASettings.cpp and change the
  //variables data (dataset), numExp (training examples) or numFeat (features)
  //Notice thst we start counting from zero.
  arma::mat rawdata;
  rawdata.load(data.c_str()); //load data (that is FisherIris.dat) into rawdata.
  arma::mat X;
  X=rawdata(arma::span(0,numExp-1), arma::span(0,numFeat-1));

  /*Read the fifth column of FisherIris.dat (ie. the labels) into y*/
  //numZero and numOne are counters that keep track of the number of
  //examples from each class
  arma::mat y(numExp,1);
  double numZero=0;
  double numOne=0;
  for (int i=0;i<numExp;i++) {
    y(i,0) = rawdata(i,rawdata.n_cols-1);
    if (rawdata(i,rawdata.n_cols-1)==0) {++numZero;}
    else {++numOne;}
  }

  /*fit parameters phy, mu0, mu1*/
  double phy=numOne/numExp;
  arma::mat mu0(1,numFeat);
  mu0.zeros();
  arma::mat mu1(1,numFeat);
  mu1.zeros();
  for (int i=0; i<numExp; i++) {
    if (y(i,0)==0.0) {mu0 += X.row(i);}
    else {mu1 += X.row(i);}
  }
  mu0 = mu0/numZero;
  mu1 = mu1/numOne;

  /*fit covariance*/
  arma::mat cov0(X.n_cols,X.n_cols);
  arma::mat cov1(X.n_cols,X.n_cols);
  cov0.zeros();
  cov1.zeros();
  for (int i=0; i<numExp; i++) {
    if (y(i,0)==0) {
      cov0 += trans(X.row(i)-mu0)*(X.row(i)-mu0);
    }
    else {
      cov1 += trans(X.row(i)-mu1)*(X.row(i)-mu1);
    }
  }
  cov0 = cov0/numExp;
  cov1 = cov1/numExp;

  /*Read X,y and the parameters into files to be used for plotting*/
  std::ofstream of_X("X_GDA.out");
  std::ofstream of_y("y_GDA.out");
  std::ofstream of_phy("phy.out");
  std::ofstream of_mu0("mu0.out");
  std::ofstream of_mu1("mu1.out");
  std::ofstream of_cov0("cov0.out");
  std::ofstream of_cov1("cov1.out");
  for (int i=0;i<X.n_rows;i++) {
    of_X << X.row(i) << std::endl;
    of_y << y.row(i) << std::endl;
  }
  of_phy << phy;
  of_mu0 << mu0.row(0);
  of_mu1 << mu1.row(0);
  for (int i=0;i<cov0.n_rows;i++) {
    of_cov0 << cov0.row(i) << std::endl;
    of_cov1 << cov1.row(i) << std::endl;
  }

//You can now move on to the file LDA_plot.m to plot the results

  /*use parameters to predict new data*/
  /*
  Need to calculate P(y=1|X). That is the probabilty that a data point
  beongs to class 1 given the data X. The threshold is set at 0.5.
  That is when the data point is equally likely to belong to either
  class.
  Using Bayes rule we can write P(y=1|x) = (P(y=1)*P(x|y=1))/P(x).
  Using the law of total probability we can expand P(x):
  P(x)=P(x|y=0)*P(y=0) + P(x|y=1)*P(y=1).
  With the parameters we can calculate all that needed to get P(y=1|x):
  P(y=1)=phy (and P(y=0)=1-phy).
  P(X|y=1) and P(X|y=0) are modeled using a multivariate Gaussian
  distribution with means mu1 and mu0, and covariance cov0
  and cov1, respectively (or just cov where cov=cov0=cov1 in the
  LDA case).
  Note, PXy0=P(X|y=0) and PXy1=P(X|y=1).
  */
  if (newDat) {  //if newDat == "True"
    arma::mat newX;
    newX.load(newData.c_str());
    double PXy0;
    double PXy1;
    arma::mat h(newX.n_rows,1);
    if (LDA) {
      for (int i=0;i<newX.n_rows;i++) {
        PXy0 = Gauss(newX.n_cols, newX.row(i), mu0, cov0);
        PXy1 = Gauss(newX.n_cols, newX.row(i), mu1, cov0);
        h(i,0) = (PXy1*phy)/(PXy0*(1-phy)+PXy1*phy);
        if (h(i,0) > threshold) {h(i,0) = 1;}
      }
    }
    if (!LDA) {
      for (int i=0;i<newX.n_rows;i++) {
        PXy0 = Gauss(newX.n_cols, newX.row(i), mu0, cov0);
        PXy1 = Gauss(newX.n_cols, newX.row(i), mu1, cov1);
        h(i,0) = (PXy1*phy)/(PXy0*(1-phy)+PXy1*phy);
        if (h(i,0) > threshold) {h(i,0) = 1;}
      }
    }

    std::ofstream ofs_h("GDAHypothesis.out");
    for (int i=0;i<newX.n_rows;i++) {
      ofs_h << h.row(i) << std::endl;
    }
  }
}

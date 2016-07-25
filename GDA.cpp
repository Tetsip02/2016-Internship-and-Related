//Linear/Quadratic Discriminant Analysis for two classes
#include <iostream>
#include <fstream>
#include <armadillo>
#include <cmath>  // exp()

#include "GDASettings.cpp"


double Gauss(int n, arma::mat x, arma::mat mu, arma::mat sigma) {
  arma::mat gmat = exp(-0.5*(x-mu)*inv(sigma)*trans(x-mu))/sqrt(std::pow(2*PI,n)*det(sigma));
  double g = gmat(0,0);  //need to convert the one-by-one 'matrix' into a scalar
  return g;
}


int main() {
  //initialization
  arma::mat rawdata;
  rawdata.load(data.c_str());
  int n = rawdata.n_cols;
  arma::mat X;
  X=rawdata(arma::span(0,numExp-1), arma::span(0,numFeat-1));
  
  arma::mat y(numExp,1);
  double numZero=0;
  double numOne=0;
  for (int i=0;i<numExp;i++) {
    y(i,0) = rawdata(i,n-1);
    if (rawdata(i,n-1)==0) {++numZero;}
    else {++numOne;}
  }
  
  //fit parameters phy, mu0, mu1
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
  
  arma::mat cov(X.n_cols,X.n_cols);
  arma::mat cov0(X.n_cols,X.n_cols);
  arma::mat cov1(X.n_cols,X.n_cols);
  //fit covariance
  if (LDA) {
    cov.zeros();
    for (int i=0; i<numExp; i++) {
      if (y(i,0)==0) {
        cov += trans(X.row(i)-mu0)*(X.row(i)-mu0);
      }
      else {
        cov += trans(X.row(i)-mu1)*(X.row(i)-mu1);
      }
    }
  }
  
  if (!LDA) {
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
  }
  
  //fit the new data and output predictions
  if (newDat) {  //if newDat == "True"
    arma::mat newX;
    newX.load(newData.c_str());
    double PXy0;  //probability of X given y=0
    double PXy1;  //probability of X given y=1
    arma::mat h(newX.n_rows,1);  //prediction, probability of y=1 given X
    if (LDA) {
      for (int i=0;i<newX.n_rows;i++) {
        PXy0 = Gauss(newX.n_cols, newX.row(i), mu0, cov);
        PXy1 = Gauss(newX.n_cols, newX.row(i), mu1, cov);
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


    //decision boundary for LDA (y=ax+b). For a pair of classes this can be obtained in the following way:it must pass through the midpoint of their respective means, ie 1/2(mu0+mu1), and be perpendiclar to SIGMA^-1*(mu0-mu1)
  if (LDA) {
    arma::mat midpoint(mu0.n_rows, mu0.n_cols);
    midpoint=0.5*(mu0+mu1);
    arma::mat decboundgrad(1,2);
    //at this point, need to specify which two features to plot:
    arma::mat mu0plot;
    mu0plot << mu0(0,plotFeat1) << mu0(0,plotFeat2) << arma::endr;
    arma::mat mu1plot;
    mu1plot << mu1(0,plotFeat1) << mu1(0,plotFeat2) << arma::endr;
    arma::mat covplot;
    covplot << cov(plotFeat1,plotFeat1) << cov(plotFeat1,plotFeat2) << arma::endr << cov(plotFeat2,plotFeat1) << cov(plotFeat2,plotFeat2) << arma::endr;
    decboundgrad = (mu0plot-mu1plot)*inv(covplot);
    double a=-decboundgrad(0,1)/decboundgrad(0,0);
    double b=midpoint(0,plotFeat2)-midpoint(0,plotFeat1)*a;
    std::cout << "a=" << a << "  " << "b=" << b << std::endl;
  }
  
}

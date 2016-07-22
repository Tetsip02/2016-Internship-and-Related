//QDA for two classes
#include <iostream>
#include <fstream>
#include <armadillo>
#include <cmath>  // exp()

#include "LDAtwoClassesSettingsFile.cpp"



int main() {
arma::mat rawdata;
rawdata.load(data.c_str());
int n = rawdata.n_cols; //n is needed to specify y
//extract sepal length and sepal width for setosa and vergicolor
arma::mat X;
X=rawdata(arma::span(0,numExp-1), arma::span(0,numFeat-1));
//create label vector and count occurences of each class
arma::mat y(numExp,1);
double numZero=0;  //setosa
double numOne=0;  //vergicolor
for (int i=0;i<numExp;i++) {
  y(i,0) = rawdata(i,n-1);
  if (rawdata(i,n-1)==0) {++numZero;}
  else {++numOne;}
}

//fit parameters phy, mu0, mu1, cov0, cov1
double phy=numOne/numExp;
arma::mat mu0(1,numFeat);
arma::mat mu1(1,numFeat);
for (int i=0; i<numExp; i++) {
  if (y(i,0)==0) {mu0 += X.row(i);}
  else {mu1 += X.row(i);}
}
mu0 = mu0/numZero;
mu1 = mu1/numOne;
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
//prediction for new data
const double PI=3.14159265358979323;
double threshold=0.5;

arma::mat newdata;
newdata << 5.4710 << 3.0940 << 2.8620 << 0.7850 << arma::endr; //mid-point between means, this should give oredict = 0.5 if everything is correct5.4710,3.0940

double pre0 = sqrt(det(cov0))*sqrt(2*PI);
double pre1 = sqrt(det(cov1))*sqrt(2*PI);
arma::mat Pxy0 = exp(-0.5*(newdata-mu0)*inv(cov0)*trans(newdata-mu0))/pre0;
double PXY0 = Pxy0(0,0);
arma::mat Pxy1 = exp(-0.5*(newdata-mu1)*inv(cov1)*trans(newdata-mu1))/pre1;
double PXY1 = Pxy1(0,0);
double predict = (PXY1*phy)/(PXY0*(1-phy)+PXY1*phy);
std::cout << predict << std::endl;

if (predict>threshold) {
  std::cout << "The flower belongs to the Setosa species" << std::endl;
}
else {
  std::cout << "The flower belongs to the Versicolor species" << std::endl;
}




//plotting decision boundary: ax^2 + bx + c
}

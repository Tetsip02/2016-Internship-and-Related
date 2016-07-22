//LDA for two classes
#include <iostream>
#include <fstream>
#include <armadillo>
#include <cmath>  // exp()

#include "LDAtwoClassesSettingsFile.cpp"


int main() {
arma::mat rawdata;
rawdata.load(data.c_str());
int n = rawdata.n_cols; //n is needed to specify y
//extract sepal length, sepal width and petal length for setosa and vergicolor
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

//fit parameters phy, mu0, mu1, cov
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
cov.zeros();

for (int i=0; i<numExp; i++) {
  if (y(i,0)==0) {
    cov += trans(X.row(i)-mu0)*(X.row(i)-mu0);
  }
  else {
    cov += trans(X.row(i)-mu1)*(X.row(i)-mu1);
  }
}
//prediction for new data
const double PI=3.14159265358979323;
double threshold=0.5;


arma::mat newdata;
newdata << 5.4710 << 3.0940 << 2.8620 << 0.7850 << arma::endr; //mid-point between means, this should give oredict = 0.5 if everything is correct5.4710,3.0940

double pre = sqrt(det(cov))*sqrt(2*PI);
arma::mat Pxy0 = exp(-0.5*(newdata-mu0)*inv(cov)*trans(newdata-mu0))/pre;
double PXY0 = Pxy0(0,0);
arma::mat Pxy1 = exp(-0.5*(newdata-mu1)*inv(cov)*trans(newdata-mu1))/pre;
double PXY1 = Pxy1(0,0);
double predict = (PXY1*phy)/(PXY0*(1-phy)+PXY1*phy);
std::cout << predict << std::endl;

if (predict>threshold) {
  std::cout << "The flower belongs to the Setosa species" << std::endl;
}
else {
  std::cout << "The flower belongs to the Versicolor species" << std::endl;
}



//outsource data to be used for gnuplot:
std::ofstream ofs_X("LDA_X.out");
if(!ofs_X) {
  std::cout << "Problem with opening data file." << std::endl;
  return 1;
}
for (int i=0; i<X.n_rows ;i++) {
  for (int j=0; j<X.n_cols ;j++) {
    ofs_X << X(i,j) << " ";
  }
  ofs_X << std::endl;
}

//decision boundary (y=ax+b). For a pair of classes this can be obtained in the following way:it must pass through the midpoint of their respective means, ie 1/2(mu0+mu1), and be perpendiclar to SIGMA^-1*(mu0-mu1)
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


/* gnuplot commands:
set xlabel "Sepal length"
set ylabel "Sepal width"
set autoscale
set label "mid-point" at 5.4710, 3.0940
plot 1.22744*(x) -3.62134, "LDA_X.out" using 1:1 1:2
when X consists of two columns, gnuplot will treat the first one as the independent variable by default. we can reverse this with the command "using 2:1"
still need a way to differentiate between defferent labels
*/



}

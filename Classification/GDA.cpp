//Fisher Linear/Quadratic Discriminant Analysis for two classes
#include <iostream> // cin,cout
#include <fstream> //ifstream,ofstream
#include <armadillo> //documenatation available at http://arma.sourceforge.net/docs.html
//append -larmadillo to compile
#include <cmath>  // exp(), log()

//#include <boost/tuple/tuple.hpp>
//#include "../include/gnuplot-iostream/gnuplot-iostream.h"
//append -lboost_iostreams -lboost_system -lboost_filesystem to compile

#include "GDASettings.cpp"

/*Gaussian probability density function*/
double Gauss(int n, arma::mat x, arma::mat mu, arma::mat sigma) {
  arma::mat gmat = exp(-0.5*(x-mu)*inv(sigma)*trans(x-mu))/sqrt(std::pow(2*PI,n)*det(sigma));
  double g = gmat(0,0);  //need to convert the one-by-one 'matrix' into a scalar
  return g;
}

int main()
try {
  /*initialization*/
  //To change the training set, modify GDASettings.cpp.
  //If you don't want to use all 50 data points of each species
  //for training, adjust lines 27 to 29.
  //Note that we start counting from zero.
  arma::mat rawdata;
  rawdata.load(data.c_str()); //load data (that is FisherIris.dat) into rawdata.
  arma::mat X_dummy;
  arma::mat X;
  if(Setosa) {X_dummy = arma::join_vert(X_dummy,rawdata(arma::span(0,49),arma::span(0,4)));}
  if(Versicolor) {X_dummy = arma::join_vert(X_dummy,rawdata(arma::span(50,99),arma::span(0,4)));}
  if(Virginica) {X_dummy = arma::join_vert(X_dummy,rawdata(arma::span(100,149),arma::span(0,4)));}
  int numExp = X_dummy.n_rows; //number of training examples
  if(numExp != 100) {throw 1;}
  if(SepalLength) {X = arma::join_horiz(X,X_dummy.col(0));}
  if(SepalWidth) {X = arma::join_horiz(X,X_dummy.col(1));}
  if(PetalLength) {X = arma::join_horiz(X,X_dummy.col(2));}
  if(PetalWidth) {X = arma::join_horiz(X,X_dummy.col(3));}
  int numFeat = X.n_cols; //number of features used for training
  /*Create label vector y of zeros and ones from the fifth column of X_dummy*/
  //numZero and numOne are counters that keep track of the number of examples from each class
  arma::mat y(numExp,1);
  double numZero=0;
  double numOne=0;
  double check = X_dummy(0,4);
  for (int i=0;i<numExp;i++) {
    if(X_dummy(i,4)==check) {
      y(i,0) = 0;
      ++numZero;
    }
    else {
      y(i,0) = 1;
      ++numOne;
    }
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

  /*plotting files, ONLY for two features*/
  if (numFeat!=2) {throw 'a';}
  //data
  std::ofstream of_C1("Class0.out"); //training examples belonging to first class
  std::ofstream of_C2("Class1.out"); //training examples belonging to second class
  for (int i=0;i<y.n_rows;i++) {
    if (y(i,0)==0) {
      of_C1 << X.row(i) << std::endl;
    }
    else {
      of_C2 << X.row(i) << std::endl;
    }
  }

  //LDA projection line
  arma::mat mid; //midpoint between means
  mid = 0.5 * (mu0+mu1);
  arma::mat w; //direction of line of projection
  w = (mu0 - mu1) * inv(cov0);
  //bring line into form X2 = p * X1 + q:
  double p = -w(0,1)/w(0,0);
  double q = mid(0,1) - mid(0,0) * p;
  std::ofstream of_p("p.out");
  std::ofstream of_q("q.out");
  of_p << p;
  of_q << q;

  //copy gnuplot script into LDA_gnuplot.gnu:
  std::ofstream of_LDAgnu("LDA_plot.gnu");
  of_LDAgnu << "reset" << std::endl;
  of_LDAgnu << "set terminal png" << std::endl;
  of_LDAgnu << "set output 'LDA_plot.png'" << std::endl;
  of_LDAgnu << "#load first row first column of p.out into p" << std::endl;
  of_LDAgnu << "p = system(\"head -n1 p.out | awk '{print $1}'\")" << std::endl;
  of_LDAgnu << "q = system(\"head -n1 q.out | awk '{print $1}'\")" << std::endl;
  of_LDAgnu << "y(x) = p * x + q" << std::endl;
  of_LDAgnu << std::endl;
  of_LDAgnu << "set xlabel 'Feature 1'" << std::endl;
  of_LDAgnu << "set ylabel 'Feature 2'" << std::endl;
  of_LDAgnu << std::endl;
  of_LDAgnu << "plot y(x) title 'decision boundary', 'Class0.out' title 'Species 1', 'Class1.out' title 'Species 2'" << std::endl;

  //QDA
  //Decision boundary is a parabola in the form X^T*A*X + X^T*B + C = Z, at Z=0
  arma::mat A = -0.5*(inv(cov0)-inv(cov1));
  arma::mat B = (mu0*inv(cov0)) - (mu1*inv(cov1));
  arma::mat C = log((1-phy)/phy) - 0.5*log(det(cov0)/det(cov1))
              - 0.5*((mu0*inv(cov0)*trans(mu0))-(mu1*inv(cov1)*trans(mu1)));

  /*files for QDA_plot1.gnu*/
  //Solve X^T*A*X + X^T*B + C = 0 using the quadratic formula
  std::ofstream of_A("A.out");
  std::ofstream of_B("B.out");
  std::ofstream of_C("C.out");
  for (int i=0;i<A.n_rows;i++) {
    of_A << A.row(i);
  }
  of_B << B;
  of_C << C;

  std::ofstream of_QDA1gnu("QDA_plot1.gnu");
  of_QDA1gnu << "reset" << std::endl;
  of_QDA1gnu << "#read in A,B and C" << std::endl;
  of_QDA1gnu << "plot 'A.out' every ::0::0 using (A11 = $1, A12 = $2), 'A.out' every ::1::1 using (A21 = $1, A22 = $2), 'B.out' using (b1 = $1, b2 = $2), 'C.out' using (c = $1)" << std::endl;
  of_QDA1gnu << std::endl;
  of_QDA1gnu << "set terminal png" << std::endl;
  of_QDA1gnu << "set output 'QDA_plot.png'" << std::endl;
  of_QDA1gnu << std::endl;
  of_QDA1gnu << "set xlabel 'Feature 1'" << std::endl;
  of_QDA1gnu << "set ylabel 'Feature 2'" << std::endl;
  of_QDA1gnu << std::endl;
  of_QDA1gnu << "plot (-(((A12+A21) * x) + b2) + sqrt(((A12+A21) + b2)**2 - 4 * A22 * (A11*(x**2) + b1*x + c)))/(2*A22) title 'boundary1', (-((A12+A21)*x+b2) - sqrt(((A12+A21)+b2)**2-4*A22*(A11*x**2 + b1*x +c)))/(2*A22) title 'boundary2', 'Class0.out' title 'Species 1', 'Class1.out' title 'Species 2'" << std::endl;
  //when you run >'gnuplot QDA_plot1.gnu' you'll get an error message because the plot function was used to read in the data for which no terminal was defined (since we don't want a plot). So you can safely ignore the error message


  //for QDA_plot2.gnu
  //evaluate Z = X^T*A*X + X^T*B + C and save x,y coordinates where Z=0
  arma::vec a = arma::linspace(min(X.col(0)),max(X.col(0))); //a is a vector of length 100
  arma::vec b = arma::linspace(min(X.col(1)),max(X.col(1)));
  //create meshgrid
  arma::mat a_grid(a.n_rows,a.n_rows); //a_grid is a 100 by 100 square matrix
  arma::mat b_grid(a.n_rows,a.n_rows);
  for (int i=0;i<a.n_rows;i++) {
    for (int j=0;j<a.n_rows;j++) {
      a_grid(j,i) = a(i,0);
      b_grid(i,j) = b(i,0);
    }
  }
  //evaluate Z at each point in the grid
  arma::mat a_grid_vec; //a_grid_vec is a vector of length 10000
  arma::mat b_grid_vec;
  a_grid_vec = arma::vectorise(a_grid);
  b_grid_vec = arma::vectorise(b_grid);
  arma::mat ab;
  ab = arma::join_horiz(a_grid_vec,b_grid_vec);
  arma::mat Z(ab.n_rows,1);
  for(int i=0;i<Z.n_rows;i++) {
    Z.row(i) = ab.row(i)*A*trans(ab.row(i)) + B*trans(ab.row(i)) + C;
  }
  //extract values where Z is close to zero
  arma::mat Z_zero;
  for (int i=0;i<Z.n_rows;i++) {
    if (Z(i,0)<0.25 && Z(i,0)>-0.25) {
      Z_zero = join_vert(Z_zero,ab.row(i));
    }
  }
  //read Z_zero into a file
  std::ofstream of_Zzero("QDA_dat.out");
  for (int i=0;i<Z_zero.n_rows;i++) {
    of_Zzero << Z_zero.row(i);
  }

  std::ofstream of_QDA2gnu("QDA_plot2.gnu");
  of_QDA2gnu << "reset" << std::endl;
  of_QDA2gnu << "set terminal png" << std::endl;
  of_QDA2gnu << "set output 'QDA_plot.png'" << std::endl;
  of_QDA2gnu << std::endl;
  of_QDA2gnu << "set xlabel 'Feature 1'" << std::endl;
  of_QDA2gnu << "set ylabel 'Feature 2'" << std::endl;
  of_QDA2gnu << std::endl;
  of_QDA2gnu << "plot 'QDA_dat.out' with lines title 'decision boundary', 'Class0.out' title 'Species 1', 'Class1.out' title 'Species 2'" << std::endl;


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
catch (const int& error1) {
  std::cerr << "You can only classify two classes at a time" << std::endl;
}
catch (const char& error2) {
  std::cerr << "Code allows for only two features to be plotted at a time" << std::endl;
}

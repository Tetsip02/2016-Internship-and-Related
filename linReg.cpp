#include <iostream>
#include <fstream>
#include "vectorclass.cpp"
#include <armadillo>

int main() {

/*
  vec y;
  y.read("q2y.dat");


for (int i=0; i<yrows; ++i) {
  ofs_y << y_data[i] << std::endl;
}


// ** X matrix **

std::ifstream ifs_X("q2x.dat");

int Xrows=100;
int Xcols=1;

if (!ifs_X) {
  std::cout << "could not read file " << std::endl;
  return 1;
} 

std::ofstream ofs_X("X_matrix");

if(!ofs_X) {
  std::cout << "could not open file" << std::endl;
}

// **read data **

double Xelement;

double** X_data = new double*[Xrows];
for (int i=0 ;i<Xrows; ++i) {
  X_data[i] = new double[Xcols]; //assgin an array for each element
}
int Xr=0;
int Xc=0;
while (ifs_X >> Xelement) { 
  X_data[Xr][Xc] = Xelement;
  if (Xc==Xcols-1) {
    Xc=0;
    ++Xr;
  } else {
    ++Xc;
  }
}


// ** add intercept term **

double intercept[Xrows];
for (int i=0; i<Xrows; ++i) {
  intercept[i]=1;
}
for (int i=0; i<Xrows; ++i) {
  for (int j=0; j<Xcols; ++j) {
    X_data[i][j+1]=X_data[i][j];
  }
  X_data[i][0]=intercept[i];
}

// ** output data **

for (int i=0; i<Xrows ;++i) {
  for (int j=0; j<Xcols+1; ++j) {
    ofs_X << X_data[i][j] << " ";
  }
  ofs_X << std::endl;
}

*/

/** ML part **/

/** initialize theta **/

//vec theta(Xcols+1);
//double* theta = new double[Xcols + 1];
//
//for (int i=0 ; i<Xcols+1; ++i) {
//  theta[i]= 0;
//}



arma::mat y;
y.load("q2y.dat");
arma::mat X_data;
X_data.load("q2x.dat");
arma::mat icept(X_data.n_rows,1);
icept.ones();
arma::mat X=join_horiz(icept, X_data);
arma::mat theta(X.n_cols,1);
theta.zeros();
theta=(pinv((X.t()*X)))*(X.t())*y;
arma::mat h=X*theta;



}

//GDA settings file
std::string data = "FisherIris.dat";
//in FisherIris.dat the first four columns contain the data (one feature per
//column and one data poin per row). The fifth column contains the labels.
//First 50 rows belong to Setosa class
//Rows 51 to 100 belong to versicolor class
//Last 50 rows belong to Virginica class

int numExp = 100; //number of training examples that are read into X
int numFeat = 2; //number of features that are read into X

const double PI=3.14159265358979323;
const double threshold=0.5;


//If you have new data that you want to fit, change the boolean below to true and uncomment the line below. The algorithm will output a file LDAHypothesis.out, containing the predictions.
bool newDat = false;
std::string newData;
//newData = "[name-of-data-file]";

//LDA or QDA?
bool LDA = false;

//Settings File for Logistic Regression

//in FisherIris.dat the first four columns contain the data (one feature per
//column and one data poin per row). The fifth column contains the labels.
std::string dat = "FisherIris.dat";

const int defaultNumIter = 50;

//If you have new data that you want to fit, change the boolean below to trueand uncomment the line below. The algorithm will output a file logRegHypothesis.out, containing the predictions.
bool newDat = false;
std::string newData;
//newData = "[name-of-data-file]";

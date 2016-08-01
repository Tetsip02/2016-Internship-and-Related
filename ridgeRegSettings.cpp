//Ridge Regression settings

/*Training data
--Elements within the same row have to be seperated with a space; each row in one line.
--X represents one example in each row and one feature in each column
--target vector y contains one example in each row
*/
std::string y_dat = "y_OLS.dat";
std::string X_dat = "X_OLS.dat";

const double lambda = 1;
const double alpha = 0.01;  //learning rate used in batch Gradient descent
const int numIter = 1500;  //iterations for batch gradient descent

//If you have new data that you want to fit, change the boolean below to true and uncomment the line below. The algorithm will output a file linRegHypothesis.out, containing the predictions.
bool newDat = false;
std::string newData;
//newData = "[name-of-data-file]";

//Settings File for Logistic Regression

/*Training data
--Elements within the same row have to be seperated with a space; each row in one line.
--rows of X are training samples and rows of y are corresponding 0/1
*/
//std::string y_dat = "y_logReg.dat";
//std::string X_dat = "X_logReg.dat";
std::string dat = "irisDataRevised";

const int defaultNumIter = 50;

//If you have new data that you want to fit, change the boolean below to trueand uncomment the line below. The algorithm will output a file logRegHypothesis.out, containing the predictions.
bool newDat = false;
std::string newData;
//newData = "[name-of-data-file]";


/*Information on Iris flower data:
number of classses = 3
classes:
    0 = setosa
    1 = versicolor
    2 = virginica
number of training examples = 150
number of features = 4
features:
    1 = sepal length
    2 = sepal width
    3 = petal length
    4 = petal width

//GDA settings file
//format for training data: each line row represents one training example and each column one feature, except for the last one. last colmn consists of labels

std::string data = "irisDataRevised";  //in Iris data, 0 is the label for setosa and 1 for versicolor
int numExp = 100;
int numFeat = 2;

const double PI=3.14159265358979323;
const double threshold=0.5;


//If you have new data that you want to fit, change the boolean below to true and uncomment the line below. The algorithm will output a file LDAHypothesis.out, containing the predictions.
bool newDat = false;
std::string newData;
//newData = "[name-of-data-file]";

//LDA or QDA?
bool LDA = false;

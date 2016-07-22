//settings file for ordinary least squares

/*Training data
--Elements within the same row have to be seperated with a space; each row in one line.
--X represents one example in each row and one feature in each column
--target vector y contains one example in each row
*/
std::string y_dat = "q2y.dat";
std::string X_dat = "q2x.dat";

//If you have new data that you want to fit, change the boolean below to trueand uncomment the line below. The algorithm will output a file linRegHypothesis.out, containing the predictions.
bool newDat = "False";
std::string newData;
//newData = "[name-of-data-file]";

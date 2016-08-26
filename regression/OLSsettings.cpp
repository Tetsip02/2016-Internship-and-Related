//settings file for ordinary least squares

/*Training data
--Elements within the same row have to be seperated with a space; each row in one line.
--X represents one example in each row and one feature in each column
--target vector y contains one example in each row
*/
//std::string y_dat = "y_OLS.dat";
//std::string X_dat = "X_OLS.dat";
std::string dat = "OLS_dataset_BostonHousingData.dat";

//If you have new data that you want to fit, change the boolean below to true and uncomment the line below. The algorithm will output a file linRegHypothesis.out, containing the predictions.
bool newDat = false;
std::string newData;
//newData = "[name-of-data-file]";

/*Information on boston housing data
Source: UCI machine learning database
number of training examples: 506
number of features: 13
 1. CRIM      per capita crime rate by town
 2. ZN        proportion of residential land zoned for lots over 25,000 sq.ft.
 3. INDUS     proportion of non-retail business acres per town
 4. CHAS      Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 5. NOX       nitric oxides concentration (parts per 10 million)
 6. RM        average number of rooms per dwelling
 7. AGE       proportion of owner-occupied units built prior to 1940
 8. DIS       weighted distances to five Boston employment centres
 9. RAD       index of accessibility to radial highways
 10. TAX      full-value property-tax rate per $10,000
 11. PTRATIO  pupil-teacher ratio by town
 12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 13. LSTAT    % lower status of the population
The 14th column contains the value of the homes in $1000's
*/

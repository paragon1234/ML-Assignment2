# ML-Assignment2
Python program to execute Naive Bayes and SVM.

An executable shell script with the name - run.sh.

## Running run.sh

Depending on the input command line arguments, the shell script should call/invoke the appropriate function/code and 
generate all the output for the respective question.

The first input argument is always the question number, second argument - relative or absolute path of the train file, third argument - absolute or relative path of test file and further arguments, if any depends on the question.

### Arguments for different questions:

#### Question 1: Naive Bayes
./run.sh 1 <path_of_train_data> <path_of_test_data>  <part_num>

./run.sh 1 train.json test.json a

Here, 'part_num' can be a-e or g. This should train classifier using train data and report accuracy train/test data.


#### Question 2: Weighted Linear Regression
./run.sh 2 <path_of_train_data> <path_of_test_data> <binary_or_multi_class> <part_num>

./run.sh 2 test.csv train.csv 0 a

Here, 'binary_or_multi_class' is 0 for binary classification and 1 for multi-class. 'part_num' is part number which can be a-c for binary classification and a-d for multi-class.

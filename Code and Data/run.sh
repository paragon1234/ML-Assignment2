#!/bin/sh

if [ $1 -eq 1 ]
then
 ./NaiveBayes.py $2 $3 $4
elif [ $4 -eq 0 ]
then
 ./SVM.py $2 $3 $4 $5
else
 ./SVM_multiclass.py $2 $3 $4 $5
fi
#!/bin/bash

### definitions

path_to_zip="Enron.zip"
path_to_full="ls Enron/*.txt"
path_to_spam="Enron/spam"
path_to_ham="Enron/ham"

### create the folder that'll receive the data

mkdir Enron/

### unzip to the folder

unzip -d Enron/ $path_to_zip 

### create ham and spam folders

mkdir $path_to_ham
mkdir $path_to_spam

### loop through the folders and move to spam or ham folders

for f in $path_to_full; do
	if [[ $f == *".ham."* ]]; then
		mv $f $path_to_ham
	else
		mv $f $path_to_spam
	fi
done
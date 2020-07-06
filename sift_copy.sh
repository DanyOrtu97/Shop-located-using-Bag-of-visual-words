#!/bin/bash

cd /media/andrea/Dati2/CV_Proj/handsonbow
for i in `seq 1 16`;
do
	cd dataset/train_set/split_by_class/$i 
	cp *.sift ../../../../SIFT/train/$i
	cd /media/andrea/Dati2/CV_Proj/handsonbow
	cd dataset/test_set/split_by_class/$i 
	cp *.sift ../../../../SIFT/test/$i
	cd /media/andrea/Dati2/CV_Proj/handsonbow
done

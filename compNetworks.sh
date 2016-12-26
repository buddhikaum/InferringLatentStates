#!/bin/bash

N=$(head -1 config.txt)
python GenerateNW.py $N
tail -$N adj_temp.txt > random_adj${N}.txt
jot $N 0 > int.txt
./var-inf 1
mkdir Training
mv meas_1* hidden_1* ./Training/
./var-inf 2
mkdir Parameters
mv theta* lambda_* ./Parameters
./var-inf 1
mkdir Testing
mv meas_1* hidden_1* ./Testing/
./var-inf 3
	

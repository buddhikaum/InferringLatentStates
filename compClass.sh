#!/bin/bash

tot_cls=$(($(head -3 config.txt | tail -1)-1))
N=$(head -1 config.txt)
t=$(head -2 config.txt | tail -1)
for cls_v in `seq 0 ${tot_cls}`;
do
	echo $cls_v > cls_label.txt
	python mydata.py $t $N $N $cls_v
	cat my_pvals.txt > my_pfact.txt
	bash roc_datagen.sh
	python getroc.py $cls_v
done

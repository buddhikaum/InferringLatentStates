
#!/bin/bash

rm true_states.txt

while read line
do
	cat ./Testing/hidden_1_${line}.txt >> true_states.txt
done < int.txt


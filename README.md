# covid-cnn

cat metadata.csv | awk -F ',' '{ print $1 }' > full_patient_set.txt

less full_patient_set.txt | wc -l

grep -o -f full_patient_set.txt test_COVIDx_CT-3A.txt | uniq > test_patients.txt

grep -v -w -f test_patients.txt full_patient_set.txt > train_val_patients.txt

less train_val_patients.txt | wc -l
5566
less full_patient_set.txt | wc -l
6069
less test_patients.txt | wc -l
503

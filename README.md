# covid-cnn

### Keep the original test set, then rebuild the train and validation sets from the full data set

---

- cat metadata.csv | awk -F ',' '{ print $1 }' > full_patient_set.txt
- grep -o -f full_patient_set.txt test_COVIDx_CT-3A.txt | uniq > test_patients.txt
- grep -v -w -f test_patients.txt full_patient_set.txt > train_val_patients.txt
- grep -v -w -f test_patients.txt metadata.csv > train_val_metadata.csv
- grep -v -w -f train_val_patients.txt metadata.csv > test_metadata.csv

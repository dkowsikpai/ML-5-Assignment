## Source Files

- Training file is in `src/train_model.py`
To run use the following code
```
python src/train_model.py --data ./data/Monthly_data_cmo.csv
```
- EDA for the dataset can be found in the `src/eda-preprocessing.ipynb`
- Trainied models are and standard scaler are saved in the `./data` dictionary
- Testing of the model can be done using `src/src/test_model.py`
To run use the following code
```
python src/test_model.py --data ./data/Monthly_data_cmo.csv --model ./data/model.bin --scaler ./data/std_scaler.bin 
```

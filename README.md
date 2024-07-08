# Instructions

## Divide the data with different step lengths and create the dataset
- `python grouping.py`
  - Origin data in folder `Datas` will be divided into a new folder called `Step_XXX`.

## Run different models to classify the dataset
- `python ./models/MLP.py ./models/CNN.py ./models/RNN.py ./models/LSTM.py ./models/Transformer.py`
  - Running results will be saved in folder `logs`.

## Plot the results
- `python merged.py`
  - Precision, Recall, F1 Score, mAP, and Loss in different methods will be shown in `merged.pdf`.

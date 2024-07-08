#![Final result](merged.pdf "Final result")

#Divide the datas with different steps lengths and create the dataset
-`python grouping.py` 
##origin datas in folder `Datas` will be divided into a new folder called `Step_XXX`.

#Run different models to to classify the dataset
-`python ./models/MLP.py ./models/CNN.py ./moelds/RNN.py ./models/LSTM.py ./models/Transformer.py`
##running result will be saved in folder `logs`

#Plot the results
-`python merged.py`
##Precision,Recall,F1 Score,mAP and Loss in different methods will be shown in `merged.pdf`

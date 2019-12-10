# Diamond_price

This is a project based on a [kaggle competition](https://www.kaggle.com/c/diamonds-datamad1019/overview)

In this competition we need to predict the diamond price.

My mains features were the 4 Cs of a diamond (carat, cut, color and clarity)

## Cleaning and transformation
--------------------
In the src folder you would find a file named cleaning.py in which one I change the categorical colums ('clarity', 'cut' and 'color') to numerical and having in mind how close are between them, i.e:

```
def num_clarity(value):
    dict_clarity = {
        'I1': 0,
        'SI1': 1,
        'SI2': 2,
        'VS1': 3,
        'VS2': 4,
        'VVS1': 5,
        'VVS2': 6,
        'IF': 7
    }
    return dict_clarity[value]
```
Also I tried to create a new colum named 'magic' based on the 'magic size' but no improve was observed. If you are curious [here](https://diamondsandpreciousgems.blogspot.com/2012/07/diamond-magic-sizes.html) you have a link.

Also did an standarization with StandarScaler from sklearn

## Approaches
-----------------
In the approaches fold, there are 2 files in which ones I did the first approaches.

In first_look.py I try several models in order to have an aproximations of which one is the best for this data set. All the outputs were recorded in records.txt in the output folder.

I observed that the best model was Random Forest Regression (RFR), so I opted to do another approach with only RFR in silva_grid.py. Also the records of all approaches were saved in records_RFR.txt in the output folder.

## Predictions
----------------
Finally in the predictions folder there are 2 files:

- silva_pred.py : Here I predict the diamond price with the best result obtained in the silva_grid.py. This file generate a .csv in the output with the predictions.

- h2o_pred.py : H2O power trust her automl. This file generate a .csv in the output with the predictions.
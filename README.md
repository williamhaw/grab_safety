# Grab Safety Challenge

https://www.aiforsea.com/safety

## Instructions

1. Please refer to the `Grab Safety Challenge Model Test` notebook.
2. Change the values of `feature_dir` and `label_dir` to the path to the directory where the feature and label data are.
3. Run the entire notebook. The model should be able to be deserialized if the required versions below are met.
4. Otherwise please run the `Grab Safety Challenge Final Models Training` notebook and use the model generated.
5. Accuracy and AUC score are calculated in the last two cells.

## Required Versions
- Python 3.7.3
- pandas 0.23.4 (newest version makes tsfresh crash)
- tsfresh 0.11.2
- keras 2.2.4
- scikit-learn 0.20.3

## Approach

Originally I tried to use a [LSTM](https://keras.io/layers/recurrent/#lstm) layer in my model, but discarded it as it was overfitting and giving poor performance and predictive power on the test set, even when augmented with extra aggregate features.

In the end I used a 2-layer perceptron with the features below as it gave better performance and predictive power.

Features generated:

tsfresh generates the following aggregate features for each column of the time series for each trip
    - length
    - min
    - max
    - mean
    - median
    - standard deviation
    - variance

I also computed derivatives of the acceleration, gyro and speed columns for each direction, then I computed the following aggregates for the new columns
    - min
    - max
    - mean
    - median
    - standard deviation

The intuition behind this is that in a perfectly smooth trip, the car is moving at a constant speed. Therefore the acceleration measurements should remain constant (non-zero because of gravity and orientation). As the movement of the car becomes more erratic, the change in acceleration values becomes bigger [Jerk (Physics)](https://en.wikipedia.org/wiki/Jerk_(physics)). Similar logic applies to the gyroscope and GPS speed values as well. 
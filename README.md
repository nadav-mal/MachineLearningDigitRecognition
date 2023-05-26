Written by Nadav Malul and Ely Asaf
Used a map data structure to easily add and remove different features.

Manualy writing digit features for the sake of dimensionality reduction.

The hand written digits used from sklearn data sets library.

From the input, we make a map with all features, so that each entry is a feature vector.

each vector element is the return value of a feature function which receives the image matrix as input.

To show the feature's quality and donation to the classification proccess,

we plot graphs of features and combination of features. At last, we print some metrics such as a confustion matrix to show the classification made.

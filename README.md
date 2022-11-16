# Naive-Bayes-Classifier-from-Scratch
Here, we are implementing Naive Bayes classifier on A-Z handwritten letters dataset, from scratch without using libraries like sklearn. The dataset is divided into two parts, training set which is 80 percent of the dats, and the testing set, which is 20 percent of the data.
First, we calculate the priors for each class, and then the class conditional densities.
Finally, using the computed priors and class conditional densities, we compute the posteriors and use them to predict a class for a data point drawn from the test set.
The accuracy can be computes as the count of correct predictions divided by the total predictions(which is equal to the length of the test dataset).

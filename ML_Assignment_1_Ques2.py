import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

Y_total=26
D_total=784
bin_size=128
n_bins=int(256/bin_size)
f=np.ones(n_bins*Y_total*D_total).reshape(n_bins,Y_total,D_total)
count=np.zeros(Y_total)
prior=np.zeros(Y_total)
X=[]
Y=[]
filename="C:\\Users\\user\\Desktop\\file_2.csv"

def load_csv_file():
    df=pd.read_csv(filename)
    for index,row in df.iterrows():
        Y.append(row[0])
        X.append(row[1:])

#Function computes priors for each class k =[0 25]
def compute_priors(y_train):
    for y in y_train:
        count[y]+=1

    global prior
    prior=count/len(y_train)

#Function computes class conditional densities for each class k = [0 25]
def calculate_class_conditional_densities(x_train,y_train):
    for i in range(len(y_train)):
        D=x_train[i]
        Y=y_train[i]
        for j in range(D_total):
            bin=D[j]//bin_size
            f[bin][Y][j]+=1

    for bin in range(n_bins):
        for k in range(Y_total):
            for j in range(D_total):
                f[bin][k][j]=f[bin][k][j]/(count[k]+n_bins)

def predict_labels(x_test,y_test):
    count_correct_predictions=0
    count_total_predictions=0
    conf_matrix=np.zeros(Y_total*Y_total).reshape(Y_total,Y_total)

    for i in range(len(y_test)):
        Y_actual=y_test[i]
        D=x_test[i]
        q=np.zeros(Y_total)
        #For each data point in the testing set, we calculate posteriors for k=[0,25]
        for j in range(D_total):
            for k in range(Y_total):
                bin=D[j]//bin_size
                q[k]+=math.log(f[bin][k][j])

        for k in range(Y_total):
            q[k]=q[k]+prior[k]
        
        max=q[0]
        Y_predicted=0
        #Predicted class is the one whose posterior is the highest for a given data point
        for k in range(Y_total):
            if q[k]>max:
                max=q[k]
                Y_predicted=k
        
        #If Predicted class is same as the actual class, we increment the count of correct predictions made so far
        if Y_predicted==Y_actual:
            count_correct_predictions+=1
        
        count_total_predictions+=1
        conf_matrix[Y_actual][Y_predicted]+=1
    
    print("The accuracy of Naive Bayes Classifier is : ",(count_correct_predictions/count_total_predictions)*100)
    print(conf_matrix)

load_csv_file()
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
compute_priors(y_train)
calculate_class_conditional_densities(x_train,y_train)
predict_labels(x_test,y_test)





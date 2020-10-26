#importing libraries
import pandas as pd
import numpy as np

#importing dataset and dropping unnecessary columns
dataset = pd.read_csv('dataset.csv')
dataset = dataset.drop('CustomerId', 1)
dataset = dataset.drop('Surname',1)
#forming the matrix
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, [10]].values
X = np.c_[np.ones((X.shape[0])),X]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1= LabelEncoder()
X[:,2]=labelencoder_X1.fit_transform(X[:,2])
labelencoder_X2= LabelEncoder()
X[:,3]=labelencoder_X2.fit_transform(X[:,3])
onehotencoder= OneHotEncoder(categorical_features=[2])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,2:]
 

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

np.random.seed(3684)
SynapseMatrix1 = np.random.random((X_train.shape[1],11))
SynapseMatrix2 = np.random.random((11,1))
def sigmoidActivation(x):
    return 1 / (1 + np.exp(-x))
    # return expit(x)     #More practical solution. However, you need to import it: from scipy.special import expit
def derivative_of_sigmoid(x):
    return x*(1-x)
for i in np.arange(1000):
    layer0 = X_train
    layer1 = sigmoidActivation(np.dot(layer0, SynapseMatrix1))
    layer2 = sigmoidActivation(np.dot(layer1, SynapseMatrix2))
        
    layer2_error = y_train - layer2
    
    if (i% 1000)==0:
        print ("Error: "+str(np.mean(np.abs(layer2_error))))
    
    layer2_delta = layer2_error*derivative_of_sigmoid(layer2)
    
    
    layer1_error = layer2_delta.dot(SynapseMatrix2.T)
    
    layer1_delta = layer1_error*derivative_of_sigmoid(layer1)
    
    #Update weights
    SynapseMatrix2 = SynapseMatrix2 + layer1.T.dot(layer2_delta)
    SynapseMatrix1 = SynapseMatrix1 + layer0.T.dot(layer1_delta)

layer2 = layer2>=0.5

#output after the training
print("Output after training:")
print(layer2)

from sklearn.metrics import accuracy_score
accuracy1= accuracy_score(y_train, layer2) 

from sklearn.metrics import classification_report
Classification1 = classification_report(y_train, layer2)

layer0test = X_test
layer1test = sigmoidActivation(np.dot(layer0test, SynapseMatrix1))
layer2test = sigmoidActivation(np.dot(layer1test, SynapseMatrix2))
layer2test = layer2test>=0.5 

from sklearn.metrics import accuracy_score
accuracy2= accuracy_score(y_test, layer2test) 

from sklearn.metrics import classification_report
Classification2 = classification_report(y_test, layer2test)

    #code for test dataset##
testdata= pd.read_csv('judge.csv')   
testdata = testdata.drop('CustomerId', 1)
testdata = testdata.drop('Surname',1)
testdata = np.c_[np.ones((testdata.shape[0])),testdata]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder1= LabelEncoder()
testdata[:,2]=labelencoder1.fit_transform(testdata[:,2])

labelencoder2= LabelEncoder()
testdata[:,3]=labelencoder2.fit_transform(testdata[:,3])

onehotencoder= OneHotEncoder(categorical_features=[2])
testdata= onehotencoder.fit_transform(testdata).toarray()
testdata= testdata[:,2:]

from sklearn.preprocessing import StandardScaler
sc_testdata = StandardScaler()
testdata = sc_testdata.fit_transform(testdata)

layer0testdata = testdata
layer1testdata = sigmoidActivation(np.dot(layer0testdata, SynapseMatrix1))
layer2testdata = sigmoidActivation(np.dot(layer1testdata, SynapseMatrix2))
layer2testdata = layer2testdata>=0.5 

print(layer2testdata)

layer2testdata = pd.DataFrame(layer2testdata)
labelencoder3 = LabelEncoder()
layer2testdata = labelencoder3.fit_transform(layer2testdata)

Col = ['CustomerId']
testdatacol = pd.read_csv('judge.csv')
testdatacol_ans = pd.DataFrame(testdatacol["CustomerId"])
testdatacol_ans[['Exited']]= pd.DataFrame(layer2testdata)
testdatacol_ans.to_csv("D:/Downloads/judge_pred.csv")


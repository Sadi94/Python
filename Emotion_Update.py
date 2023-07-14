#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Naive Bayes


# In[2]:


#Import the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, confusion_matrix

dataM = pd.read_csv('D:\Data\Emotion_New.csv') #dataM = modified Data after preprocessing


# In[3]:


dataM


# In[4]:


#Prepare the data for classification
# Extract the features (Text) and the Emotions from the DataFrame
X = dataM['Text'].values   #defined Text data from csv file
y = dataM['Emotion'].values  #defined Emotion data from csv file

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Vectorize the text data 
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)


# In[5]:


# Start the timer
start_time = time.time()


# In[6]:


#Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

k = 10  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)
predicted = cross_val_predict(classifier, X_train_counts, y_train, cv=kf)

#Make predictions on the test data
predictions = classifier.predict(X_test_counts)

# Calculate confusion matrix
confusion = confusion_matrix(y_test, predictions)
true_positives = confusion.diagonal()
false_positives = confusion.sum(axis=0) - true_positives
false_negatives = confusion.sum(axis=1) - true_positives

# Calculate precision
precision = precision_score(y_test, predictions, average='weighted')
precision_percentage = precision * 100


TN = confusion[0, 0]  # True Negative
FP = confusion[0, 1]  # False Positive
FN = confusion[1, 0]  # False Negative
TP = confusion[1, 1]  # True Positive

TPR = TP / (TP + FN)  # True Positive Rate
FPR = FP / (FP + TN)  # False Positive Rate

tpr = TP / (TP + FN)
fpr = FP / (FP + TN)

#Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, predictions)
accuracy_percentage = accuracy * 100
#print('Naive Bayes Accuracy:', accuracy)
print("Naive Bayes Accuracy: {:.2f}%".format(accuracy_percentage))
confusion = confusion_matrix(y_test,predictions)
print('Naive Bayes confusion Matrix: \n',confusion)
#print("Naive Bayes Precision:", precision)
print("Naive Bayes Precision: {:.2f}%".format(precision_percentage))
print("Naive Bayes True Positives:", true_positives)
print("Naive Bayes True Positive Rate (TPR):{:.2f}%".format(tpr))

print("Naive Bayes False Positives:", false_positives)
print("Naive Bayes False Positive Rate (FPR):{:.2f}%".format(fpr))

print("Naive Bayes False Negatives:", false_negatives)

# Calculate the training time
training_time = time.time() - start_time

# Print the training time
print("Naive Bayes Training time:{:.2f}".format(training_time), "seconds", "or")
print("Naive Bayes Training time:{:.2f}".format(training_time/60), "minutes")


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve

# Plot the TP-FP curve
plt.plot(false_positives , color='red', label='False Positive')
plt.plot(true_positives , color='blue', label='True Positive')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes True Positive vs False Positive')
# Add a legend
plt.legend()
plt.show()


# In[8]:


#Support Vector Regression


# In[9]:


#Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

dataM = pd.read_csv('D:\Data\Emotion_New.csv') #dataM = modified Data after preprocessing


# In[10]:


# Assuming your target variable is named 'Emotion' and the text feature is named 'Text'
# Extract the features and target variable
X = dataM['Text']
y = dataM['Emotion']

#Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

k = 10  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)
predicted = cross_val_predict(classifier, X_train_counts, y_train, cv=kf)

#Make predictions on the test data
predictions = classifier.predict(X_test_counts)


# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = vectorizer.transform(X_test)


# In[11]:


# Create and train the SVC model
model = SVC()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)


# In[12]:


# Generate the confusion matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
#confusion_matrix = confusion_matrix(y_test, y_pred)
#Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
accuracy_percentage = accuracy * 100
print("Supporting Vector Regression Accuracy: {:.2f}%".format(accuracy_percentage))
print('Supporting Vector Regression confusion Matrix: \n',confusion_matrix)

TN = confusion_matrix[0, 0]  # True Negative
FP = confusion_matrix[0, 1]  # False Positive
FN = confusion_matrix[1, 0]  # False Negative
TP = confusion_matrix[1, 1]  # True Positive

TPR = TP / (TP + FN)  # True Positive Rate
FPR = FP / (FP + TN)  # False Positive Rate

tpr = TP / (TP + FN)
fpr = FP / (FP + TN)
true_positives = np.diag(confusion_matrix)
#true_positives = confusion_matrix.diagonal()
false_positives = confusion_matrix.sum(axis=0) - true_positives
false_negatives = confusion_matrix.sum(axis=1) - true_positives

print("Supporting Vector Regression True Positives:", true_positives)
print("Supporting Vector Regression True Positive Rate (TPR):{:.2f}%".format(tpr))
print("Supporting Vector Regression False Positives:", false_positives)
print("Supporting Vector Regression True Positive Rate (FPR):{:.2f}%".format(fpr))
print("Supporting Vector Regression False Negatives:", false_negatives)

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')
precision_percentage = precision * 100

#print("Supporting Vector Regression Precision:", precision)
print("Supporting Vector Regression Precision: {:.2f}%".format(precision_percentage))

# Start the timer
#start_time = time.time()
# Calculate the training time
training_time = time.time() - start_time

# Print the training time
print("Supporting Vector Regression Training time:{:.2f}".format(training_time), "seconds","or")
print("Supporting Vector Regression Training time:{:.2f}".format(training_time/60), "minutes" )


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve

# Plot the TP-FP curve
plt.plot(false_positives , color='red', label='False Positive')
plt.plot(true_positives , color='blue', label='True Positive')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Supporting Vector Regression True Positive vs False Positive')
# Add a legend
plt.legend()
plt.show()


# In[16]:


#KNN 


# In[17]:


#Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# In[18]:


from sklearn.neighbors import KNeighborsClassifier


# In[19]:


k = 10  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)  #kf = kfold
predicted = cross_val_predict(classifier, X_train_counts, y_train, cv=kf)

#Make predictions on the test data
KNNPredictions = classifier.predict(X_test_counts)


# In[20]:


vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

kn = 10  # Number of neighbors to consider
knn = KNeighborsClassifier(n_neighbors=kn)
knn.fit(X_train_vectorized, y_train)


# In[22]:


#Evaluate the accuracy of the classifier
y_pred_KNN = knn.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred_KNN)
accuracy_percentage = accuracy * 100
#print('KNN Accuracy:', accuracy)
print("KNN Accuracy: {:.2f}%".format(accuracy_percentage))
confusion_mat = confusion_matrix(y_test, y_pred_KNN)
print("KNN Confusion Matrix : \n",confusion_mat)

# Calculate precision
kprecision = precision_score(y_test, KNNPredictions, average='weighted')
kprecision_percentage = kprecision * 100

#print("KNN Precision:", precision)
print("KNN Precision: {:.2f}%".format(kprecision_percentage))

KTN = confusion_mat[0, 0]  # True Negative
KFP = confusion_mat[0, 1]  # False Positive
KFN = confusion_mat[1, 0]  # False Negative
KTP = confusion_mat[1, 1]  # True Positive

KTPR = KTP / (KTP + KFN)  # True Positive Rate
KFPR = KFP / (KFP + KTN)  # False Positive Rate

Ktpr = KTP / (KTP + KFN)
Kfpr = KFP / (KFP + KTN)
Ktrue_positives = np.diag(confusion_mat)
#true_positives = confusion_matrix.diagonal()
Kfalse_positives = confusion_mat.sum(axis=0) - Ktrue_positives
Kfalse_negatives = confusion_mat.sum(axis=1) - Ktrue_positives

print("KNN True Positives:", Ktrue_positives)
print("KNN True Positive Rate (TPR):{:.2f}%".format(Ktpr))
print("KNN False Positives:", Kfalse_positives)
print("KNN True Positive Rate (FPR):{:.2f}%".format(Kfpr))
print("KNN False Negatives:", Kfalse_negatives)

# Start measuring the training time
#start_time = time.time()
# Calculate the training time
training_time = time.time() - start_time

# Print the training time
print("KNN Training time:{:.2f}".format(training_time), "seconds","or")
print("KNN Training time:{:.2f}".format(training_time/60), "minutes")


# In[24]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve

# Plot the TP-FP curve
plt.plot(Kfalse_positives , color='red', label='False Positive')
plt.plot(Ktrue_positives , color='blue', label='True Positive')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN True Positive vs False Positive')
# Add a legend
plt.legend()
plt.show()


# In[25]:


#decision Tree


# In[26]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from sklearn import tree
import matplotlib.pyplot as plt


# In[28]:


X = dataM['Text']
y = dataM['Emotion']

# Preprocess the text data from text to numeric for tree
vectorizer = TfidfVectorizer(stop_words='english') 
X = vectorizer.fit_transform(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_predDT = clf.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_predDT)
print("Decision Tree Confusion Matrix : \n",cm)

#Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_predDT)
accuracy_percentage = accuracy * 100
print("Decision Tree Accuracy: {:.2f}%".format(accuracy_percentage))

# Calculate precision
DTprecision = precision_score(y_test, y_predDT, average='weighted')
DTprecision_percentage = DTprecision * 100

#print("Decision Tree Precision:", precision)
print("Decision Tree Precision: {:.2f}%".format(DTprecision_percentage))

DTN = cm[0, 0]  # True Negative
DFP = cm[0, 1]  # False Positive
DFN = cm[1, 0]  # False Negative
DTP = cm[1, 1]  # True Positive

DTPR = DTP / (DTP + DFN)  # True Positive Rate
DFPR = DFP / (DFP + DTN)  # False Positive Rate

Dtpr = DTP / (DTP + DFN)
Dfpr = DFP / (DFP + DTN)
Dtrue_positives = np.diag(cm)
#true_positives = confusion_matrix.diagonal()
Dfalse_positives = cm.sum(axis=0) - Dtrue_positives
Dfalse_negatives = cm.sum(axis=1) - Dtrue_positives

print("Decision Tree True Positives:", Dtrue_positives)
print("Decision Tree True Positive Rate (TPR):{:.2f}%".format(Dtpr))
print("Decision Tree False Positives:", Dfalse_positives)
print("Decision Tree True Positive Rate (FPR):{:.2f}%".format(Dfpr))
print("Decision Tree False Negatives:", Dfalse_negatives)

# Calculate the training time
training_time = time.time() - start_time

# Print the training time
print("Decision Tree Training time:{:.2f}".format(training_time), "seconds","or")
print("Decision Tree Training time:{:.2f}".format(training_time/60), "minutes")


# In[29]:


pip install scikit-learn matplotlib


# In[30]:


# # Create and train the decision tree classifier
clf = DecisionTreeClassifier(max_leaf_nodes=20,random_state=42)
clf.fit(X_train, y_train)

#visualize decision Tree
plt.figure(figsize=(40, 20))
tree.plot_tree(clf, filled=True, feature_names=vectorizer.get_feature_names())
plt.title('Decision Tree',fontsize=50)
plt.show()


# In[38]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve

# Plot the TP-FP curve
plt.plot(Dfalse_positives , color='red', label='False Positive')
plt.plot(Dtrue_positives , color='blue', label='True Positive')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree True Positive vs False Positive')
# Add a legend
plt.legend()
plt.show()


# In[32]:


#Random Forest


# In[33]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#Train the Random Forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# In[34]:


#Make predictions
y_pred_RF = rf.predict(X_test)

#Evaluate the model
RFconfusion_mat = confusion_matrix(y_test, y_pred_RF)
print("Random Forest Confusion Matrix: \n", RFconfusion_mat)

#Evaluate the accuracy of the classifier
RFaccuracy = accuracy_score(y_test, y_pred_RF)
RFaccuracy_percentage = RFaccuracy * 100
print("Random Forest Accuracy: {:.2f}%".format(RFaccuracy_percentage))

# Calculate precision
RFprecision = precision_score(y_test, y_pred_RF, average='weighted')
RFprecision_percentage = RFprecision * 100

#print("Decision Tree Precision:", precision)
print("Random Forest Precision: {:.2f}%".format(RFprecision_percentage))

RF_TN = RFconfusion_mat[0, 0]  # True Negative
RF_FP = RFconfusion_mat[0, 1]  # False Positive
RF_FN = RFconfusion_mat[1, 0]  # False Negative
RF_TP = RFconfusion_mat[1, 1]  # True Positive

RF_TPR = RF_TP / (RF_TP + RF_FN)  # True Positive Rate
RF_FPR = RF_FP / (RF_FP + RF_TN)  # False Positive Rate

RF_tpr = RF_TP / (RF_TP + RF_FN)
RF_fpr = RF_FP / (RF_FP + RF_TN)
RFtrue_positives = np.diag(RFconfusion_mat)
#true_positives = confusion_matrix.diagonal()
RFfalse_positives = RFconfusion_mat.sum(axis=0) - RFtrue_positives
RFfalse_negatives = RFconfusion_mat.sum(axis=1) - RFtrue_positives

print("Random Forest True Positives:", RFtrue_positives)
print("Random Forest True Positive Rate (TPR):{:.2f}%".format(RF_tpr))
print("Random Forest False Positives:", RFfalse_positives)
print("Random Forest True Positive Rate (FPR):{:.2f}%".format(RF_fpr))
print("Random Forest False Negatives:", RFfalse_negatives)

# Calculate the training time
training_time = time.time() - start_time
minutes = ((training_time/60))

# Print the training time
#print(" Training time: %.2f seconds" % training_time)
print("Random Forest Training time:{:.2f}".format(training_time), "seconds","or")
print("Random Forest Training time:{:.2f}".format(minutes), "minutes")


# In[35]:


# # Create and train the decision tree classifier
rf = RandomForestClassifier(max_leaf_nodes=20,random_state=42)
rf.fit(X_train, y_train)

#visualize decision Tree
plt.figure(figsize=(40, 20))
tree.plot_tree(rf.estimators_[0], feature_names=vectorizer.get_feature_names(), class_names=rf.classes_, filled=True)
plt.title('Random Forest',fontsize=50)
plt.show()


# In[39]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve

# Plot the TP-FP curve
plt.plot(RFfalse_positives , color='red', label='False Positive')
plt.plot(RFtrue_positives , color='blue', label='True Positive')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest True Positive vs False Positive')
# Add a legend
plt.legend()
plt.show()


# In[37]:


#end -> effort never dies


# In[ ]:





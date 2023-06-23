#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


data = pd.read_csv('D:\Data\Emotion_final.csv')


# In[4]:


data


# In[5]:


data.head(10)


# In[6]:


data['Text'][0:10] #training set size[10] from head or "0" index


# In[7]:


punctuation ='''!()-[]{};:'"\,<>./?@#$%^&*_~'''
my_str = "i didnt feel humiliated"
# remove punctuation from the string

no_punc = ""
for char in my_str:
    if (char not in punctuation):
        no_punc = no_punc + char


# In[8]:


no_punc


# In[10]:


import re      #regex expression
s = "string. with. Punctuation?"
s = re.sub(r'[^\w\s]','',s)


# In[11]:


s


# In[12]:


import nltk


# In[12]:


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# In[13]:


nltk.word_tokenize("i feel romantic too")


# In[14]:


#Using python split
s = 'i feel romantic too'
s.split(' ') 


# In[15]:


from nltk.corpus import stopwords


# In[16]:


stop_words = stopwords.words('english')
print(stop_words)


# In[17]:


# Add or remove Stopwords
stop_words.append('work')


# In[18]:


stop_words


# In[19]:


#or
print(stop_words)


# In[24]:


data['Text'][0:10] #train set size[n] from head


# In[25]:


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


# In[22]:


for index,row in data.iterrows():
    filter_sentence = []
    sentence = row['Text']
    sentence = re.sub(r'[^\w\s]',' ',sentence)  #cleaning
    words = nltk.word_tokenize(sentence)  #tokenization
    words = [w for w in words if not w in stop_words]   #stop words removal
    for word in words:
        filter_sentence.append(lemmatizer.lemmatize (word)) 
        print(filter_sentence) 
       


# In[23]:


data.to_csv("D:\Data\Emotion_New.csv", index=False)


# In[24]:


data


# In[36]:


import csv


# In[37]:


import string


# In[38]:


# Open the input CSV file
with open('D:\Data\Emotion_final.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    # Create a list to store the modified rows
    modified_rows = []
    
    for row in reader:
        modified_row = []
        
        for item in row:
            # Remove punctuation from each item in the row
            item_without_punctuation = ''.join(char for char in item if char not in string.punctuation)
            modified_row.append(item_without_punctuation)
        
        modified_rows.append(modified_row)
        
        # Open a new CSV file for writing
        with open('D:\Data\Emotion_New.csv', 'w', newline='') as file:
            writer = csv.writer(file)
        # Write the modified rows to the new CSV file
        # writer.writerows(modified_rows)       
            writer.writerows(modified_rows)


# In[53]:


data1


# In[6]:


import pandas as pd
data1 = pd.read_csv('D:\Data\Emotion_New.csv')


# In[7]:


data1


# In[1]:


#Import the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[2]:


dataM = pd.read_csv('D:\Data\Emotion_New.csv') #dataM = modified Data after preprocessing


# In[3]:


dataM #print or show data


# In[11]:


#Prepare the data for classification
# Extract the features (Text) and the Emotions from the DataFrame
X = dataM['Text'].values   #defined Text data from csv file
y = dataM['Emotion'].values  #defined Emotion data from csv file

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


#Vectorize the text data 
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)


# In[13]:


#Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)


# In[28]:


#Make predictions on the test data
predictions = classifier.predict(X_test_counts)


# In[15]:


#Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, predictions)
print('Accuracy:', accuracy)


# In[16]:


confusion = confusion_matrix(y_test,predictions)
print(confusion)


# In[21]:


# Start the timer
start_time = time.time()


# In[18]:


# Calculate the training time
training_time = time.time() - start_time

# Print the training time
print("Naive Bayes Training time: %.2f seconds" % training_time)


# In[ ]:


#Support Vector Regression


# In[19]:


from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


# In[20]:


# Assuming your target variable is named 'Emotion' and the text feature is named 'Text'
# Extract the features and target variable
X = dataM['Text']
y = dataM['Emotion']


# In[21]:


# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()


# In[22]:


# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = vectorizer.transform(X_test)


# In[23]:


# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = vectorizer.transform(X_test)


# In[24]:


# Create and train the SVC model
model = SVC()
model.fit(X_train_tfidf, y_train)


# In[25]:


# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)


# In[26]:


# Generate the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[30]:


# Start the timer
start_time = time.time()
# Calculate the training time
training_time = time.time() - start_time

# Print the training time
print("Supporting Vector Regression Training time: %.2f seconds" % training_time)


# In[29]:


#Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# In[ ]:





# In[47]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


# In[48]:


# Extract the features and target variable
X = dataM['Text']
y = dataM['Emotion']


# In[49]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[50]:


# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


# In[51]:


# Initialize and train the SVM classifier
svm = SVC()
svm.fit(X_train_vectors, y_train)


# In[54]:


# Predict the target variable for the test set
y_pred = svm.predict(X_test_vectors)

# Calculate the confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("SVM Confussion Matrix : \n",confusion_mat)


# In[55]:


#Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# In[ ]:


#KNN


# In[56]:


from sklearn.neighbors import KNeighborsClassifier


# In[57]:


vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# In[58]:


k = 5  # Number of neighbors to consider
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_vectorized, y_train)


# In[62]:


y_pred_KNN = knn.predict(X_test_vectorized)
confusion_mat = confusion_matrix(y_test, y_pred_KNN)
print("KNN Confussion Matrix : \n",confusion_mat)


# In[63]:


#Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred_KNN)
print('Accuracy:', accuracy)


# In[65]:


# Calculate the training time
training_time = time.time() - start_time

# Print the training time
print("KNN Training time: %.2f seconds" % training_time)


# In[ ]:


#decision Tree


# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from sklearn import tree
import matplotlib.pyplot as plt


# In[9]:


X = dataM['Text']
y = dataM['Emotion']


# In[10]:


# Preprocess the text data from text to numeric for tree
vectorizer = TfidfVectorizer(stop_words='english') 
X = vectorizer.fit_transform(X)


# In[8]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[11]:


# Make predictions on the test set
y_predDT = clf.predict(X_test)


# In[79]:


# Generate the confusion matrix
cm = confusion_matrix(y_test, y_predDT)
print("Decision Tree Confussion Matrix : \n",cm)


# In[80]:


#Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_predDT)
print('Accuracy:', accuracy)

# Calculate the training time
training_time = time.time() - start_time

# Print the training time
print("Decision Tree Training time: %.2f seconds" % training_time)


# In[12]:


#visualize decision Tree
plt.figure(figsize=(40, 20))
tree.plot_tree(clf, filled=True, feature_names=vectorizer.get_feature_names())
plt.show()


# In[ ]:


#Random Forest


# In[24]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn import tree
import matplotlib.pyplot as plt


# In[14]:


#Train the Random Forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# In[17]:


#Make predictions
y_pred_RF = rf.predict(X_test)


# In[19]:


#Evaluate the model
confusion_mat = confusion_matrix(y_test, y_pred_RF)
print("Random Forest Confusion Matrix: \n", confusion_mat)


# In[23]:


#Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred_RF)
print('Accuracy:', accuracy)

# Calculate the training time
training_time = time.time() - start_time

# Print the training time
print("Random Forest Training time: %.2f seconds" % training_time)


# In[27]:


# Visualize the first decision tree in the Random Forest
plt.figure(figsize=(12, 6))
tree.plot_tree(rf.estimators_[0], feature_names=vectorizer.get_feature_names(), class_names=rf.classes_, filled=True)
plt.show()


# In[ ]:


#LogisticRegression


# In[81]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[41]:


X = dataM['Text']  # Replace 'text_column' with the actual column name containing the text data
y = dataM['Emotion']  # Replace 'target_column' with the actual column name containing the target variable


# In[30]:


#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
# Adjust the test_size and random_state as needed


# In[31]:


#Convert the text data into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# In[32]:


model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


# In[33]:


y_pred_LR = model.predict(X_test_vectorized)


# In[34]:


confusion = confusion_matrix(y_test, y_pred_LR)
print("Logistic Regression Confusion Matrix: \n",confusion)


# In[35]:


#Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred_LR)
print('Accuracy:', accuracy)

# Calculate the training time
training_time = time.time() - start_time

# Print the training time
print("Logistic Regression Training time: %.2f seconds" % training_time)


# In[83]:


# Generate random x values
np.random.seed(0)
x = np.random.normal(0, 1, 100).reshape(-1, 1)

# Define the logistic regression function
def logistic_regression(x):
    return 1 / (1 + np.exp(-x))

# Generate random y values based on logistic regression function
y_prob = logistic_regression(x)
y = np.random.binomial(1, y_prob).reshape(-1, 1)

# Fit logistic regression model
model = LogisticRegression()
model.fit(x, y)

# Plot the logistic regression curve
x_vals = np.linspace(-3, 3, 100).reshape(-1, 1)
y_pred = model.predict_proba(x_vals)[:, 1]

plt.scatter(x, y, color='blue', label='Data')
plt.plot(x_vals, y_pred, color='red', label='Logistic Regression Curve')
plt.xlabel('Text')
plt.ylabel('Emotion')
plt.legend()
plt.show()


# In[ ]:


#Visualize data


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt


# In[85]:


dataM['Emotion'].value_counts().plot(kind='bar')
plt.xlabel('Text')
plt.ylabel('Emotion')
plt.title('Emotional Data')
plt.show()


# In[88]:


dataM['Emotion'].value_counts().plot(kind='pie')
plt.axis('equal')
plt.title('Emotional Data')
plt.show()


# In[92]:


pip install wordcloud


# In[4]:


pip install --upgrade Pillow


# In[7]:


from wordcloud import WordCloud

text = ' '.join(dataM['Text'])
wordcloud = WordCloud(width=800, height=400).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Emotional Data in Word Cloud')
plt.show()


# In[ ]:





# In[9]:


##end


# In[ ]:





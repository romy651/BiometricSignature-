import numpy as np
from sklearn.model_selection import train_test_split
import process as pr
import start as st



# using the training size of 75%
train_s = 0.75


# Setting our variables

all_data = pr.get_data()
data_train = []
target_train = []
data_test = []
target_test = []

# Splitting our dataset according to the train_s variable 

for data in  all_data:
    X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], train_size = train_s, random_state = 0) 
    data_train.append(X_train)
    target_train.append(y_train)
    data_test.append(X_test)
    target_test.append(y_test)
    

data_train, target_train, data_test, target_test = np.concatenate(data_train), np.concatenate(target_train), np.concatenate(data_test), np.concatenate(target_test)

# Here we use sklearn framework to normalise and to classify our dta 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
data_train = sc.fit_transform(data_train)
data_test  = sc.transform(data_test)

classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(data_train, target_train)
 
total_data = np.concatenate((data_train, data_test), axis = 0)
total_target = np.concatenate((target_train, target_test), axis = 0)

#Getting the score of the model 
classifier.score(data_test, target_test)   
#Cross validation to get the overall score of the model 

# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(classifier, total_data, total_target
 

#for plotting 
    
color = ['black', 'gray', 'silver', 'firebrick', 'red', 'sienna', 'bisque', 'tan', 'moccasin', 'gold', 'navy', 'blue',
          'plum', 'm', 'coral', 'olive', 'skyblue', 'violet', 'pink', 'hotpink', 'magenta', 'purple', 'lavender', 'yellow',
          'tomato', 'salmon', 'darkred', 'chocolate', 'peru', 'maroon']

data_train = sc.inverse_transform(data_train)
color[0]

# Here we are using PCA for feature reduction in order to plot classified data 

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components = 2)
tr_data = pca.fit_transform(data_train)
j = 0
for ii in range(0, len(data_train), 6):
    plt.plot(tr_data[ii:ii+6, 0], tr_data[ii:ii+6, 1], color = color[j])
    j = j + 1
plt.show()

from sklearn.feature_selection import SelectKBest, chi2
data_test = sc.inverse_transform(data_test)

new_data = SelectKBest(chi2, k = 2).fit_transform(data_test, target_test)
j = 0
for ii in range(0, len(data_train), 6):
    plt.plot(new_data[ii:ii+6, 0], new_data[ii:ii+6, 1], color = color[j])
    j = j + 1 

plt.show()



#single prediction 
# file = '026_g_1.sig'
# points_file = st.get_points_file(file)
# pr.plot_point(np.array(points_file))

# features = st.get_features_file(file)
# features = np.array(features)
# features = features.reshape((1, len(features)))
# features = features[:, :-1]
# features = sc.transform(features)
# classifier.predict(features)
# predicted = classifier.predict(features)[0]


# points = pr.get_points_class(int(predicted))
# pr.plot_point(points)

































import os

import pickle
#ibrary for reading image
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# prepare data
input_dir= '/home/huynguyen/Machine Learning/Image_Classification/clf-data'
categories = ['empty','not_empty']

data=[]
labels=[]
# load and format these data 
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15,15))
        # append all data and flatter all the image to fit the model -> make it an 1D array
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)



#train / test split 
# tratify to make sure that the data lables have the same proportion
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle= True, stratify=labels)

# train classifier
# Support Vector Classification: is a branch of SVM 
classifier = SVC()
# train many image classifier with each combination pair of parameters -> choose the best pair of parameters for model
parameter = [{'gamma':[0.01, 0.001, 0.0001], 'C':[1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameter)

grid_search.fit(x_train, y_train)

#test performance
# choose the best one for model by using best_estimator
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score*100)))

#save the model 
pickle.dump(best_estimator, open('./model.p', 'wb'))


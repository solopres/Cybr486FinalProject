import numpy as np
#used for centering
from scipy.ndimage import center_of_mass
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#import to create a random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from PIL import Image
#import file from OS for testing
import os
#Import needed to shift images
from scipy.ndimage import shift as nd_shift


def preprocess_img(img):
    #convert to grayscale and make an array
    arr = np.array(img.convert("L"))
    #We can then resize the images using
    arr = np.array(Image.fromarray(arr).resize((20,20)))
    #create a blank canvas for the image, needs to be 28x28
    cvs = np.zeros((28,28))
    #place the digits in the center and leave 4 px worth of padding to match the dataset requirements
    cvs[4:24, 4:24] = arr
    #find the weight and where it sits on the canvas
    #not entirley sure if my centering is happening, when the code below to return canvas was added it increased accuracy by 20%
    centY, centX = center_of_mass(cvs)
    #ensures center is row 14 for y this would be the middle of 28
    moveY = int(round(14 - centY))
    #does the same as above but for x
    moveX = int(round(14-centX))
    #does the actual shifiting
    cvs = nd_shift(cvs, shift=[moveY, moveX])
    return cvs.flatten() / 255

#load in mnist data set
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
#input
X = mnist.data / 255
#output
Y = mnist.target.astype(int)
#leave out 20% of mnist data for testing
mnistX_train, mnistX_test, mnistY_train, mnistY_test = train_test_split(X, Y, test_size=.20, random_state=42)


#TEST DATA GOES HERE
folder = r"C:\Users\natha\Downloads\testing_data"
#holds preprocecced images
custom_X = []
#contains the labels(answers) for X
custom_Y = []
#loops through every file in the folder ending in png
for filename in os.listdir(folder):
    if filename.endswith('.png'):
        #sets the actual label to the zero index of the file, this works due to the naming convention we used in class
        actual_label = int(filename[0])
        path = os.path.join(folder, filename)
        #load the image in
        img = Image.open(path)
        #process the image using the preprocess helper
        processed = preprocess_img(img)
        #add the processed images to custom X
        custom_X.append(processed)
        #store the labels in Custom Y
        custom_Y.append(actual_label)
#converts to a numpy array for processed hand drawn images and their label
custom_X = np.array(custom_X)
custom_Y = np.array(custom_Y)
#check to ensure all images and their labels are loaded properly
print(f"\n\tHand drawn images loaded: {len(custom_X)}\n\tlabels loaded: {len(custom_Y)}")

#train test split
customX_train, customX_test, customY_train, customY_test = train_test_split(custom_X, custom_Y, test_size=.30, random_state=42)
#combine with mnist data by stacking
combined_X = np.vstack([mnistX_train, customX_train])
#matches the labels in the same order they were stacked
combined_Y = np.concatenate([mnistY_train, customY_train])
#With us having more hand drawn images I ended up mainly using all of those for testing since that is what will be done for the actual grade
print(f"\tTotal samples after combining the two: {len(combined_X)}")


#training the froest
forest = RandomForestClassifier(n_estimators=250, max_depth=35, min_samples_leaf=3,random_state=42)
#fit the forest with the combined data
forest.fit(combined_X, combined_Y)
#test the forest with mnist data
print(f"\tThe accuracy of forest on MNIST data: {accuracy_score(mnistY_test, forest.predict(mnistX_test)):.2f}")
#validate (test) using the hand drawn images
y_pred_forest = forest.predict(customX_test)
#compare the labels of the images
acc_forest = accuracy_score(customY_test, y_pred_forest)


#uses testing data to validate, dedided to use quite a few print statements to keep track of what is going on
#only used to show the output, helps to see what numbers are being predicted wrong
correct = 0
total = 0
#loop through each test image
for i in range(len(customX_test)):
    #reshape to a 2D array
    sample = customX_test[i].reshape(1, -1)
    #make the prediction then store the zero index element
    forest_pred = forest.predict(sample)[0]
    #stores the actual label to be used to compare labels
    actual_label = customY_test[i]
    #prints output to show what it is getting correct and what it is getting wrong
    print(f"actual={actual_label}, predicted={forest_pred} {'good' if forest_pred == actual_label else 'bad'}")
    #keeps count so we can get an actual percentage later
    if forest_pred == actual_label:
        correct += 1
    total += 1
#second printout for testing accuracy
print(f"\n\tAccuracy on images we made: {correct/total:.2f}")

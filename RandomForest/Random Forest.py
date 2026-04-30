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
    #find rows and columns that contain a non black pixel
    rows, cols = np.where(arr>15)
    #crop the array to the bounding box of the digit
    arr = arr[rows.min():rows.max()+1, cols.min():cols.max()+1]
    #We can then resize the images using LANCZOS
    arr = np.array(Image.fromarray(arr).resize((20,20), Image.LANCZOS))
    #create a blank canvas for the image, needs to be 28x28
    cvs = np.zeros((28,28))
    #place the digits in the center and leave 4 px worth of padding to match the dataset requirements
    cvs[4:24, 4:24] = arr
    #find the weight and where it sits on the canvas
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
X = mnist.data / 255.0
#output
Y = mnist.target.astype(int)
mnistX_train, mnistX_test, mnistY_train, mnistY_test = train_test_split(X, Y, test_size=.30, random_state=42)


#gets the folder where the images are held
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
#converts to a numpy array
custom_X = np.array(custom_X)
custom_Y = np.array(custom_Y)
#check to ensure all images are being loaded
print(f"Hand drawn images loaded: {len(custom_X)}")

#combine the two sets for training
customX_train, customX_test, customY_train, customY_test = train_test_split(custom_X, custom_Y, test_size=.99, random_state=42)
#combine with mnist data by stacking
combined_X = np.vstack([mnistX_train, customX_train])
#matches the labels in the same order they were stacked
combined_Y = np.concatenate([mnistY_train, customY_train])
print(f"all samples after comning the two: {len(combined_X)}")

#make the tree
tree = DecisionTreeClassifier(max_depth=25, random_state=42)
#fit the tree with the trainng data
tree.fit(combined_X, combined_Y)
print(f"Accuracy of a single tree with MNIST data: {accuracy_score(mnistY_test, tree.predict(mnistX_test)):.2f} ")
#now we will want to test with x test and compare to the actual answers which is y test (labels)
Y_pred = tree.predict(customX_test)
#test the accuracy
acc = accuracy_score(customY_test, Y_pred)
print(f"Accuracy of the single tree using Hand drawn images: {acc:.2f}")


#train the froest
forest = RandomForestClassifier(n_estimators=251, max_depth=35, min_samples_leaf=13, random_state=42)
#fit the forest with the combined data
forest.fit(combined_X, combined_Y)
#test the forest with mnist data
print(f"accuracy of forest on MNIST data: {accuracy_score(mnistY_test, forest.predict(mnistX_test)):.2f}")
#validate using the hand drawn images
y_pred_forest = forest.predict(customX_test)
#compare the labels of the images
acc_forest = accuracy_score(customY_test, y_pred_forest)
#just print the accuracy
print(f'The accuracy of the forest is {acc_forest:.2f}')

#uses testing data to validate, dedided to use quite a few print statements to keep track of what is going on
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
if total > 0:
    #outputs the accuracy
    print(f"Accuracy on images we made: {correct/total:.2f}")
else:
    print("No image files found")
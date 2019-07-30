# USAGE
# python classify.py --model fashion.model --labelbin mlb.pickle --image examples/example_01.jpg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import pickle
import os
import data_handling as dh

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to trained model model")
# ap.add_argument("-l", "--labelbin", required=True,
# 	help="path to label binarizer")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

# grab file pathes
print("[INFO] loading user data...")
filePaths = dh.list_file_paths("/home/tp/Workspace/mu_practical_work/data")
filePaths = sorted(filePaths)


# initialize the data and labels
data = None
labels = None
weights = None

# loop over the input files
index = len(filePaths) - 50
for fileName in filePaths[index - 1: index]:
    print("[INFO] ... " + fileName)
    (user_data, user_label, user_weights, timestamps, feature_names, label_names) = dh.read_user_data(fileName)
    # replacing all NaN in the features with 0
    user_data[np.isnan(user_data)] = 0.;
    if data is None:
        data = user_data
        labels = user_label
        weights = user_weights
    else:
        data = np.append(data, user_data)
        labels = np.append(labels, user_label)
        weights = np.append(weights, user_weights)
data = data.reshape((int(data.size/user_data.shape[1]), user_data.shape[1]))
labels = labels.reshape((int(labels.size/user_label.shape[1]), user_label.shape[1]))
weights = weights.reshape((int(weights.size/user_weights.shape[1]), user_weights.shape[1]))
print("[INFO] done loading user data...")

# load the trained convolutional neural network and the multi-label
# binarizer
print("[INFO] loading network...")
model = load_model("trained_model")
mlb = pickle.loads(open("labels", "rb").read())

# classify the input image then find the indexes of the two class
# labels with the *largest* probability
print("[INFO] classifying ...")
index = 500
proba = model.predict(data)[index]

#experimental
groundtruth = labels[index]

# idxs = np.argsort(proba)[::-1][:2]
#
# # loop over the indexes of the high confidence class labels
# for (i, j) in enumerate(idxs):
# 	# build the label and draw the label on the image
# 	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
# 	cv2.putText(output, label, (10, (i * 30) + 25),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# show the probabilities for each of the individual labels
for (label, p, g) in zip(mlb, proba, groundtruth):
	print("{}: {:.2f}% expected: {}".format(label, p * 100, g))

# # show the output image
# cv2.imshow("Output", output)
# cv2.waitKey(0)
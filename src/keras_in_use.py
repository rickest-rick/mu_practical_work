# USAGE
# python train.py --dataset dataset --model fashion.model --labelbin mlb.pickle

# set the matplotlib backend so figures can be saved in the background
import gc
import matplotlib
from keras import Sequential, optimizers
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense

from data_handling import load_user_data, split_features_labels

matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import os

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset (i.e., directory of images)")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to output model")
# ap.add_argument("-l", "--labelbin", required=True,
# 	help="path to output label binarizer")
# ap.add_argument("-p", "--plot", type=str, default="plot.png",
# 	help="path to output accuracy/loss plot")
# args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 75
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# # grab the image paths and randomly shuffle them
# print("[INFO] loading csv...")
# imagePaths = sorted(list(paths.list_images(args["dataset"])))
# random.seed(42)
# random.shuffle(imagePaths)

# initialize the data and labels
data = []
labels = []

# read the user data
user_data = load_user_data()
features_df, labels_df = split_features_labels(user_data)
user_data = None
features_df.reset_index(inplace=True)

# get uuid column and remove them, the timestamps and the label_source from
# the labels
Y = labels_df.values
index_cols = features_df.columns[[0, 1, 2, -1]]
features_df.drop(index_cols, axis=1, inplace=True)
X = features_df.values
features_df = None
gc.collect()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state=42)

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = Sequential()
model.add(Dense(64, input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(51))
model.add(Activation('sigmoid'))

# initiate RMSprop optimizer
opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

model.summary()

# train the network
print("[INFO] training network...")
H = model.fit(x_train, y_train,
              batch_size=32,
              epochs=100,
              validation_data=(x_test, y_test),
              shuffle=True,
              verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save("model")

# # save the multi-label binarizer to disk
# print("[INFO] serializing label binarizer...")
# f = open(args["labelbin"], "wb")
# f.write(pickle.dumps(mlb))
# f.close()
#
# # plot the training loss and accuracy
# plt.style.use("ggplot")
# plt.figure()
# N = EPOCHS
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="upper left")
# plt.savefig(args["plot"])

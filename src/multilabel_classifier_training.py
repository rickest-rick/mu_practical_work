# USAGE ## NOT IMPLEMENTED LIKE THIS CURRENTLY{TP}
# python train.py --dataset dataset --model fashion.model --labelbin mlb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
import matplotlib.pyplot as plt
from keras import Sequential, optimizers
from keras.layers import Activation, Dropout, Dense

matplotlib.use("Agg")

# import the necessary packages
from sklearn.model_selection import train_test_split
import numpy as np
import random
import data_handling as dh
import pickle

# construct the argument parse and parse the arguments
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
FILE_DIMS = (225, 0, 0)

# grab file pathes
print("[INFO] loading user data...")
filePaths = dh.list_file_paths("/home/tp/Workspace/mu_practical_work/data")
filePaths = sorted(filePaths)
random.seed(42)
random.shuffle(filePaths)

# initialize the data and labels
data = None
labels = None
weights = None

# loop over the input files
for fileName in filePaths:
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

#Think this is not needed {TP}
# # binarize the labels using scikit-learn's special multi-label
# # binarizer implementation
# print("[INFO] class labels:")
# mlb = MultiLabelBinarizer()
# labels = mlb.fit_transform(labels)
#
# # loop over each of the possible class labels and show them
# for (i, label) in enumerate(mlb.classes_):
# 	print("{}. {}".format(i + 1, label))

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.2, random_state=42)

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = Sequential()
model.add(Dense(64, input_shape=trainX.shape[1:]))
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

value = testY.shape[1:]
model.add(Dense(testY.shape[1]))
model.add(Activation('sigmoid'))

# initiate RMSprop optimizer
opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

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
H = model.fit(trainX, trainY,
              batch_size=32,
              epochs=EPOCHS,
              validation_data=(testX, testY),
              shuffle=True,
              verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save("model")

# save the labels
print("[INFO] serializing label binarizer...")
f = open("labels", "wb")
f.write(pickle.dumps(label_names))
f.close()

# # plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("plot")

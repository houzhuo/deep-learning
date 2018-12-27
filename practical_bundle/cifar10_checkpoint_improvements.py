
# coding: utf-8

# In[18]:


# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from CV.nn.conv.MiniVGGNet import MiniVGGNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse 
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", 
                required=True, help="path to weights directory")
args = vars(ap.parse_args())


print("[INFO] loading CIFAR-10 data...")
((train_X,train_y),(test_X,test_y)) = cifar10.load_data()
train_X = train_X.astype("float")/255.0
test_X = test_X.astype("float")/255.0

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# construct the callback to save only the *best* model to disk
# based on the validation loss

fname = os.path.sep.join([args['weights'],
        "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss",mode="min",
                             save_best_only=True, verbose=1)
callbacks = [checkpoint]


# train the network
print("[INFO] training network...")
H = model.fit(train_X, train_y, validation_data=(test_X, test_y),
    batch_size=64, epochs=40, callbacks=callbacks, verbose=2)


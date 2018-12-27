
# coding: utf-8

# In[8]:


from CV.nn.conv.LeNet import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras import backend as K
from CV.preprocessing.ImageToArrayPreprocessor import ImageToArrayPreprocessor
from CV.preprocessing.SimplePreprocessor import SimplePreprocessor
from CV.datasets.SimpleDatasetLoader import SimpleDatasetLoader
from CV.nn.conv.ShallowNet import ShallowNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
help="path to the input data")
args = vars(ap.parse_args())

print("[info] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp,iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float")/255.0

#if K.image_data_format() == "channels_first":
#    data = data.reshape(data.shape[0],1,28,28)
#else:
#    data = data.reshape(data.shape[0], 28, 28, 1)
    

(train_X, test_X, train_y, test_y) = train_test_split(data,
                               labels, test_size=0.25,
                                                     random_state=42)

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)



EPOCHS = 25
INIT_LR = 1e-3
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LeNet.build(width=28, height=28, depth=3, classes=2)
model.compile(loss="binary_crossentropy", optimizer=opt,
metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(train_X, train_y, validation_data=(test_X, test_y),
    batch_size=128, epochs=EPOCHS, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(test_X, batch_size=128)
print(classification_report(test_y.argmax(axis=1),predictions.argmax(axis=1),
      target_names=[str(x) for x in lb.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


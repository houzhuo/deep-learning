# import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K

class ResNet:
	@staticmethod
	def residual_module(data, K, stride, chanDim, red=False,
					 reg=0.0001, bnEps=2e-5, bnMom=0.9):
		# the shortcut branch of the ResNet module should be
		# initialize as the input (identity) data
		shortcut = data

		# the first block of the ResNet module are the 1x1 CONVs
		bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,
			momentum=bnMom)(data)
		act1 = Activation("relu")(bn1)
		conv1 = Conv2D(int(kK * 0.25),(1, 1), use_bias = False,
			kernel_regularizer = l2(reg)(act1))
	# the second block of the ResNet module are the 3x3 CONVs
	bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,
		momentum=bnMom)(conv1)
	act2 = Activation("relu")(bn2)
	conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,
		padding="same", use_bias=False,
		 kernel_regularizer=l2(reg))(act2)
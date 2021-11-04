# save the final model to file
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Lambda, MaxPooling2D # convolution layers
from tensorflow.keras.layers import Dense, Dropout, Flatten # core layers
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
import numpy as np
 
# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	(trainBX, trainBY), (testBX, testBY) = mnist.load_data()
	
	mean_px = trainX.mean().astype(np.float32)
	std_px = trainX.std().astype(np.float32)
	trainX = (trainX - mean_px)/(std_px)
	mean_pbx = trainBX.mean().astype(np.float32)
	std_pbx = trainBX.std().astype(np.float32)
	trainBX = (trainBX - mean_pbx)/(std_pbx)



	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	trainBX = trainBX.reshape((trainBX.shape[0], 28, 28, 1))
	testBX = testBX.reshape((testBX.shape[0], 28, 28, 1))

	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	trainBY = to_categorical(trainBY)
	testBY = to_categorical(testBY)

	return trainX, trainY, testX, testY, trainBX, trainBY, testBX, testBY
 
# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm
 
# define cnn model
def define_model():
	#model = Sequential()

	def IMG_model():
		inputs = Input(shape=(28,28,1), name='inA')
		print(inputs)

		x=Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform')(inputs)#, input_shape=(28, 28, 1)))
		x=Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform')(x)
		x=BatchNormalization()(x)
		x=layers.Activation(activations.relu)(x)
		x=MaxPooling2D((2, 2))(x)
		x=Dropout(0.25)(x)
		x=Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
		x=Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
		x=BatchNormalization()(x)
		x=layers.Activation(activations.relu)(x)
		x=MaxPooling2D((2, 2))(x)
		x=Dropout(0.25)(x)
		x=Flatten()(x)
		x=Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
		x=BatchNormalization()(x)
		x=layers.Activation(activations.relu)(x)
		x=Dense(84, activation='relu', kernel_initializer='he_uniform')(x)
		x=BatchNormalization()(x)
		x=layers.Activation(activations.relu)(x)
		x=Dropout(0.25)(x)
		x=Dense(10, activation='softmax')(x)

		inputsB = Input(shape=(28,28,1), name='inB')
		print(inputs)
		y=Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform')(inputsB)#, input_shape=(28, 28, 1)))
		y=Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform')(y)
		y=BatchNormalization()(y)
		y=layers.Activation(activations.relu)(y)
		y=MaxPooling2D((2, 2))(y)
		y=Dropout(0.25)(y)
		y=Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(y)
		y=Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(y)
		y=BatchNormalization()(y)
		y=layers.Activation(activations.relu)(y)
		y=MaxPooling2D((2, 2))(y)
		y=Dropout(0.25)(y)
		y=Flatten()(y)
		y=Dense(256, activation='relu', kernel_initializer='he_uniform')(y)
		y=BatchNormalization()(y)
		y=layers.Activation(activations.relu)(y)
		y=Dense(84, activation='relu', kernel_initializer='he_uniform')(y)
		y=BatchNormalization()(y)
		y=layers.Activation(activations.relu)(y)
		y=Dropout(0.25)(y)
		y=Dense(10, activation='softmax')(y)

		model=Model([inputs, inputsB],[x,y])
		return model
	
	# compile model
	opt = Adam(lr=0.001)#, momentum=0.9)
	model = IMG_model()
	#model.compile(optimizer=opt, 
	#		   loss='categorical_crossentropy',metrics=['accuracy'])
	model.compile(optimizer=opt, 
			   loss=['categorical_crossentropy','categorical_crossentropy'],
			metrics=[['accuracy'],['accuracy']])
	return model
 
# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY, trainBX, trainBY, testBX, testBY= load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	trainBX, testBX = prep_pixels(trainBX, testBX)
	# define model
	model = define_model()
	# fit model
	model.fit([trainX, trainX], [trainY,trainY], epochs=35, batch_size=32, verbose=0)
	# save model
	model.save('final_model.h5')
 
# entry point, run the test harness

run_test_harness()

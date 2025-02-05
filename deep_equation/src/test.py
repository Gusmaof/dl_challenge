from typing import List
import numpy as np


# evaluate the deep model on the test dataset
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
 
# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	#mean_px = trainX.mean().astype(np.float32)
	#std_px = trainX.std().astype(np.float32)
	#trainX = (trainX - mean_px)/(std_px)
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
 
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
 
# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# load model
	model = load_model('resources/trained_model/final_model.h5')
	# evaluate model on test dataset
	acc=np.empty(2)
	acc = model.evaluate([testX, testX], [testY, testY], verbose=0)
	print('> %.3f' % (100*((acc[3]+acc[4])/2)))
	print('Digit A:','> %.3f' %(100*acc[3]))
	print('Digit B:','> %.3f' %(100*acc[4]))
# entry point, run the test harness
run_test_harness()

# example of loading the mnist dataset
from tensorflow import keras 
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Lambda, MaxPooling2D # convolution layers
from tensorflow.keras.layers import Dense, Dropout, Flatten # core layers

from tensorflow.keras.layers import LayerNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.utils.np_utils import to_categorical

from tensorflow.keras.optimizers import SGD

from model import evaluate_model

#from keras.datasets import mnist

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	
	# summarize loaded dataset
	print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
	print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
	# plot first few images
	for i in range(9):
		# define subplot
		pyplot.subplot(330 + 1 + i)
		# plot raw pixel data
		pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
	# show the figure
	pyplot.show()
	
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

# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(2, 1, 1)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(2, 1, 2)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['acc'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_acc'], color='orange', label='test')
	pyplot.show()

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()

#training

trainX,trainY,testX,testY = load_dataset()

trainX,testX = prep_pixels(trainX,testX)
trainY,testY = prep_pixels(trainY,testY)

scores, histories, model = evaluate_model(trainX, trainY, n_folds=5)

#Performance
summarize_diagnostics(histories)
summarize_performance(scores)

# save model
model.save('final_model.h5')
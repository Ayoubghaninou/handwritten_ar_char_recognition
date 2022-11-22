# import the necessary packages
from tensorflow.keras.datasets import mnist
import numpy as np
# import pandas as pd
# from IPython.display import display # Allows the use of display() for DataFrames

# Import libraries needed for reading image and processing it
# import csv
# from PIL import Image
# from scipy.ndimage import rotate

# def load_az_dataset(datasetPath,labelsPath): FOR ARABIC 
def load_az_dataset(datasetPath):
	# initialize the list of data and labels
	data = []
	labels = []

	# loop over the rows of the A-Z handwritten digit dataset
	for row in open(datasetPath):

		# parse the label and image from the row
		row = row.split(",")
		label = int(row[0])
		image = np.asarray([int(x) for x in row[1:]], dtype="uint8")

		# images are represented as single channel (grayscale) images
		# that are 28x28=784 pixels -- we need to take this flattened
		# 784-d list of numbers and repshape them into a 28x28 matrix
		# image = image.reshape(28,28).astype('uint8')
		image = image.reshape((28, 28))
		# image = np.flip(image, 0)
		# image = rotate(image, -90)

		# update the list of data and labels
		data.append(image)
		labels.append(label)

	# convert the data and labels to NumPy arrays
	data = np.array(data, dtype="float32")
	labels = np.array(labels, dtype="int")
	
	# for row in open(labelsPath): FOR ARABIC
	# 	row = row.split(",")
	# 	label = int(row[0])
	# 	labels.append(label)

	# return a 2-tuple of the A-Z data and labels
	return (data, labels)


# def testing_data(): FOR ARABIC
# 	letters_testing_images_file_path = "dataset/ar_dataset/csvTestImages 3360x1024.csv"
# 	letters_testing_labels_file_path = "dataset/ar_dataset/csvTestLabel 3360x1.csv"
# 	testing_letters_images = pd.read_csv(letters_testing_images_file_path, header=None)
# 	testing_letters_labels = pd.read_csv(letters_testing_labels_file_path, header=None)

# 	return (testing_letters_images, testing_letters_labels)

def load_mnist_dataset():

	# load the MNIST dataset and stack the training data and testing
	# data together (we'll create our own training and testing splits
	# later in the project)
	((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
	data = np.vstack([trainData, testData])
	labels = np.hstack([trainLabels, testLabels])

	# return a 2-tuple of the MNIST data and labels
	return (data, labels)
import numpy as np
from keras import layers, models, optimizers

from keras.utils import to_categorical
from keras import callbacks
from keras.models import load_model

from keras import backend as K
K.set_image_data_format('channels_last')

#Chosing GPU if multiple GPUs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

#Setting usage restriction for GPU memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

#hypperparameters
batch_size =32
epochs = 20
lr = 1e-4
lr_decay = 0.99

test = False


# Load MNIST dataset
def load_mnist():
	from keras.datasets import mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
	x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
	y_train = to_categorical(y_train.astype('float32'))
	y_test = to_categorical(y_test.astype('float32'))

	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	return (x_train,y_train), (x_test, y_test)


class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
    	if(logs.get('acc')>0.998):
      		print("\nReached 99.8%")
      		self.model.stop_training = True


#Training model
if not test:
	#BUilding model
	model = models.Sequential([
	layers.Flatten(input_shape=(28,28,1)),
	layers.Dense(32, activation='relu'),
	layers.Dense(32, activation='relu'),
	layers.Dense(10, activation='softmax')
	])

	'''
	model = models.Sequential([
	layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
	layers.MaxPooling2D(2,2),
	layers.Conv2D(64, (3,3), activation='relu'),
	layers.MaxPooling2D(2,2),
	layers.Flatten(),
	layers.Dense(32, activation='relu'),
	layers.Dense(32, activation='relu'),
	layers.Dense(10, activation='softmax')
	])
	'''
	
	# compile the model
	model.compile(optimizer=optimizers.Adam(lr=lr),
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
	model.summary()

	#Defining callbacks
	tb = callbacks.TensorBoard(log_dir='tensorboard-logs', batch_size=batch_size, histogram_freq=10)
	checkpoint = callbacks.ModelCheckpoint('weights-{epoch:02d}.h5', monitor='val_acc',
	                                           save_best_only=True, save_weights_only=False, verbose=1)
	lr_decay_sche = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (lr_decay ** epoch))

	# Training 
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
	          validation_data=[x_test, y_test], callbacks=[tb, checkpoint, lr_decay_sche])

	#model.save_weights('trained_model.h5')
	#model.save('model.h5')
	#print('Trained model saved to trained_model.h5\'' % args.save_dir)


#Testing model
if test:

	#If model is saved. Comment this bloak if only weights are loaded
	#model = load_model('weights-18.h5')		# Give the model file name

	
	# if only weights saved. Comment this block if load_model is used
	model = models.Sequential([
	layers.Flatten(input_shape=(28,28,1)),
	layers.Dense(32, activation='relu'),
	layers.Dense(32, activation='relu'),
	layers.Dense(10, activation='softmax')
	])
	model.load_weights('weights-19.h5')   	# Give the weight file name
	


	model.summary()

	'''
	#If model.evaluate is used
	model.compile(optimizer=optimizers.Adam(lr=lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
	print "Evaluation: ", model.evaluate(x_test, y_test)
	'''

		# Manually evaluating model
    	y_pred = model.predict(x_test, batch_size=100)
    	print('-'*30 + 'Begin: test' + '-'*30)
    	print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/np.float(y_test.shape[0]))







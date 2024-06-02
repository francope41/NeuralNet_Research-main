import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# warnings.filterwarnings("ignore")
# %matplotlib inline
from tensorflow.keras.layers import Input, Add, Dense, Layer, Activation, concatenate,Conv2D, Flatten, MaxPooling2D,BatchNormalization, Dropout
from keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
# Clear the session
# K.clear_session()

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PrintShapeCallback(Callback):
    def __init__(self, model_layer_name):
        super(PrintShapeCallback, self).__init__()
        self.model_layer_name = model_layer_name

    def on_epoch_end(self, epoch, logs=None):
        # Get the output of the layer with the specified name
        layer_output = self.model.get_layer(self.model_layer_name).output
        print(f"After epoch {epoch+1}, shape of x (from layer {self.model_layer_name}): {layer_output.shape}")

print_shape_callback = PrintShapeCallback(model_layer_name='dense_1')  # Assuming 'dense_1' is the name of the Dense layer after h2(x)

class StopAtThresholdCallback(Callback):
    def __init__(self, threshold):
        super(StopAtThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss < self.threshold:
            print(f"\nStopping training as validation loss {val_loss} is below the threshold {self.threshold}")
            self.model.stop_training = True

# callback = StopAtThresholdCallback(threshold=1e-03)
callback = StopAtThresholdCallback(threshold=9.8023e-06)

class H1Layer(Layer):
    def __init__(self, **kwargs):
        super(H1Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.b = self.add_weight(shape=(input_shape[-1],),
                                initializer=RandomNormal(mean=0.0,stddev=0.03),
                                trainable=True)
        super(H1Layer, self).build(input_shape)

    def call(self, x):
        return self.b * (2 * x)
        #return (2 * x) 


class H2Layer(Layer):
    def __init__(self, **kwargs):
        super(H2Layer, self).__init__(**kwargs)

    def call(self, x, h1):
        return (2*x*(h1))-2
    
class H3Layer(Layer):
    def __init__(self, **kwargs):
        super(H3Layer, self).__init__(**kwargs)
        
    def call(self, x, h1, h2):
        return (2 * x * (h2))-(4 * h1)

class H4Layer(Layer):
    def __init__(self, **kwargs):
        super(H4Layer, self).__init__(**kwargs)

    def call(self, x, h2, h3):
        return (2*x*(h3))-(6*h2)   

class H5Layer(Layer):
    def __init__(self, **kwargs):
        super(H5Layer,self).__init__(**kwargs)

    def call(self,x, h3, h4):
        return (2*x*(h4))-(8*h3)
    
class H6Layer(Layer):
    def __init__(self, **kwargs):
        super(H6Layer,self).__init__(**kwargs)
        
    def call(self,x, h4, h5):
        return (2*x*(h5))-(10*h4)

class TensorDecompositionLayer(Layer):
    def __init__(self, rank, **kwargs):
        self.rank = rank
        super(TensorDecompositionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.factors_a = self.add_weight(shape=(input_shape[-1], self.rank),
                                         initializer=RandomNormal(mean=0.0,stddev=0.05),
                                         trainable=True)
        self.factors_b = self.add_weight(shape=(self.rank, input_shape[-1]),
                                        initializer=RandomNormal(mean=0.0,stddev=0.05),
                                        trainable=True)
        super(TensorDecompositionLayer, self).build(input_shape)

    def call(self, x):
        return tf.matmul(tf.matmul(x, self.factors_a), self.factors_b)

 
class Relu_With_Weight(Layer):
    def __init__(self, **kwargs):
        super(Relu_With_Weight, self).__init__(**kwargs)

    def build(self, input_shape):
        self.b = self.add_weight(shape=(input_shape[-1],),
                                initializer=RandomNormal(),
                                trainable=True)
        super(Relu_With_Weight, self).build(input_shape)

    def call(self, x):
        return K.tanh(x * self.b)
    
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

print('Train Images Shape:      ', X_train.shape)
print('Train Labels Shape:      ', y_train.shape)

print('\nValidation Images Shape: ', X_valid.shape)
print('Validation Labels Shape: ', y_valid.shape)

print('\nTest Images Shape:       ', X_test.shape)
print('Test Labels Shape:       ', y_test.shape)

# CIFAR-10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# Convert pixel values data type to float32
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
X_valid = X_valid.astype('float32')
# Calculate the mean and standard deviation of the training images
mean = np.mean(X_train)
std  = np.std(X_train)

# Normalize the data
# The tiny value 1e-7 is added to prevent division by zero
X_train = (X_train-mean)/(std+1e-7)
X_test  = (X_test-mean) /(std+1e-7)
X_valid = (X_valid-mean)/(std+1e-7)

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_valid = to_categorical(y_valid, 10)
y_test  = to_categorical(y_test, 10)

# Data augmentation
data_generator = ImageDataGenerator(
    # Rotate images randomly by up to 15 degrees
    rotation_range=15,
    
    # Shift images horizontally by up to 12% of their width
    width_shift_range=0.12,
    
    # Shift images vertically by up to 12% of their height
    height_shift_range=0.12,
    
    # Randomly flip images horizontally
    horizontal_flip=True,
    
    # Zoom images in by up to 10%
    zoom_range=0.1,
    
    # Change brightness by up to 10%
    brightness_range=[0.9,1.1],

    # Shear intensity (shear angle in counter-clockwise direction in degrees)
    shear_range=10,
    
    # Channel shift intensity
    channel_shift_range=0.1,
)

# Printing the shape of the dataset to confirm
# print("Training data shape:", x_train.shape, y_train.shape)
# print("Test data shape:", x_test.shape, y_test.shape)

def build_model(input_shape, num_classes, filters):
    rank = 3
    input_layer = Input(shape=input_shape)
    x = input_layer
    weight_decay = 0.0001

    h1 = H1Layer()
    h2 = H2Layer()
    h3 = H3Layer()
    h4 = H4Layer()
    # # x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
    # # x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
    # # x = MaxPooling2D((2,2))(x)
    # # x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
    # # x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
    # # x = MaxPooling2D((2,2))(x)
    # # x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
    # # x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
    # # x = MaxPooling2D((2,2))(x)
    # x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
    # x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(rate=0.2)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(rate=0.2)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(rate=0.2)(x)

    x = x_h1 = h1(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = TensorDecompositionLayer(rank)(x)
    x = x_h2 = h2(x,x_h1)
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = TensorDecompositionLayer(rank)(x)
    x = x_h3 = h3(x,x_h1,x_h2)
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = TensorDecompositionLayer(rank)(x)
    x = h4(x,x_h2, x_h3)
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = TensorDecompositionLayer(rank)(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3))(x)
    x = Dropout(rate=0.2)(x)
    x = Flatten()(x)

    #x = Dense(128, activation='relu')(x)  # Increase the number of units
    # x = Dense(64, activation='relu')(x)  # Add more dense layers as needed
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model 


input_shape = (32, 32, 3) # CIFAR-10 images size
num_classes = 10 # Number of classes in CIFAR-10
model_img = build_model(input_shape, num_classes, 128)
model_img.summary()
optimizer = Adam(learning_rate=0.001) # Reduce learning rate
model_img.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


batch_size = 64
epochs = 50

history = model_img.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(X_valid, y_valid),
                    callbacks=[callback])

train_loss = history.history['loss']
val_loss = history.history['val_loss']

val_loss = model_img.evaluate(X_valid, y_valid, verbose=0)
print(f"Validation loss: {val_loss}")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

X_test = X_test
y_true = y_test
y_pred = model_img.predict(X_test)

# Assuming y_true and y_pred are already one-hot encoded, convert them back to class labels
y_true_labels = np.argmax(y_true, axis=1)
y_pred = model_img.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Use the model to make predictions, evaluate on test data
test_loss, test_acc = model_img.evaluate(X_test, y_test, verbose=1)

print('\nTest Accuracy:', test_acc)
print('Test Loss:    ', test_loss)
# # Plotting a confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true_labels, y_pred_labels)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Displaying a few images with their predictions
num_samples = 10  # Number of samples to display
fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))

for i, ax in enumerate(axes):
    ax.imshow(X_test[i])  # Show the image
    ax.set_title(f"True: {y_true_labels[i]}\nPred: {y_pred_labels[i]}")
    ax.axis('off')

plt.show()
#897418
#227530
#225930


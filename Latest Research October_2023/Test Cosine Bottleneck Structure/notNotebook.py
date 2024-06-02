import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Layer, BatchNormalization, Activation, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
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

callback = StopAtThresholdCallback(threshold=1e-05)

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
    def __init__(self, h1, **kwargs):
        super(H2Layer, self).__init__(**kwargs)
        self.h1 = h1

    def call(self, x):
        return (2*x*(self.h1(x)))-2
    
class H3Layer(Layer):
    def __init__(self, h1, h2, **kwargs):
        super(H3Layer, self).__init__(**kwargs)
        self.h1 = h1
        self.h2 = h2
        
    def call(self, x):
        return (2*x*(self.h2(x)))-(4*self.h1(x))

class H4Layer(Layer):
    def __init__(self, h2, h3, **kwargs):
        super(H4Layer, self).__init__(**kwargs)
        self.h2 = h2
        self.h3 = h3

    def call(self, x):
        return (2*x*(self.h3(x)))-(6*self.h2(x))   

class H5Layer(Layer):
    def __init__(self, h3, h4, **kwargs):
        super(H5Layer,self).__init__(**kwargs)
        self.h3 = h3
        self.h4 = h4

    def call(self,x):
        return (2*x*(self.h4(x)))-(8*self.h3(x))
    
class H6Layer(Layer):
    def __init__(self, h4, h5, **kwargs):
        super(H6Layer,self).__init__(**kwargs)
        self.h4 = h4
        self.h5 = h5

    def call(self,x):
        return (2*x*(self.h5(x)))-(10*self.h4(x))

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



# Define the function
def f(x1, x2):
    return 3 * np.cos(2 * np.pi * (x1**2 - x2**2))

# Set the parameters
lower_bound = -1
upper_bound = 1
n_samples = 1500

# Generate the data
x1_values = np.linspace(lower_bound, upper_bound, n_samples).reshape(n_samples, 1)
x2_values = np.linspace(lower_bound, upper_bound, n_samples).reshape(n_samples, 1)

# Get a meshgrid for x1 and x2 values
X1, X2 = np.meshgrid(x1_values, x2_values)

# Calculate y values using the function
y_values = f(X1, X2)

# # Reshape the data for training
# X = np.column_stack((X1.ravel(), X2.ravel()))
# y = y_values.ravel()

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten the data for training
X1_flat = X1.ravel().reshape(-1, 1)  # Reshaped as a 2D array for model input
X2_flat = X2.ravel().reshape(-1, 1)  # Reshaped as a 2D array for model input
y_flat = y_values.ravel()

# Split the data into training and validation sets
X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1_flat, X2_flat, y_flat, test_size=0.2, random_state=42)


# Assuming H1Layer, H2Layer, H3Layer, H4Layer, H5Layer, and TensorDecompositionLayer are pre-defined
def build_model(input_shape, filters):

    rank = 4
    input_x1 = Input(shape=(1,))
    input_x2 = Input(shape=(1,))

    h1 = H1Layer()
    h2 = H2Layer(h1)
    h3 = H3Layer(h1,h2)
    h4 = H4Layer(h2,h3)
    h5 = H5Layer(h3,h4)
    #Branch 1
    branch_x1 = Dense(filters)(input_x1)
    branch_x1 = Dense(filters)(input_x1)

    #Branch 2
    branch_x2 = Dense(filters)(input_x2)
    branch_x2 = Dense(filters)(branch_x2)

    #Merge
    merged = concatenate([branch_x1, branch_x2])
    merged = Dense(filters)(merged)
    merged = h2(merged)
    merged = Dense(filters)(merged)
    merged = TensorDecompositionLayer(rank)(merged)
    merged = h3(merged)
    merged = Dense(filters)(merged)
    merged = TensorDecompositionLayer(rank)(merged)
    merged = h4(merged)
    merged = Dense(filters)(merged)
    merged = TensorDecompositionLayer(rank)(merged)

    output = Dense(1)(merged)  # Single output for your function
    model = Model(inputs=[input_x1, input_x2], outputs=output)

    return model 


input_shape = (2,)
filters = 128
modelN4 = build_model(input_shape, filters)
optimizer = Adam(learning_rate=0.0001) # Reduce learning rate
modelN4.compile(optimizer=optimizer, loss='mse')

batch_size = 128
epochs = 20

# history = modelN4.fit([X1_train], y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(X_val, y_val),
#                     callbacks=[callback])

history = modelN4.fit([X1_train, X2_train], y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=([X1_val, X2_val], y_val),
                      callbacks=[callback])


val_loss = modelN4.evaluate([X1_val, X2_val], y_val, verbose=0)
print(f"Validation loss: {val_loss}")

import matplotlib.pyplot as plt
modelN4.summary()

# 1. Extract loss values
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# 2. Determine the number of epochs
actual_epochs = len(train_loss)

# 3. Create a plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, actual_epochs + 1), train_loss, label='Training Loss', color='b', linewidth=2)
plt.plot(range(1, actual_epochs + 1), val_loss, label='Validation Loss', color='r', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.grid()
plt.show()
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

def main(dirpath: str):
    # Load train and test data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Ensure inputs have correct shape
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    # Convert classes to binary class matrices
    # e.g. 5 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Model building
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), 
            activation='relu',
            input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), 
            activation='relu')),
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(50, activation = 'relu'))
    model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

    # model.summary()

    # Model training
    model.compile(optimizer = 'adam',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
                
    model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split=0.2)

    # Model evaluation
    # results = model.evaluate(X_test, y_test)
    # print('Loss:', results[0])
    # print('Accuracy:', results[1])

    model.save(dirpath + '\model')
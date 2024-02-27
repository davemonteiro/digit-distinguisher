# Digit Distinguisher

A streamlit game/app where the player intentionally tries to confuse a simple CNN trained on MNIST data.

This was formerly online at http://digit-distinguisher.herokuapp.com/.

The game provides two digits, and the user tries to draw a figure (using streamlit-drawable-canvas) that the CNN believes confidently is one of them, but has trouble truly distinguishing.

The MNIST data was imported from tf.keras.datasets.mnist, and the original study can be found here: http://yann.lecun.com/exdb/mnist/

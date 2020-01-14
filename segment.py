# Example of how to load a PNG image
# load the pre-trained U-Net
# Apply U-Net to image and get predicted labels

import tensorflow as tf
from tensorflow.python.keras import models, losses
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# read image
img_rgb = mpimg.imread('dti_short_axis_example.png')
# image is grayscale but it has three identical channels in this particular file, so using only the first one
img = img_rgb[:, :, 0]

# define the soft dice loss function
def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    numerator = 2. * tf.reduce_sum(tf.math.multiply(y_pred, y_true), [1, 2])
    denominator = tf.reduce_sum(tf.math.square(y_pred), [1, 2]) + tf.reduce_sum(tf.math.square(y_true), [1, 2])

    return 1 - tf.math.reduce_mean(tf.math.divide(numerator, (denominator + epsilon)))  # average over classes and batch

# load pre-trained U-Net
cnn_name = 'unet_find_heart_full_fov_5c.hdf5'
model = models.load_model(cnn_name, custom_objects={'soft_dice_loss': soft_dice_loss})

# make sure image is in the right orientation and also if it is an array of images instead of just one
img_shape = img.shape
img = img / np.amax(img)
if img.ndim < 3:
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
else:
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 0, 2)
    img = np.expand_dims(img, axis=-1)

# apply model prediction to each pixel
predicted_labels = model.predict(img)

# discretise the labels to the highest probability class for each pixel
predicted_labels = np.argmax(predicted_labels[0,:,:,:], axis=2)

# display input and output
plt.figure(figsize=(5, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb[:, :, 0], cmap='gray')
plt.title("Mag image",fontsize=8)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(predicted_labels, cmap='cubehelix')
plt.title("Predicted mask",fontsize=8)
plt.axis('off')
plt.show()
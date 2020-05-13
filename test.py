import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

model = tf.keras.models.load_model('model.h5')

model.summary()

img_width, img_height = 96, 96
img = image.load_img('data/test/knife_5.jpg', target_size=(img_width, img_height))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

prediction1 = model.predict(img)

print("knife", prediction1[0])

# make a prediction for a new image.
# load and prepare the image

IMAGE_SIZE = 96

def load_image(filename):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_normalized = (tf.cast(image_decoded, tf.float32)/127.5) - 1
    image_resized = tf.image.resize(image_normalized, (IMAGE_SIZE, IMAGE_SIZE))
    return image_resized


image_res = load_image('data/train/12019.jpg')
img1 = np.expand_dims(image_res, axis=0)

result = model.predict(img1)
print('new knife', result[0])
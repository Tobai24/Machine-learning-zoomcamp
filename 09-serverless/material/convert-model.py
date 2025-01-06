import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('xception_v4_46_0.889.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('clothing-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)
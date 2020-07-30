import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
from keras.applications import VGG16


def create_model():
    model = VGG16()
    cnn=Sequential()
    for capa in model.layers:
        cnn.add(capa)
    cnn.layers.pop()
    for layer in cnn.layers:
        layer.trainable=False
    cnn.add(Dense(6,activation='softmax'))
    
    return cnn


disable_eager_execution()
K.clear_session()

data_entrenamiento = './data/Train'
data_validacion = './data/Validation'
numeroDePiezas = 6



filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
learningRate = 0.0005

batch_size = 16
steps_per_epoch = 50
validation_steps = 100
longitud, altura = 224, 224
epocas=10


entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

print("Indices: ")
print(entrenamiento_generador.class_indices)


validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')


cnn = create_model()

cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=learningRate),
            metrics=['accuracy'])


cnn.fit(
    entrenamiento_generador,
    steps_per_epoch=steps_per_epochs,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps)




target_dir = './conocimiento/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./conocimiento/modelo.h5')
cnn.save_weights('./conocimiento/pesos.h5')

print("Indices: ")
print(entrenamiento_generador.class_indices)
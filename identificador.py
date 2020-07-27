import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura = 150, 150
modelo = './conocimiento/modelo.h5'
pesos_modelo = './conocimiento/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  print(array)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("pred: Caballo")
  elif answer == 1:
    print("pred: Peón")
  elif anwser == 2:
    print("pred: Rey")

  return answer

predict("IMG_0811.JPG")
predict("IMG_0830.JPG")
predict("IMG_0833.JPG")
predict("IMG_0835.JPG")
predict("IMG_0831.JPG")
predict("IMG_0832.JPG")

predict("IMG_0834.JPG")

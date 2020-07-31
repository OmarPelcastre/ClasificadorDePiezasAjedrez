import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from tkinter import * 
from tkinter import filedialog
from PIL import ImageTk, Image 

longitud, altura = 224, 224
modelo = './conocimiento/modelo.h5'
pesos_modelo = './conocimiento/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

root = Tk() 
root.geometry("550x300+300+150") 
root.resizable(width=True, height=True) 

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  text.set("Espere...")
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  print(array)
  result = array[0]
  answer = np.argmax(result)
  
  if answer == 0:
    print("pred: Alfil")
    text.set("Alfil")
  elif answer == 1:
    print("pred: Caballo")
    text.set("Caballo")
  elif answer == 2:
    print("pred: Dama")
    text.set("Dama")
  elif answer == 3:
    print("pred: Peón")
    text.set("Peón")
  elif answer == 4:
    print("pred: Rey")
    text.set("Rey")
  elif answer == 5:
    print("pred: Torre")
    text.set("Torre")

  
  return answer

def openfn(): 
    filename = filedialog.askopenfilename(title='Selecciona una imágen') 
    return filename 
def open_img():
    children = root.winfo_children()
    for child in children: 
      child.destroy() 
    
    x = openfn() 

    img = Image.open(x) 
    img = img.resize((250, 250), Image.ANTIALIAS) 
    img = ImageTk.PhotoImage(img) 
    panel = Label(root, image=img) 
    panel.image = img 
    panel.pack() 
    predict(x)
    text.set("")
    label = Label(root,textvariable=text).pack()
    btn = Button(root, text='open image', command=open_img).pack()

text = StringVar()
text.set("Comienza")
label = Label(root,textvariable=text).pack()
btn = Button(root, text='open image', command=open_img).pack() 

root.mainloop() 










import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt 
import time

start = time.time()

#Define Path
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
test_path = 'Fractions DataSet/Alien_test'


#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)

#Define image parameters
img_width, img_height = 150, 150

#Prediction Function
def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  plt.imshow(x)
  plt.show()
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  dirs=os.listdir(test_path)
  #imgs = np.array(dirs)
  
#  for i in dirs:
#      img=mpimg.imread(os.path.join (test_path , i))
#      plt.imshow(img)
#      plt.show()
#   
  
  array = model.predict(x)
  result = array[0]
  print(result)
  answer = np.argmax(result)
  if answer == 0:
      print("Predicted: Half")
  elif answer == 1:
      print("Predicted: One")
  elif answer == 2:
      print("Predicted: 1/4")
  elif answer == 3:
      print("Predicted: 1/3")  
  elif answer == 4:
      print("Predicted: 3/4")  
  elif answer == 5:
      print("Predicted: 2/3")
  
      return answer
 

#Walk the directory for every image
for i, ret in enumerate(os.walk(test_path)):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue

    print(ret[0] + '/' + filename)
    result = predict(ret[0] + '/' + filename)
    print(" ")
    


#Calculate execution time

end = time.time()
dur = end-start

if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")
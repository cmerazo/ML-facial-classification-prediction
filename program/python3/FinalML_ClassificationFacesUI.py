## =========================================================================
## @author Carlos Mauricio Erazo (erazo.carlos@javeriana.edu.co)
## =========================================================================
# Importar Bibliotecas
import numpy as np
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten,Conv2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
#from sklearn.metrics import confusion_matrix
from pathlib import Path
import os,sys
sys.path.append('./lib/python3/')
import PUJ
import PUJ.Model.Logistic
import joblib


file = 'data/age_gender.csv'

categories = ['0','1','2','3','4']
etnia_names = ['White','Black','Asian','Indian','Hispanic']
gender_names = ['Male','Female']


def load_data():
  df = pd.read_csv(file)
  num_pixels = len(df['pixels'][0].split(" "))
  img_height = int(np.sqrt(len(df['pixels'][0].split(" "))))
  img_width = int(np.sqrt(len(df['pixels'][0].split(" "))))
  print(num_pixels, img_height, img_width)  
  df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(),dtype='float32'))
  return df

def print_img_array(pixeles, index):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(pixeles.iloc[index].reshape(48, 48),"gray")
    plt.show( )

def print_img(pixeles):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(pixeles,"gray")
    plt.show( )

def resize_image(image_file):
  real_image = cv2.imread(image_file)
  new_image = cv2.resize(real_image,(48,48), cv2.IMREAD_GRAYSCALE)
  #print (new_image)
  return new_image

def resize_image_2(image_file):
  real_image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
  new_image = cv2.resize(real_image,(48,48))
  #print (new_image)
  return new_image

def loadConfig():
    cfg_name = './configs/config_experimentos_v1.json'
    f = open(cfg_name)
    hyperpara = json.load(f)
    f.close
    return hyperpara

def executeTrainingAge(X, y, cfg):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
  train_datagen=ImageDataGenerator(rescale=1/255)
  train_generator_age=train_datagen.flow(
      X_train ,y_train ,batch_size=32 
  )

  test_datagen=ImageDataGenerator(rescale=1/255)
  test_generator_age=test_datagen.flow(
      X_test ,y_test ,batch_size=32 
  
  )
  
  model_age = Sequential()

  model_age.add(Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)))
  model_age.add(MaxPooling2D(2,2))


  model_age.add(Conv2D(64,(3,3),activation='relu'))
  model_age.add(MaxPooling2D(2,2))


  model_age.add(Conv2D(64,(3,3),activation='relu'))
  model_age.add(MaxPooling2D(2,2))
  model_age.add(Dropout(0.2))

  model_age.add(Conv2D(128,(3,3),activation='relu'))
  model_age.add(MaxPooling2D(2,2))
  model_age.add(Dropout(0.2))          
            
  model_age.add(Flatten())
  model_age.add(Dropout(0.5))            

  model_age.add(Dense(1,activation='relu'))

  model_age.compile(optimizer='adam' ,loss='mean_squared_error',metrics=['mae'])

  model_age.fit(X_train,y_train,epochs = cfg['epochs_a'], batch_size = 64)

  model_age.summary()

  with open('model_age.txt', 'w') as f:
    model_age.summary(print_fn=lambda x: f.write(x + '\n'))

  model_age.save('model_age.h5')

  return model_age
      
def executeTrainingEthnicity(X, y, cfg):
  #X = data['pixels'] 
  #data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(),dtype='float32'))
  #X = data['pixels'] 
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

  #print(X)

  red = Sequential()
  data_augmentation = Sequential([
    preprocessing.RandomZoom(cfg['zoom'],seed = 23),
    #preprocessing.RandomRotation(0.1,seed = 23),
    #preprocessing.RandomCrop(height = 150 , width = 150)
    #preprocessing.RandomFlip('horizontal',input_shape = (150,150)),
    #preprocessing.RandomTranslation(width_factor = (-0.2, 0.1),height_factor=(-0.3, 0.2))
  ])
  #red.add(data_augmentation)


  red.add(Conv2D(32 , kernel_size = (3,3), activation = 'relu', kernel_initializer = 'he_normal', input_shape = X.shape[1:]))
  red.add(MaxPooling2D(pool_size = (2,2) ))

  red.add(Conv2D(64 , kernel_size = (3,3), activation = 'relu'))
  red.add(MaxPooling2D(pool_size = (2,2)))

  red.add(Conv2D(64 , kernel_size = (3,3), activation = 'relu'))
  red.add(MaxPooling2D(pool_size = (2,2)))

  red.add(Conv2D(128 , kernel_size = (3,3), activation = 'relu'))
  red.add(MaxPooling2D(pool_size = (2,2)))

  red.add(Dropout(0.2))

      # ----
  red.add(Flatten())
  red.add(Dense(cfg['neurons'] , activation = 'relu'))
  red.add(Dense(8,activation = 'softmax'))
    
  red.compile(optimizer= Adam(learning_rate = cfg['lr']),
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

  red.fit(X_train,y_train,epochs = cfg['epochs'], batch_size = 64)

  red.summary()

  with open('model_etcnia.txt', 'w') as f:
    red.summary(print_fn=lambda x: f.write(x + '\n'))

  red.save('model_etcnia.h5')

  train_loss,train_acc = red.evaluate(X_train,y_train)
  test_loss,test_acc = red.evaluate(X_test,y_test)
  print('Accuracy train: ',round(train_acc*100,2))
  print('Accuracy test: ',round(test_acc*100,2))

  # Matriz de confusión para train
  #confusion_matrix_train = pd.DataFrame(confusion_matrix(y_train, red.predict(X_train).argmax(axis=1)[:,np.newaxis]))
  #confusion_matrix_train

  # Matriz de confusión para test
  #confusion_matrix_test = pd.DataFrame(confusion_matrix(y_test, red.predict(X_test).argmax(axis=1)[:,np.newaxis]))
  #confusion_matrix_test

  
  #predictions = red.predict(X)

  return red
   
def main(args):  

    cfg = loadConfig()

    path_to_file = os.getcwd()+'/models/model_etcnia.h5'
    path_to_file_2 = os.getcwd()+'/models/model_age.h5'
    path_to_file_3 = os.getcwd()+'/models/model_gender.joblib'
    
    path = Path(path_to_file)
    path2 = Path(path_to_file_2)
    path3 = Path(path_to_file_3)

    
    print("loading data")
    data = load_data()
    print("data loaded")

    model_g = joblib.load(path3)
    print("model gender loaded")
    #print(data['pixels'][0])
    #print(gender_names[int(model_g.threshold(data['pixels'][0]))])

    X = np.array(data['pixels'].tolist())
    X = np.reshape(X, (-1, 48, 48,1))
    y = data['ethnicity']
    y_age = data['age']
    y_gender = data['gender']
    

    
    #print(pixeles_new_image)
    #print(X[0])
    #print(y)

    if path.is_file():
      model_e = load_model(path_to_file)
      print('model etnia loaded')
    else:
      model_e = executeTrainingEthnicity(X, y, cfg)
      print('new model etnia created')
      
    if path2.is_file():
      model_a = load_model(path_to_file_2)
      print('model age loaded')
    else:
      model_a = executeTrainingAge(X, y_age, cfg)
      print('new model age created')


    
    #####


    # y, X, red, predictions = execute(cfg)
    
    y_size = len(data) - 1
    
   
    
    
    #print(prediction_a)
    #print(prediction_a[5])

    

    if cfg['validate_test']:
      prediction_e = model_e.predict(X)
      prediction_a = model_a.predict(X)
      print('predicting models ... please wait') 
      i = int( input( 'enter a number images between 0 and ' + str( y_size ) + ': ' ) )     
      while i >= 0 and i < y_size:
        #print_img_array(data['pixels'],i)
        print('Etnia real: {} '.format(etnia_names[int(y[i])]))
        print('Etnia predicha: {} '.format(etnia_names[prediction_e[i].argmax()]))

        print('Age real: {} '.format(int(y_age[i])))
        print('Age predicha: {} '.format(prediction_a[i]))

        print('Gender real: {} '.format(gender_names[int(y_gender[i])]))
        print('Gender predicha: {} '.format(gender_names[int(model_g.threshold(data['pixels'][0]))]))
        #print(prediction[i].argmax())
        #print(str(y[i]))
        print_img_array(data['pixels'],i)
        i = int( input( 'again other number of image between 0 and  ' + str( y_size ) + ' or press other key to next step: ' ) )
    
    next = True
    while next:
      file_image = input( 'please introduce url image to predict: ')
      path = os.getcwd()+'/'+file_image    
      path_url = Path(path)
      if path_url.is_file():
        pixeles_new = resize_image(file_image)
        pixeles_new_image = np.reshape(pixeles_new.tolist(), (-1, 48, 48,1))
        prediction_e_n = model_e.predict(pixeles_new_image)
        prediction_a_n = model_a.predict(pixeles_new_image)
        pixeles_new_g = resize_image_2(file_image)
        x_data = np.array([np.array(pixeles_new_g)])
        pixels = x_data.flatten().reshape(1,2304)
        print('your Etnia is: {} '.format(etnia_names[prediction_e_n[0].argmax()]))
        print('your Age is: {} '.format(prediction_a_n[0]))
        print('your Gender is: {} '.format(gender_names[int(model_g.threshold(pixels))]))
        print_img(pixeles_new)
      else:
        r = input( 'url incorrect, you want continue [S/N]: ')
        if (r == 'N' or r == 'n'):
          next = False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, 
                        default="sample", help="Config of hyperparameters")
    args = parser.parse_args()

    
    main(args)
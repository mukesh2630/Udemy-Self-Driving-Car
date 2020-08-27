import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import random

np.random.seed(0)

with open('german-traffic-signs/train.p', 'rb') as f:
    train_data = pickle.load(f)
with open('german-traffic-signs/valid.p', 'rb') as f:
    val_data = pickle.load(f)
with open('german-traffic-signs/test.p', 'rb') as f:
    test_data = pickle.load(f)

#print(type(train_data))

x_train, y_train = train_data['features'], train_data['labels']
x_val, y_val = val_data['features'], val_data['labels']
x_test, y_test = test_data['features'], test_data['labels']

#print(x_train.shape)
#print(x_val.shape)
#print(x_test.shape)

# assert(x_train.shape[0] == y_train.shape[0])
# assert(x_val.shape[0] == y_val.shape[0])
# assert(x_test.shape[0] == y_test.shape[0])
# assert(x_train.shape[1:] == (32, 32, 3))
# assert(x_val.shape[1:] == (32, 32, 3))
# assert(x_test.shape[1:] == (32, 32, 3))

data = pd.read_csv('german-traffic-signs/signnames.csv')
#print(data)

num_of_samples = []

cols = 5
num_classes = 43

# fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(5, 50))
# fig.tight_layout()
# for i in range(cols):
#     for j, row in data.iterrows():
#         x_selected = x_train[y_train == j]
#         axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), :, :], cmap=plt.get_cmap("gray"))
#         axs[j][i].axis("off")
#         if i == 2:
#             axs[j][i].set_title(str(j) + "-" + row["SignName"])
#             num_of_samples.append(len(x_selected))

#print(num_of_samples)
# plt.figure(figsize=(12, 4))
# plt.bar(range(0, num_classes), num_of_samples)
# plt.title("Distribution of the training dataset")
# plt.xlabel("Class number")
# plt.ylabel("Number of images") 
#plt.show()

import cv2

# plt.imshow(x_train[1000])
# plt.axis("off")
#print(x_train[1000].shape)
#print(y_train[1000])

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

img = grayscale(x_train[1000])    
# plt.imshow(img)
# plt.axis("off")    
#print(img.shape)

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

img = equalize(img)    
# plt.imshow(img)
# plt.axis("off")    
#print(img.shape)
    
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

x_train = np.array(list(map(preprocessing, x_train)))
x_val = np.array(list(map(preprocessing, x_val)))
x_test = np.array(list(map(preprocessing, x_test)))

# plt.imshow(x_train[random.randint(0, len(x_train) - 1)])
# plt.axis("off")
print(x_train.shape)
    
x_train = x_train.reshape(34799, 32, 32, 1)    
x_val = x_val.reshape(4410, 32, 32, 1)    
x_test = x_test.reshape(12630, 32, 32, 1)    

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10.)

datagen.fit(x_train)
# for X_batch, y_batch in

batches = datagen.flow(x_train, y_train, batch_size = 15)
x_batch, y_batch = next(batches)

fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(x_batch[i].reshape(32, 32))
    axs[i].axis("off")
#plt.show()
#print(x_batch.shape)
    
#print(x_train.shape)
#print(x_val.shape)
#print(x_test.shape)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
y_val = to_categorical(y_val, 43)

def leNet_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    #Compile Model
    model.compile(Adam(lr=0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def modified_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
#   model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    #Compile Model
    model.compile(Adam(lr=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

#model = leNet_model()
model = modified_model()
print(model.summary())
     
#history = model.fit(x_train, y_train, epochs = 10, validation_data=(x_val, y_val), batch_size = 400, verbose = 1, shuffle = 1)   
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=50),
                            steps_per_epoch=2000,
                            epochs=10,
                            validation_data=(x_val, y_val), shuffle = 1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test Score', score[0])
print(('Test Accuracy', score[1]))


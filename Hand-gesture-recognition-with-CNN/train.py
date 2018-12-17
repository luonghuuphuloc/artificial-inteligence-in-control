import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import model_from_json
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


batch_size = 512
epochs = 8
num_classes = 5
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import os

path = 'images_data'
data = []
labels = []
for label in os.listdir(path):
    print(label)
    imgPath  = path  +'/' + label
    i = 0 
    for image in os.listdir(imgPath):
        img = cv2.imread(path  +'/' + label + '/' + image,0)
        img = cv2.resize(img,(28,28))
        data.append(img)
        if label == 'punch':
            labels.append(1) 
        elif label == 'hand':
            labels.append(2) 
        elif label == 'none':
            labels.append(3) 
        elif label == 'one':
            labels.append(4)
        elif label == 'two':
            labels.append(5)
data = np.asarray(data)
labels = np.asarray(labels)
# print(labels)
data = data.reshape(-1,28,28,1)
labels = labels.reshape(-1,1)
labels_one_hot = to_categorical(labels)
labels_one_hot = np.delete(labels_one_hot,0,1)
print(data.shape, labels.shape)

train_X,test_X,train_Y,test_Y = train_test_split(data, labels_one_hot, test_size=0.2, random_state=13)
train_X,valid_X,train_Y,valid_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(train_X, train_Y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(valid_X, valid_Y))
score = model.evaluate(valid_X, valid_Y, verbose=0)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# y_pred = model.predict(test_X)
model.save('cnn_model.h5')

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

from keras import backend as K
K.set_image_dim_ordering('th')

import os
import numpy as np
import h5py

weights_filename='weights/top_model_weights/fc_inception_cats_dogs_250.hdf5'

def complete_model(weights_path=weights_filename):
    inc_model=InceptionV3(include_top=False,
                          weights='imagenet',
                          input_shape=((3, 150, 150)))

    x = Flatten()(inc_model.output)
    x = Dense(64, activation='relu', name='dense_one')(x)
    x = Dropout(0.5, name='dropout_one')(x)
    x = Dense(64, activation='relu', name='dense_two')(x)
    x = Dropout(0.5, name='dropout_two')(x)
    top_model=Dense(1, activation='sigmoid', name='output')(x)
    model = Model(input=inc_model.input, output=top_model)
    model.load_weights(weights_filename, by_name=True)

    for layer in inc_model.layers[:205]:
        layer.trainable = False


    return model
if __name__ == "__main__":
    print(' ')
    print('-'*50)
    print('''
  ___                  _   _       __   ______  ___ _
 |_ _|_ _  __ ___ _ __| |_(_)___ _ \ \ / /__ / | _ |_)_ _  __ _ _ _ _  _
  | || ' \/ _/ -_) '_ \  _| / _ \ ' \ V / |_ \ | _ \ | ' \/ _` | '_| || |
 |___|_||_\__\___| .__/\__|_\___/_||_\_/ |___/ |___/_|_||_\__,_|_|  \_, |
                 |_|                                                |__/
    ''')
    print('Step_3')
    print('Training complete model with images')
    print('-'*50)

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            'data/img_train/',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            'data/img_val/',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    pred_generator=test_datagen.flow_from_directory('data/img_val/',
                                                         target_size=(150,150),
                                                         batch_size=100,
                                                         class_mode='binary')

    epochs=int(input('How much epochs we need?:'))

    filepath="weights/complete_model_checkpoint_weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

    if not os.path.exists('weights/complete_model_checkpoint_weights/'):
        os.makedirs('weights/complete_model_checkpoint_weights/')
        print('Directory "weights/complete_model_checkpoint_weights/" has been created')


    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model=complete_model()
    print('-'*50)
    print (
    ''' Compiling model with:
    - stochastic gradient descend optimizer;
    - learning rate=0.0001;
    - momentum=0.9 ''')

    model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
                #optimizer='rmsprop',
              metrics=['accuracy'])
    print(' Compiling has been finished')
    print('-'*50)
    print ('Training the model...')
    print ('A half of Cristmas tree is coming :)')
    model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=epochs,
        validation_data=validation_generator,
        nb_val_samples=2000,
        callbacks=callbacks_list,
        verbose=1)

    print('-'*50)
    print('Training the model has been completed')

    loss, accuracy = model.evaluate_generator(pred_generator, val_samples=100)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

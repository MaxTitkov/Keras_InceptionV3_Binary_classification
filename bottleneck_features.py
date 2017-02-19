from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
import os

# Create an Inception model
inc_model=InceptionV3(include_top=False,
                      weights='imagenet',
                      input_shape=((3, 150, 150)))

#
bottleneck_datagen = ImageDataGenerator(rescale=1./255)  #our image generator

train_generator = bottleneck_datagen.flow_from_directory('data/img_train/',
                                        target_size=(150, 150),
                                        batch_size=32,
                                        class_mode=None,
                                        shuffle=False)

validation_generator = bottleneck_datagen.flow_from_directory('data/img_val/',
                                                               target_size=(150, 150),
                                                               batch_size=32,
                                                               class_mode=None,
                                                               shuffle=False)


print(' ')
print('-'*50)
print('''
  ___                  _   _       __   ______  ___ _
 |_ _|_ _  __ ___ _ __| |_(_)___ _ \ \ / /__ / | _ |_)_ _  __ _ _ _ _  _
  | || ' \/ _/ -_) '_ \  _| / _ \ ' \ V / |_ \ | _ \ | ' \/ _` | '_| || |
 |___|_||_\__\___| .__/\__|_\___/_||_\_/ |___/ |___/_|_||_\__,_|_|  \_, |
                 |_|                                                |__/
            ''')
print('Step_1')
print('Fetching the bottleneck features from pretrained InceptionV3 as numpy arrays')
print('-'*50)

if not os.path.exists('bottleneck_features/'):
    os.makedirs('bottleneck_features/')
    print('Directory "bottleneck_features/" has been created')
    print(' ')

print('Saving bn_features_train.npy to bottleneck_features/')
bottleneck_features_train = inc_model.predict_generator(train_generator, 2000)
np.save(open('bottleneck_features/bn_features_train.npy', 'wb'), bottleneck_features_train)

print('Saving bn_features_validation.npy to bottleneck_features/')
bottleneck_features_validation = inc_model.predict_generator(validation_generator, 2000)
np.save(open('bottleneck_features/bn_features_validation.npy', 'wb'), bottleneck_features_validation)
print('Finished')

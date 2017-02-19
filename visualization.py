from keras.preprocessing.image import ImageDataGenerator, img_to_array
from complete_model import complete_model
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
import matplotlib.pyplot as plt

print(' ')
print('-'*50)
print('''
  ___                  _   _       __   ______  ___ _
 |_ _|_ _  __ ___ _ __| |_(_)___ _ \ \ / /__ / | _ |_)_ _  __ _ _ _ _  _
  | || ' \/ _/ -_) '_ \  _| / _ \ ' \ V / |_ \ | _ \ | ' \/ _` | '_| || |
 |___|_||_\__\___| .__/\__|_\___/_||_\_/ |___/ |___/_|_||_\__,_|_|  \_, |
                 |_|                                                |__/
    ''')
print('Step_4')
print('Visualize results')
print('-'*50)
images=int(input('How much images we need?:'))

datagen = ImageDataGenerator(rescale=1./255)

img_generator=datagen.flow_from_directory('data/img_val/',
                                        target_size=(150,150),
                                        batch_size=images,
                                        class_mode='binary')

model=complete_model()

imgs,labels=img_generator.next()
array_imgs=np.transpose(np.asarray([img_to_array(img) for img in imgs]),(0,2,1,3))
predictions=model.predict(imgs)
rounded_pred=np.asarray([round(i) for i in predictions])
wrong=[im for im in zip(array_imgs, rounded_pred, labels, predictions) if im[1]!=im[2]]

plt.figure(figsize=(12,12))
plt.title('Wrong predictions')
for ind, val in enumerate(wrong[:100]):
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace = 0.2, hspace = 0.2)
    plt.subplot(5,5,ind+1)
    im=val[0]
    plt.axis('off')
    plt.text(120, 0, round(val[3], 2), fontsize=11, color='red')
    plt.text(0, 0, val[2], fontsize=11, color='blue')
    plt.imshow(np.transpose(im,(2,1,0)))
plt.show()

right=[im for im in zip(array_imgs, rounded_pred, labels, predictions) if im[1]==im[2]]

plt.figure(figsize=(12,12))
plt.title('First 20 correct predictions')
for ind, val in enumerate(right[:20]):
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace = 0.2, hspace = 0.2)
    plt.subplot(5,5,ind+1)
    im=val[0]
    plt.axis('off')
    plt.text(120, 0, round(val[3], 2), fontsize=11, color='red')
    plt.text(0, 0, val[2], fontsize=11, color='blue')
    plt.imshow(np.transpose(im,(2,1,0)))
plt.show()

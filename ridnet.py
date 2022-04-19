# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:02:26 2022

@author: maydin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 00:43:38 2022

@author: maydin
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,Activation,Add,Multiply,Concatenate,GlobalAveragePooling2D
from tensorflow.keras.models import Model

#Getting the filepaths for train and test data
train_files=['data/patches/train/'+filename for filename in os.listdir('data/patches/train')]
test_files=['data/patches/test/'+filename for filename in os.listdir('data/patches/test')]

BATCH_SIZE=64
NOISE_LEVELS=[15]

def _parse_function(filename):
    '''This function performs adding noise to the image given by Dataset'''
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image = tf.cast(image_decoded, tf.float32)/255.

    noise_level=np.random.choice(NOISE_LEVELS)
    noisy_image=image+tf.random.normal(shape=(40,40,1),mean=0,stddev=noise_level/255)
    noisy_image=tf.clip_by_value(noisy_image, clip_value_min=0., clip_value_max=1.)

    return noisy_image,image


#Creating the Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(np.array(train_files)) 
train_dataset = train_dataset.map(_parse_function)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices(np.array(test_files))
test_dataset = test_dataset.map(_parse_function)
test_dataset = test_dataset.batch(BATCH_SIZE)


iterator = iter(train_dataset)
a, b = iterator.get_next()

print('Shape of single batch of x : ',a.shape)
print('Shape of single batch of y : ',b.shape)



#Plotting the images from dataset to verify the dataset
fig, axs = plt.subplots(1,5,figsize=(20,4))
for i in range(5):
  axs[i].imshow(a[i])
  axs[i].get_xaxis().set_visible(False)
  axs[i].get_yaxis().set_visible(False)
fig.suptitle('Noisy Images',fontsize=20)
plt.show()
fig, axs = plt.subplots(1,5,figsize=(20,4))
for i in range(5):
  axs[i].imshow(b[i])
  axs[i].get_xaxis().set_visible(False)
  axs[i].get_yaxis().set_visible(False)
fig.suptitle('Ground Truth Images',fontsize=20)
plt.show()


#Helper functions

def get_patches(file_name,patch_size,crop_sizes):
    '''This functions creates and return patches of given image with a specified patch_size'''
    image = cv2.imread(file_name,0) 
    #image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = np.reshape(image, (image.shape[0],image.shape[0], 1))
    height, width , channels= image.shape
    patches = []
    for crop_size in crop_sizes: #We will crop the image to different sizes
        crop_h, crop_w = int(height*crop_size),int(width*crop_size)
        image_scaled = cv2.resize(image, (crop_w,crop_h), interpolation=cv2.INTER_CUBIC)
        for i in range(0, crop_h-patch_size+1, patch_size):
            for j in range(0, crop_w-patch_size+1, patch_size):
              x = image_scaled[i:i+patch_size, j:j+patch_size] # This gets the patch from the original image with size patch_size x patch_size
              patches.append(x)
    return patches

def create_image_from_patches(patches,image_shape):
  '''This function takes the patches of images and reconstructs the image'''
  image=np.zeros(image_shape) # Create a image with all zeros with desired image shape
  patch_size=patches.shape[1]
  p=0
  for i in range(0,image.shape[0]-patch_size+1,patch_size):
    for j in range(0,image.shape[1]-patch_size+1,patch_size):
      image[i:i+patch_size,j:j+patch_size]=patches[p] # Assigning values of pixels from patches to image
      p+=1
  return np.array(image)

def predict_fun(model,image_path,noise_level=30):
  #Creating patches for test image
  patches=get_patches(image_path,40,[1])
  test_image=cv2.imread(image_path,0)

  patches=np.array(patches)
  ground_truth=create_image_from_patches(patches,test_image.shape)

  #predicting the output on the patches of test image
  patches = patches.astype('float32') /255.
  patches_noisy = patches+ tf.random.normal(shape=patches.shape,mean=0,stddev=noise_level/255) 
  patches_noisy = tf.clip_by_value(patches_noisy, clip_value_min=0., clip_value_max=1.)
  noisy_image=create_image_from_patches(patches_noisy,test_image.shape)

  denoised_patches=model.predict(patches_noisy)
  denoised_patches=tf.clip_by_value(denoised_patches, clip_value_min=0., clip_value_max=1.)
  denoised_patches = np.squeeze(denoised_patches, axis=3)

  #Creating entire denoised image from denoised patches
  denoised_image=create_image_from_patches(denoised_patches,test_image.shape)

  return patches_noisy,denoised_patches,ground_truth/255.,noisy_image,denoised_image


def plot_patches(patches_noisy,denoised_patches):
  fig, axs = plt.subplots(2,10,figsize=(20,4))
  for i in range(10):

    axs[0,i].imshow(patches_noisy[i])
    axs[0,i].title.set_text(' Noisy')
    axs[0,i].get_xaxis().set_visible(False)
    axs[0,i].get_yaxis().set_visible(False)

    axs[1,i].imshow(denoised_patches[i])
    axs[1,i].title.set_text('Denoised')
    axs[1,i].get_xaxis().set_visible(False)
    axs[1,i].get_yaxis().set_visible(False)
  plt.show()


def plot_predictions(ground_truth,noisy_image,denoised_image, path):
  fig, axs = plt.subplots(1,3,figsize=(15,15))
  axs[0].imshow(ground_truth, cmap='gray')
  axs[0].title.set_text('Ground Truth')
  axs[1].imshow(noisy_image, cmap='gray')
  axs[1].title.set_text('Noisy Image')
  axs[2].imshow(denoised_image, cmap='gray')
  axs[2].title.set_text('Denoised Image')
  plt.savefig(path)
  plt.close()

#https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
def PSNR(gt, image, max_value=1):
    """"Function to calculate peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((gt - image) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def EAM(input):

  x=Conv2D(64, (3,3), dilation_rate=1, padding='same',activation='relu')(input)
  x=Conv2D(64, (3,3), dilation_rate=2, padding='same',activation='relu')(x)

  y=Conv2D(64, (3,3), dilation_rate=3, padding='same',activation='relu')(input)
  y=Conv2D(64, (3,3), dilation_rate=4, padding='same',activation='relu')(y)

  z=Concatenate(axis=-1)([x,y])
  z=Conv2D(64, (3,3),padding='same',activation='relu')(z)
  add_1=Add()([z, input])

  z=Conv2D(64, (3,3),padding='same',activation='relu')(add_1)
  z=Conv2D(64, (3,3),padding='same')(z)
  add_2=Add()([z,add_1])
  add_2 = Activation('relu')(add_2)

  z=Conv2D(64, (3,3),padding='same',activation='relu')(add_2)
  z=Conv2D(64, (3,3),padding='same',activation='relu')(z)
  z=Conv2D(64, (1,1),padding='same')(z)
  add_3=Add()([z,add_2])
  add_3 = Activation('relu')(add_3)

  z = GlobalAveragePooling2D()(add_3)
  z = tf.expand_dims(z,1)
  z = tf.expand_dims(z,1)
  z=Conv2D(4, (3,3),padding='same',activation='relu')(z)
  z=Conv2D(64, (3,3),padding='same',activation='sigmoid')(z)
  mul=Multiply()([z, add_3])

  return mul

def RIDNET():

  input = Input((40, 40, 1),name='input')
  feat_extraction =Conv2D(64, (3,3),padding='same')(input)
  eam_1=EAM(feat_extraction)
  eam_2=EAM(eam_1)
  eam_3=EAM(eam_2)
  eam_4=EAM(eam_3)
  x=Conv2D(1, (3,3),padding='same')(eam_4)
  add_2=Add()([x, input])
  
  model=Model(input,add_2)

  return model



tf.keras.backend.clear_session()
tf.random.set_seed(6908)
ridnet = RIDNET()
ridnet.compile(optimizer=tf.keras.optimizers.Adam(1e-03), loss=['mse'])

ridnet.summary()


def scheduler(epoch,lr):
  return lr*0.9

"""
# TRAINING
noise_level = 15
ridnet.fit(train_dataset, shuffle=True,epochs=30, validation_data = test_dataset)
ridnet.save('ridnet_mydata.h5')
"""

# EVALUATION
#from tensorflow import keras
noise_level = 15
ridnet = tf.keras.models.load_model(f"ridnet_mydata.h5")


psrn_list = []
for i in range(200, 450):
  patches_noisy,denoised_patches,ground_truth,noisy_image,denoised_image=predict_fun(ridnet,f'data/test/{i}.png',noise_level=noise_level)
  print(i)
  print('PSNR of Noisy Image : ',PSNR(ground_truth,noisy_image))
  denoised_psnr = PSNR(ground_truth,denoised_image)
  print('PSNR of Denoised Image : ', denoised_psnr)
  plot_patches(patches_noisy,denoised_patches)
  psrn_list.append(denoised_psnr)

  """
  # SAVE IMAGES
  path = f"result_ridnet/noise_{noise_level}/{i}/"
  os.mkdir(path)
  plt.imsave(f"{path}/gt_{i}.png", ground_truth, cmap = 'gray')
  plt.imsave(f"{path}/pred_{i}.png", denoised_image, cmap = 'gray')
  plt.imsave(f"{path}/noisy_{i}.png", noisy_image, cmap = 'gray')
  plot_predictions(ground_truth, noisy_image, denoised_image, f"result_ridnet/noise_{noise_level}/{i}.png")
  """

print("Average PSNR:", sum(psrn_list) / len(psrn_list))
























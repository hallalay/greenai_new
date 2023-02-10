# -*- coding: utf-8 -*-
"""
Source: D:\GREENAI\Data\ExperimentsAndResults\230_greenAI_imggenerator\Results\13class500Epochs_512x512\modelINCEPTION
modelName: greenAi_13cls_512size_500epochs_INCEPTIONV3_backbone_batch8.hdf5

"""

# from google.colab import drive
# drive.mount('/content/drive')

# !pip3 install patchify
# !pip3 install tensorflow==2.2
# !pip3 install keras==2.3.1
# !pip3 install -U segmentation-models

# # https://youtu.be/jvZm8REF2KY
# """
# Explanation of using RGB masks: https://youtu.be/sGAwx4GMe4E
# """


# import os
# import cv2
import numpy as np

from matplotlib import pyplot as plt
# from PIL import Image
from skimage import io
from keras.models import load_model

import segmentation_models as sm
import cv2
from patchify import patchify, unpatchify

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()

from greenAI_helperfunctions import  label_to_rgb,  calcGYF, paddImage


# test_img_root_directory='G:\My Drive\ProjectGreenAI\Data\\randomImagesTamarind\\images/'

#Read images from repsective 'images' subdirectory
#As all images are of ddifferent sizes resize all to a fixed size
# But, using following code it is possible to crop into a fixed size

filename= 'randomImage2jpeg.jpg'
# filename= 'Capture.jpg'


def getPrediction(img, model1, model2):

    cv2.imwrite('polytesting.png', img)
   
    # test_image_dataset = []   
    # SIZE = 256
    # n_classes=14 #Number of classes for segmentation
    patch_size = 512
#     img_path = 'static/images/'+filename
    
# ##### Read the input image
#     # img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
#     img = cv2.imread(img_path, 1)  
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #NOTE May need to be on
    # plt.imshow(img)
    img_padded=paddImage(img)
    # plt.imshow(img_padded)
    large_img = np.array(img_padded)  

#####  Load the model    
# my_model = load_model("model/unet_bata.hdf5",compile=False)
    # model1 = load_model("model/resnet50_14cls_512size.hdf5",compile=False)
    # model2 = load_model("model/inceptionv3_13cls_512size.hdf5",compile=False)
    opt_weights = [0.2, 0.3]
    
        
############### PRE-PROCESS IMAGES ############################################
    BACKBONE1 = 'resnet50'
    preprocess_input1 = sm.get_preprocessing(BACKBONE1)
    
############### PRE-PROCESS IMAGES ############################################
    BACKBONE2 = 'inceptionv3'
    preprocess_input2 = sm.get_preprocessing(BACKBONE2)

########### Now patchify input image to size 512x512 ##########################
    patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  #Step=512 for 512 patches means no overlap
    patches_img = patches_img[:,:,0,:,:,:]    

    ################## Now apply the model to each patch ##########################
    
    patched_prediction = []
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            
            single_patch_img = patches_img[i,j,:,:,:]
            
            #Use minmaxscaler instead of just dividing by 255. 
            single_patch_img = np.expand_dims(single_patch_img, axis=0)
            
            single_patch_img_processed =preprocess_input1(single_patch_img)
            pred1 = model1.predict(single_patch_img_processed)
            
            single_patch_img_processed =preprocess_input2(single_patch_img)
            pred2 = model2.predict(single_patch_img_processed)
            
            # test_preds=np.array([test_pred1, test_pred2, test_pred3])
            test_preds=np.array([pred1, pred2])
            #Use tensordot to sum the products of all elements over specified axes.
            weighted_test_preds = np.tensordot(test_preds, opt_weights, axes=((0),(0)))
            weighted_ensemble_test_prediction = np.argmax(weighted_test_preds, axis=3)[0,:,:]
            
            # pred = np.argmax(pred, axis=3)
            # pred = pred[0, :,:]
                                     
            patched_prediction.append(weighted_ensemble_test_prediction)
    
    patched_prediction = np.array(patched_prediction)
    
    ################## Now merge the pacthes to a single image#####################
    
    patched_prediction = np.reshape(patched_prediction, [patches_img.shape[0], patches_img.shape[1], 
                                      patches_img.shape[2], patches_img.shape[3]])
    
    segmentedImg = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))
    
    
    ########### Calculate GYF and Print the Result #################################
    pred_GYF=calcGYF(segmentedImg)
    test_img_number=1
    print('Image No.:{}, Predicted GYF:{:.2f} '.format(test_img_number,pred_GYF) )
    
    
    ################ Visualize the segmnetd image and Predicted GYF################
    # fig=plt.figure(figsize=(25,18 ))
    # parameters = {'axes.labelsize':30, 'axes.titlesize': 30, 'xtick.labelsize': 30,
    #               'ytick.labelsize':30,'axes.titlesize':30, 'figure.titlesize':30 }
    # plt.rcParams.update(parameters)
    
    # plt.figure(figsize=(25, 20))
    # plt.subplot(121)
    # plt.title('Test Image')
    # # img=test_img 
    # # scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    # plt.imshow(img_padded)
    # plt.subplot(122)
    # plt.title('Predicted GYF = % 1.2f' % pred_GYF)
    # plt.imshow(label_to_rgb(segmentedImg ))
    # plt.show()
    
    # fig.savefig('static/predicted_images/'+filename,bbox_inches='tight')
###############################################################################
###############################################################################  
    # ### img_path = 'static/predicted_images/'+filename 
    # img_path = 'static/predicted_images/inputImage.jpg' 
    # ### img_path = img_path+".tif"
    # io.imsave(img_path, large_img) 
    
    # ### seg_img_path = 'static/predicted_images/seg_'+filename
    # seg_img_path = 'static/predicted_images/seg_image.jpg'
    # ### seg_img_path=seg_img_path+".tif"
    # io.imsave(seg_img_path, label_to_rgb(segmentedImg)) 


    return pred_GYF


# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 11:18:43 2023

@author: gmati
"""

"""
Partially acknowledgement Author: Sreenivas Bhattiprolu

"""

import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import keras 

# from keras.utils import normalize
from keras.metrics import MeanIoU

def paddImage(original_image):
    # import tensorflow as tf
    # img = np.array(tf.image.resize_with_crop_or_pad( original_image, newH, newW))

    scale=512
    
    width=original_image.shape[0]
    height=original_image.shape[1]
    
    if width>= scale and  height >= scale:
        newW = width+((scale*(np.ceil(width/scale)))-width).astype(dtype=int)
        newH = height+((scale*(np.ceil(height/scale)))-height).astype(dtype=int)
    elif width < scale and  height >= scale:
         newW=scale
         newH = height+((scale*(np.ceil(height/scale)))-height).astype(dtype=int)
    elif width>= scale and  height < scale:
        newW = width+((scale*(np.ceil(width/scale)))-width).astype(dtype=int)
        newH = scale
    else:
        newW = scale
        newH = scale
        
        
    imSize= len(original_image.shape)
    
    if imSize<3:
        color=(0) # check if it shoudl be 16 while albumentation
        channels=1
    else:
        color=(0,0,0) 
        channels=original_image.shape[2]
    
 
    result = np.squeeze(np.full((newW,newH, channels), color, dtype=np.uint8))
    # compute center offset
    x_center = abs(newW - width) // 2
    y_center = abs(newH - height) // 2
# copy img image into center of result image
    result[x_center:x_center+width, y_center:y_center+height] = original_image
    return result

#############################################################################
def label_to_rgb(predicted_image):
    '''    
    New classes = {0:'outside',1:'Road',2:'Building',3:'Bush',4:'cultivArea',5:'grass',6:'newTree', 
           7:'oldTree',8:'pergolla',9: 'playGround',10:'socialArea1',11:'socialArea2',
           12:'socialArea3', 13:'shadow'}
    '''   
    Road = '#BFBF00'.lstrip('#') # Black
    Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228
    
    Building = '#FFFFFF'.lstrip('#') # White
    Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152
    
    Bush = '#0000FF'.lstrip('#') # Blue
    Bush = np.array(tuple(int(Bush[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41

    Grass =  '#00FF00'.lstrip('#') # Lime green
    Grass = np.array(tuple(int(Grass[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58

    newTree = '#649B00'.lstrip('#') # deep green
    newTree = np.array(tuple(int(newTree[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155
    
    oldTree = '#FFFF00'.lstrip('#') # yellow
    oldTree = np.array(tuple(int(oldTree[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155
 
    playGround = '#FF00FF'.lstrip('#') # magenta
    playGround = np.array(tuple(int(playGround[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155
 
    socialArea1= '#A3C5FF'.lstrip('#') # deep sky
    socialArea1 = np.array(tuple(int(socialArea1[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    
    socialArea2= '#A31DF4'.lstrip('#') # violet
    socialArea2 = np.array(tuple(int(socialArea2[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    
    socialArea3= '#800080'.lstrip('#') # purple
    socialArea3 = np.array(tuple(int(socialArea3[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    
    shadow= '#FF0000'.lstrip('#') #RED
    shadow = np.array(tuple(int(shadow[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    
    cultivArea= '#FFF000'.lstrip('#') #RED
    cultivArea = np.array(tuple(int(cultivArea[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    
    pergolla= '#000FFF'.lstrip('#') #RED
    pergolla = np.array(tuple(int(pergolla[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    
    
    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))
    '''
    New classes(after label_merging) = {0:'outside',1:'Road',2:'Building',3:'Bush',4:'cultivArea',5:'grass',6:'newTree', 
           7:'oldTree',8:'pergolla',9: 'playGround',10:'socialArea1',11:'socialArea2',
           12:'socialArea3', 13:'shadow'}
    '''
    # segmented_img[(predicted_image == 0)] = Background
    segmented_img[(predicted_image == 1)] = Road
    segmented_img[(predicted_image == 2)] = Building
    segmented_img[(predicted_image == 3)] = Bush
    segmented_img[(predicted_image == 4)] = cultivArea
    segmented_img[(predicted_image == 5)] = Grass
    segmented_img[(predicted_image == 6)] = newTree
    segmented_img[(predicted_image == 7)] = oldTree
    segmented_img[(predicted_image == 8)] = pergolla 
    segmented_img[(predicted_image == 9)] = playGround 
    segmented_img[(predicted_image == 10)] = socialArea1
    segmented_img[(predicted_image == 11)] = socialArea2
    segmented_img[(predicted_image == 12)] = socialArea3
    segmented_img[(predicted_image == 13)] = shadow
   
    
    segmented_img = segmented_img.astype(np.uint8)
    return(segmented_img)



def label_merging(label):
    """
    legends = {0:'outside',1:'Asphalt',2:'Building',3:'Bush',4:'Car', 5:'cultivArea',6:'cycles',7:'Grass', 
           8:'newTree',9:'oldTree',10: 'Pergola',11:'playGr',12:'socialArea1',13:'socialArea2',
           14:'socialArea3', 15:'shadow'}
    
    New classes = {0:'outside',1:'Road',2:'Building',3:'Bush',4:'cultivArea',5:'grass',6:'newTree', 
           7:'oldTree',8:'pergolla',9: 'playGround',10:'socialArea1',11:'socialArea2',
           12:'socialArea3', 13:'shadow'}
    
    """
    """
    Suply our labale masks as input, Replace pixels with specific label value ...
    """
    # label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg = label
    # label_seg [label == 2] = 1 # merge Building(2) with Road
    label_seg [label == 4] = 1 # merge cars(4) with Road
    label_seg [label == 6] = 1 # merge cycles(6) with Road
    
    # label_seg [label == 3] = 2 # Bush
    label_seg [label == 5] = 4 # cultiVation Land
    label_seg [label == 7] = 5 # grass
    
    label_seg [label == 8] = 6 # newTree
    label_seg [label == 9] = 7 #oldTree
    label_seg [label == 10] = 8 # perGola
        
    label_seg [label == 11] = 9 # playGround
    label_seg [label == 12] = 10 # socialArea#1
    label_seg [label == 13] = 11 # socialArea#2
    
    label_seg [label == 14] = 12 # socialArea3
    label_seg [label == 15] = 13 # shadow
    
    # label_seg [label == 2] = 1 # merge Building(2) with Road
    
    
    # label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg



def labelMergingInStorge(masks_list):
    for img in range(len(masks_list)):   #Using t1_list as all lists are of same size
        # print(img)
        mask_name = masks_list[img]
        print("Lable Merging masks#..: ", mask_name)
          
        temp_mask=cv2.imread(mask_name, 0)
        print('labels before Mering', np.unique(temp_mask))
        temp_mask= label_merging(temp_mask)
        print('labels after Mering', np.unique(temp_mask))
        cv2.imwrite(mask_name[:-4]+".png",temp_mask)
    
    return masks_list
        

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

def scale_RensNet(data):
    scaledImg=[]
    for i in range(0, data.shape[0]):
        img= data[i]
        img=scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        scaledImg.append(img)
    
    return np.array(scaledImg)


def calcClassWiseIoU(label, greenClasses,test_pred,test_mask,df,k,i):
    for label in greenClasses:
        k=k+1
        # print(label)
        classIoU=list()
        segmented_img = np.empty((test_pred.shape[0], test_pred.shape[1]))
        segmented_img2 = np.empty((test_pred.shape[0], test_pred.shape[1]))
        original_mask=test_mask
        predicted_mask=test_pred
        
        segmented_img[(predicted_mask == label)] =1
        segmented_img2[(original_mask == label)] =1
        segmented_img= segmented_img.astype(int).flatten()
        segmented_img2=segmented_img2.astype(int).flatten()
        
        m=tf.keras.metrics.MeanIoU(num_classes=2)
        m.update_state(segmented_img, segmented_img2)
        print("Class#", label, " Mean IoU =", m.result().numpy())
        
        # if m.result().numpy()==1.0:
        #     df.loc[i,df.columns[k]]='nan'
        # else:
        if np.sum(segmented_img2)==0.0 and np.sum(segmented_img)==0.0:
            df.loc[i,df.columns[k]]='nan'
        else:            
            df.loc[i,df.columns[k]]=m.result().numpy()
            
    
    return df
    
def calcClassWiseF1score(label, greenClasses,test_pred,test_mask,df2,k,i):
    from sklearn.metrics import f1_score
    for label in greenClasses:
        k=k+1
        # print(label)
        classIoU=list()
        segmented_img = np.empty((test_pred.shape[0], test_pred.shape[1]))
        segmented_img2 = np.empty((test_pred.shape[0], test_pred.shape[1]))
        original_mask=test_mask
        predicted_mask=test_pred
        
        segmented_img[(predicted_mask == label)] =1
        segmented_img2[(original_mask == label)] =1
        segmented_img= segmented_img.astype(int).flatten()
        segmented_img2=segmented_img2.astype(int).flatten()
        
        y_true=segmented_img2
        y_pred=segmented_img
        f1_val=f1_score(y_true, y_pred, average='macro')
        
        # m=tf.keras.metrics.MeanIoU(num_classes=2)
        # m.update_state(segmented_img, segmented_img2)
        
        print("Class#", label, " Mean F1-score =", f1_val)
        
        # if m.result().numpy()==1.0:
        #     df.loc[i,df.columns[k]]='nan'
        # else:
        if np.sum(segmented_img2)==0.0 and np.sum(segmented_img)==0.0:
            df2.loc[i,df2.columns[k]]='nan'
        else:            
            df2.loc[i,df2.columns[k]]=f1_val
            
    
    return df2


# #Predict on a few images

# import random
# test_img_number = random.randint(0, len(X_test))
# test_img = X_test[test_img_number]
# ground_truth=y_test[test_img_number]
# test_img_norm=test_img[:,:,:]
# test_img_input=np.expand_dims(test_img_norm, 0)

# #Weighted average ensemble
# # models = [model1, model2, model3]
# models = [model1, model2]

# # test_img1=scale_RensNet(test_img_input)
# test_img_input1 = preprocess_input1(test_img_input)
# test_img_input2 = preprocess_input2(test_img_input)
# # test_img_input3 = preprocess_input3(test_img_input)

# test_pred1 = model1.predict(test_img_input1)
# test_pred2 = model2.predict(test_img_input2)
# # test_pred3 = model3.predict(test_img_input3)

# # test_preds=np.array([test_pred1, test_pred2, test_pred3])
# test_preds=np.array([test_pred1, test_pred2])
# #Use tensordot to sum the products of all elements over specified axes.
# weighted_test_preds = np.tensordot(test_preds, opt_weights, axes=((0),(0)))
# weighted_ensemble_test_prediction = np.argmax(weighted_test_preds, axis=3)[0,:,:]

# #####################################################################
# #Convert categorical to integer for visualization and IoU calculation
# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Test Image')
# img=test_img 
# scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
# plt.imshow(img)
# plt.subplot(232)
# plt.title('Ground Truth')
# plt.imshow(label_to_rgb(ground_truth[:,:,0]))
# plt.subplot(233)
# plt.title('Predictions')
# plt.imshow(label_to_rgb(weighted_ensemble_test_prediction))
# plt.show()
# ###############################################################################
# ############ Tree Size Analysis ##############################################
# ##############################################################################

# # plt.imsave('output.jpg', weighted_ensemble_test_prediction, cmap='gray')
# # segmentedImg = cv2.imread('output.jpg')  #Read as color (3 channels)
# # segmentedImg = img[:,:,0]
# # plt.imshow(segmentedImg,cmap='gray')
# ###############################################################################
# ############ Tree Size Analysis ##############################################
# ##############################################################################


def calcGYF(segmentedImg):

    from skimage import measure
    # from skimage import measure, color, io
    
    # segmentedImg= weighted_ensemble_test_prediction
    # Check the class labels and frequency ##############################
    # plt.hist(segmentedImg.flatten())
    
    ############# In our case feed directly the binary tree image  ################
    img_grey = np.zeros(segmentedImg.shape,dtype=np.uint8)
    img_grey[(segmentedImg == 7)] = 1
    ret3, markers = cv2.connectedComponents(img_grey)
    # print(np.unique(markers))
    # plt.imshow(markers,cmap='gray')
    
    props = measure.regionprops_table(markers, intensity_image=img_grey, 
                                  properties=['label','area', 'equivalent_diameter',
                                               'solidity'])
        
    import pandas as pd
    df = pd.DataFrame(props)
    # print(df.head())
    
    # df = df[df.solidity> 0.55]  
    treeBig = df[df.equivalent_diameter >= 90]  #Remove background or other regions that may be counted as objects
    treeMedium = df[(df['equivalent_diameter'] < 90) & (df['equivalent_diameter']> 40)]  #Remove background or other regions that may be counted as objects
    treeSmall = df[df.equivalent_diameter <= 40]  #Remove background or other regions that may be counted as objects
    
    
    ################### calculateGYF #############################################
    ###########Count the number of pixels for each class of interest #############
    
    area_bush=area_grass=area_newTree=area_treeBig=area_treeMedium=0
    area_treeSmall=area_playGr=area_social1=area_social2=area_social3=0
    area_total=area_shadow=area_cultiVland=area_pergola=0
    
    area_treeBig=sum(treeBig['area'])
    area_treeMedium=sum(treeMedium['area'])
    area_treeSmall =sum(treeSmall['area'])
    
    print(area_treeBig,area_treeMedium,area_treeSmall)
    # df = df[df.area > 20]  #Remove background or other regions that may be counted as objects
    # df = df[df.solidity> 0.55]  
    # treeBig = df[df.area > 1000]  #Remove background or other regions that may be counted as objects
    # treeMedium = df[(df['area'] < 1000) & (df['area']> 500)]  #Remove background or other regions that may be counted as objects
    # treeSmall = df[df.area < 500]  #Remove background or other regions that may be counted as objects
      
    ################### calculateGYF #############################################
    ###########Count the number of pixels for each class of interest #############
    
    area_total=segmentedImg.size 
    area_bush=np.sum(segmentedImg == 3)
    area_grass=np.sum(segmentedImg == 5)
    area_newTree=np.sum(segmentedImg == 6)
    # area_oldTree=np.sum(segmentedImg == 7)
    area_playGr=np.sum(segmentedImg == 9)
    area_social1=np.sum(segmentedImg == 10)
    area_social2=np.sum(segmentedImg == 11)
    area_social3=np.sum(segmentedImg == 12)
    area_shadow=int((np.sum(segmentedImg == 13))*0.60) ## Take 60% of pixels in shadow class as grass
    ####
    area_cultiVland=np.sum(segmentedImg == 4)
    area_pergola=np.sum(segmentedImg == 8)
    
    
    
    # predGYF = ((area_bush*0.85)+(area_grass*1.5)+(area_newTree*1.63)+(pred_oldTree*1.55)+(pred_social*1.2))/(pred_whole_area-200)
    
    ## Average valur for Bush can be 0.55 or 0.85
    area_ecoSurface = ((area_bush*0.55)+(area_grass*1.5)+(area_newTree*1.63)
                       +(area_treeBig*1.8) +(area_treeMedium*1.55)+(area_treeSmall*1.3)
                       +(area_playGr*1.2)+(area_social1*1.2)+(area_social2*1.2)
                       +(area_social3*1.2)+(area_shadow*1.5) 
                        + (area_cultiVland*1.5)+ (area_pergola*0.3))
    
    pred_GYF= area_ecoSurface /area_total
    
    # print('Image No.:{}, Predicted GYF:{:.2f} '.format(test_img_number,pred_GYF) )
    
    # print('Image No.:{}, Predicted GYF:{:.2f} '.format(i+1,pred_GYF) )
    
    # plt.figure(figsize=(12, 8))
    # plt.subplot(231)
    # plt.title('Test Image')
    # img=test_img 
    # scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    # plt.imshow(img)
    # plt.subplot(232)
    # plt.title('Ground Truth')
    # plt.imshow(label_to_rgb(ground_truth[:,:,0]))
    # plt.subplot(233)
    # plt.title('Predicted GYF = % 1.2f' % pred_GYF)
    # plt.imshow(label_to_rgb(weighted_ensemble_test_prediction))
    # plt.show()
    
    return pred_GYF


#####################
# from skimage import measure, color, io

##### Apply Watershed Alogorithm for determining tree size 
# img = cv2.imread('data/results/segm.jpg')  #Read as color (3 channels)
# img_grey = np.zeros(segmentedImg.shape,dtype=np.uint8)
# img_grey[(segmentedImg == 7)] = 1
# plt.imshow(img_grey,cmap='gray')


# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(img_grey,cv2.MORPH_OPEN,kernel, iterations = 2)
# plt.imshow(opening,cmap='gray')


# sure_bg = cv2.dilate(opening,kernel,iterations=10)
# plt.imshow(sure_bg ,cmap='gray')

# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# plt.imshow(dist_transform,cmap='gray')

# ret2, sure_fg = cv2.threshold(dist_transform, 0.15*dist_transform.max(),255,0)
# plt.imshow(sure_fg ,cmap='gray')


# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
# print(np.unique(unknown))
# plt.imshow(unknown,cmap='gray')


# ret3, markers = cv2.connectedComponents(sure_fg)
# plt.imshow(markers,cmap='gray')
# print(np.unique(markers))

# markers = markers+10
# print(np.unique(markers))

# plt.imshow(unknown,cmap='gray')
# markers[unknown==1] = 0
# print(np.unique(markers))
# plt.imshow(markers,cmap='gray')

# markers = cv2.watershed(img, markers)
# print(np.unique(markers))
# plt.imshow(markers,cmap='gray')

# img[markers == -1] = [255,0,0]  
# plt.imshow(img)
# plt.imsave('output_treeMarked.jpg', img)

# img2 = color.label2rgb(markers, bg_label=0)
# plt.imshow(img2)
# plt.imsave('output_treeMarked2.jpg', img)

# # cv2.imshow('Overlay on original image', img)
# # cv2.imshow('Colored Grains',img2)
# # cv2.waitKey(0)












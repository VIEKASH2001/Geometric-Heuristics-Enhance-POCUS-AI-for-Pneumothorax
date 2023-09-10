import os
import cv2
import numpy as np
import imutils
## 6 WAY SEGMENTATIONS

# 0 bck-grd
# 1 pl-pneumo
# 2 pl-healthy
# 3 vessel
# 4 chest-wall
# 5 rib
                            
## 4 WAY SEGMENTATIONS

# 0 bck-grd
# 1 pluera
# 2 chest-wall
# 3 rib

filepath = "data1\datasets\DARPA-Dataset"

# savepath = "/home/gautam/datasets/Cropped_DARPA_Dataset/"
for file in os.listdir(filepath):
    rootdir = os.path.join(filepath, file)
    savedir = os.path.join(filepath, file)
    
    if os.path.isdir(rootdir):
        for file1 in os.listdir(rootdir): 
            d1 = os.path.join(rootdir, file1)
            save_d1 = os.path.join(savedir, file1)
            
            if os.path.isdir(d1):
                for file2 in os.listdir(d1):
                    if file2 == "crop_seg_lb_rct": # Not all clips have segmentations
                        d2 = os.path.join(d1, file2)
                        save_d2 = os.path.join(save_d1, file2)
                        chk = os.path.join(d1, "crop_image_rct")
                        save_chk = os.path.join(save_d1, "crop_image_rct")
                        
                        if len(os.listdir(d2)) == len(os.listdir(chk)): # Not all frames of a given clip having segmentations have segmentations
                                
                                x_d = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []} #assuming a max of 11 contours
                                y_d = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}
                                w_d = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}
                                h_d = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}
                                for file3 in os.listdir(d2):
                                    d3 = os.path.join(d2, file3)
                                    chk1 = os.path.join(chk, file3)
                                    save_d3 = os.path.join(save_d2, file3)
                                    save_chk1 = os.path.join(save_chk, file3)
                                    org = cv2.imread(chk1,0)
                                    img = cv2.imread(d3,0)
                                    
                                    head_tail = os.path.splitext(file3)
                                    frame_d3 = save_d2+"/"+head_tail[0]
                                    frame_chk1 = save_chk+"/"+head_tail[0]
                                    # os.makedirs(frame_d3)
                                    # os.makedirs(frame_chk1)

                                    map = img.copy()
                                    # if 1 in np.unique(img): # Check for consistency 
                                    #     # Sliding - no pneumothorax
                                    #     # No Sliding - has pneumothorax
                                    #     if file == 'Sliding':
                                    #         print(file1)

                                    img[img == 1] = 255
                                    img[img == 2] = 255
                                    img[img == 3] = 0
                                    img[img == 4] = 0
                                    img[img == 5] = 0   

                                    # contour
                                    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    cnts = imutils.grab_contours(cnts)
                                    i = 0 # count of the number of contours/pleura

                                    for j in cnts:
                                        i+=1
                                        rect = cv2.boundingRect(j)
                                        #print(cv2.boxPoints(rect))
                                        x,y,w,h = rect
                                        
                                        x_d[i].append(x)
                                        y_d[i].append(y)
                                        w_d[i].append(w)
                                        h_d[i].append(h)


                                # The above function is to find the min x,y coordinate and max h and w dimension suchthat the crop can contain all the pleura

                                # The below function is to crop and save them
                                for file3 in os.listdir(d2):
                                    d3 = os.path.join(d2, file3)
                                    chk1 = os.path.join(chk, file3)
                                    save_d3 = os.path.join(save_d2, file3)
                                    save_chk1 = os.path.join(save_chk, file3)
                                    org = cv2.imread(chk1,0)
                                    img = cv2.imread(d3,0)
                                    
                                    head_tail = os.path.splitext(file3)
                                    frame_d3 = save_d2+"/"+head_tail[0]
                                    frame_chk1 = save_chk+"/"+head_tail[0]
                                    # os.makedirs(frame_d3)
                                    # os.makedirs(frame_chk1)

                                    map = img.copy()
                                    
                                    img[img == 1] = 255
                                    img[img == 2] = 255
                                    img[img == 3] = 0
                                    img[img == 4] = 0
                                    img[img == 5] = 0              
                                    # contour
                                    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    cnts = imutils.grab_contours(cnts)
                                    i = 0 # count of the number of contours/pleura
                                    
                                    for j in cnts:
                                        i+=1
                                        x = min(x_d[i]) 
                                        y = min(y_d[i]) 
                                        w = max(w_d[i]) 
                                        h = max(h_d[i]) 
                                        f = 5 #factor of cropping
                                        
                                        #cv2.rectangle(over, (x,y-f),(x+w,y+h+f),(255,0,255),2) 
                                        
                                        crop_1 = img[y-f:y+h+f, x:x+w]  # having the pixels of just the pleura and not the rest segments
                                        crop_2 = org[y-f:y+h+f, x:x+w]  
                                        print(crop_1.shape, img.shape)
                                        
                                        #normalize the images to scale them the same way between 0 and 1
                                        
                                        crop_1 = cv2.normalize(crop_1, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                                        crop_2 = cv2.normalize(crop_2, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                                        

                                        #resize
                                        desired_size = 224
                                        crop_1 = cv2.resize(crop_1, (desired_size, desired_size), interpolation = cv2.INTER_NEAREST)
                                        crop_2 = cv2.resize(crop_2, (img.shape[1], desired_size), interpolation = cv2.INTER_CUBIC) # upscale the height first
                                        crop_2 = cv2.resize(crop_2, (desired_size, desired_size), interpolation = cv2.INTER_AREA) # shrink the width next

                                        # # resize the original image by shrinking its width using cv2.INTER_AREA interpolation
                                        # img_shrunk = cv2.resize(img, (desired_size, img.shape[0]), interpolation=cv2.INTER_AREA)

                                        # # resize the shrunk image by upscaling its height using cv2.INTER_CUBIC interpolation
                                        # img_resized = cv2.resize(img_shrunk, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

                                        cv2.imwrite("save/org.jpg", org)
                                        if(i == 1):
                                            cv2.imwrite("save/crop1.jpg", crop_1*255)
                                            cv2.imwrite("save/crop2.jpg", crop_2*255)
                                            
                                        # this is the input data to the network]
                                        data = np.stack((crop_2, crop_1))
                                        cv2.waitKey(100)
                                        # cv2.imwrite(frame_chk1+"/"+str(i)+head_tail[1],data[0]*255)#org
                                        # cv2.imwrite(frame_d3+"/"+str(i)+head_tail[1],data[1]*255)#seg
                      

# rootdir = "/home/gautam/datasets/DARPA-Dataset/Sliding"
# for file1 in os.listdir(rootdir):
#     d1 = os.path.join(rootdir, file1)
#     if os.path.isdir(d1):
#         for file2 in os.listdir(d1):
#             if file2 == "crop_seg_lb_rct":
#                  d2 = os.path.join(d1, file2)
#                  chk = os.path.join(d1, "crop_image_rct")
#                  if len(os.listdir(d2)) == len(os.listdir(chk)): # Not all frames of a given clip having segmentations have segmentations
#                         for file3 in os.listdir(d2):
#                             d3 = os.path.join(d2, file3)
#                             chk1 = os.path.join(chk, file3)
#                             org = cv2.imread(chk1,0)
#                             img = cv2.imread(d3,0)
#                             map = img.copy()
#                             img[img == 1] = 255
#                             img[img == 2] = 255
#                             img[img == 3] = 0
#                             img[img == 4] = 0
#                             img[img == 5] = 0              
#                             # contour
#                             cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                             cnts = imutils.grab_contours(cnts)
#                             i = 0
#                             for j in cnts:
#                                 i+=1
#                                 rect = cv2.boundingRect(j)
#                                 #print(cv2.boxPoints(rect))
#                                 x,y,w,h = rect
#                                 f = 35 #factor of cropping
#                                 m = 4 # to increase the depth of the image in y axis
#                                 if(y>f):
#                                     #cv2.rectangle(over, (x,y-f),(x+w,y+h+f),(255,0,255),2) 
#                                     crop_1 = map[y-f:y+h+f*m, x:x+w]  
#                                     crop_2 = org[y-f:y+h+f*m, x:x+w]  
#                                 else: 
#                                     #cv2.rectangle(over, (x,0),(x+w,y+h+f),(255,0,255),2) 
#                                     crop_1 = map[0:y+h+f*m, x:x+w]  
#                                     crop_2 = org[0:y+h+f*m, x:x+w] 
#                                 #normalize the images to scale them the same way between 0 and 1
#                                 crop_1 = cv2.normalize(crop_1, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#                                 crop_2 = cv2.normalize(crop_2, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#                                 #resize
#                                 desired_size = 512
#                                 crop_1 = cv2.resize(crop_1, (desired_size, desired_size))
#                                 crop_2 = cv2.resize(crop_2, (desired_size, desired_size))
#                                 # this is the input data to the network]
#                                 data = np.stack((crop_2, crop_1))

# rootdir = "DARPA_Dataset_v1/NoSliding"
# c = 0
# for file1 in os.listdir(rootdir): 
#     d1 = os.path.join(rootdir, file1)
#     if os.path.isdir(d1):
#         for file2 in os.listdir(d1):
#             if file2 == "crop_seg_lb_rct": # Not all clips have segmentations
#                  d2 = os.path.join(d1, file2)
#                  chk = os.path.join(d1, "crop_image_rct")
#                  if len(os.listdir(d2)) == len(os.listdir(chk)): # Not all frames of a given clip having segmentations have segmentations
#                         print(d2)
#                         #for file3 in os.listdir(d2):
#                             #d3 = os.path.join(d2, file3)
#                             #print(d3)
#                             #img = cv2.imread(d3)
#                             #if(c < len(np.unique(img))):
#                                 #c = len(np.unique(img))
#                             #if 3 in np.unique(img):
#                                 #img = img*30
#                                 #mask = np.all(img == [0, 0, 0], axis=-1)
#                                 #img[mask] = 255
#                                 #print(np.unique(img))
#                                 #cv2.imwrite("save.png",img)
#                                 #break
# #print(c)

# rootdir = "DARPA_Dataset_v1/Sliding"
# c = 0
# for file1 in os.listdir(rootdir): 
#     d1 = os.path.join(rootdir, file1)
#     if os.path.isdir(d1):
#         for file2 in os.listdir(d1):
#             if file2 == "crop_seg_lb_rct": # Not all clips have segmentations
#                  d2 = os.path.join(d1, file2)
#                  chk = os.path.join(d1, "crop_image_rct")
#                  if len(os.listdir(d2)) == len(os.listdir(chk)): # Not all frames of a given clip having segmentations have segmentations
#                         print(d2)
#                         #for file3 in os.listdir(d2):
#                             #d3 = os.path.join(d2, file3)
#                             #print(d3)
#                             #img = cv2.imread(d3)
#                             #if(c < len(np.unique(img))):
#                              #   c = len(np.unique(img))
#                             #if 3 in np.unique(img):
#                                 #img = img*30
#                                 #mask = np.all(img == [0, 0, 0], axis=-1)
#                                 #img[mask] = 255
#                                 #print(np.unique(img))
#                                 #cv2.imwrite("save.png",img)
#                                 #break
# #print(c)
#!/usr/bin/env python
# coding: utf-8

# In[70]:


#Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# In[71]:


#Pegando dado da trajetoria real do objeto
poses = pd.read_csv('C:/Users/maely/Downloads/data_odometry_poses/dataset/poses/01.txt', delimiter=' ', header=None)
gt = np.zeros((len(poses), 3, 4))
for i in range(len(poses)):
    gt[i] = np.array(poses.iloc[i]).reshape((3, 4))
    
gt[1].dot(np.array([0,0,0,1]))


# In[72]:


#Mostrando Trajetoria real
get_ipython().run_line_magic('matplotlib', 'notebook')
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(gt[:, :, 3][:, 0], gt[:, :, 3][:, 1], gt[:, :, 3][:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=-40, azim=270)


# In[73]:


# #Mostrando primeiras imagens do banco de dados (sequencia 01)
# test_img = cv2.imread('C:/Users/maely/Downloads/data_odometry_gray/dataset/sequences/01/image_0/000002.png')
# test_img2 = cv2.imread('C:/Users/maely/Downloads/data_odometry_gray/dataset/sequences/01/image_0/000003.png')
# test_img3 = cv2.imread('C:/Users/maely/Downloads/data_odometry_gray/dataset/sequences/01/image_0/000004.png')
# test_img4 = cv2.imread('C:/Users/maely/Downloads/data_odometry_gray/dataset/sequences/01/image_0/000005.png')
# %matplotlib inline
# plt.figure(figsize=(12,6))
# plt.imshow(test_img)
# plt.figure(figsize=(12,6))
# plt.imshow(test_img2)
# plt.figure(figsize=(12,6))
# plt.imshow(test_img3)
# plt.figure(figsize=(12,6))
# plt.imshow(test_img4)


# In[74]:


# #Harris Corners (imagens 2 ao 5)

# thresh = 100
# blockSize = 2
# apertureSize = 3
# a = 0.04

# img1 = cv2.imread('C:/Users/maely/Downloads/data_odometry_gray/dataset/sequences/01/image_0/000002.png')
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #deixando em escala cinza

# img2 = cv2.imread('C:/Users/maely/Downloads/data_odometry_gray/dataset/sequences/01/image_0/000003.png')
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #deixando em escala cinza

# img3 = cv2.imread('C:/Users/maely/Downloads/data_odometry_gray/dataset/sequences/01/image_0/000004.png')
# img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY) #deixando em escala cinza

# img4 = cv2.imread('C:/Users/maely/Downloads/data_odometry_gray/dataset/sequences/01/image_0/000005.png')
# img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY) #deixando em escala cinza
# dst_boxes1 = []
# dst_boxes2 = []
# dst_boxes3 = []
# dst_boxes4 = []
# # Achando corners
# dst = cv2.cornerHarris(img1, blockSize, apertureSize, a)
# dst2 = cv2.cornerHarris(img2, blockSize, apertureSize, a)
# dst3 = cv2.cornerHarris(img3, blockSize, apertureSize, a)
# dst4 = cv2.cornerHarris(img4, blockSize, apertureSize, a)

# color = (255, 0, 0)
# # Normalizando
def norm(dst,dst_boxes):
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)

    # Desenhando circulo em volta dos corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                dst_boxes.append((j-1,i-1, j+1,i+1))
                cv2.rectangle(dst_norm_scaled, (j-1,i-1), (j+1,i+1), color, 2)
               
    return dst_norm_scaled
  

# # resultados
# imagNorm1 = norm(dst,dst_boxes1)
# imagNorm2 = norm(dst2,dst_boxes2)
# imagNorm3 = norm(dst3,dst_boxes3)
# imagNorm4 = norm(dst4,dst_boxes4)
# %matplotlib inline
# plt.figure(figsize=(12,6))
# plt.imshow(imagNorm1)

# plt.figure(figsize=(12,6))
# plt.imshow(imagNorm2)

# plt.figure(figsize=(12,6))
# plt.imshow(imagNorm3)

# plt.figure(figsize=(12,6))
# plt.imshow(imagNorm4)
    


# In[75]:


#Filtrando (Usando non-max suppression)

#5X5

#5000 para 10X10 

def non_max(boxes, overlapThresh):

   
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


# In[76]:



# dst = cv2.cornerHarris(img1, blockSize, apertureSize, a)
# dst2 = cv2.cornerHarris(img2, blockSize, apertureSize, a)
# dst3 = cv2.cornerHarris(img3, blockSize, apertureSize, a)
# dst4 = cv2.cornerHarris(img4, blockSize, apertureSize, a)


def norm2(dst):
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
              
    return dst_norm_scaled
  
# imag1Filter = norm2(dst)
# imag2Filter = norm2(dst2)
# imag3Filter = norm2(dst3)
# imag4Filter = norm2(dst4)

# pick1 = non_max(np.array(dst_boxes1), 0.3)
# pick2 = non_max(np.array(dst_boxes2), 0.3)
# pick3 = non_max(np.array(dst_boxes3), 0.3)
# pick4 = non_max(np.array(dst_boxes4), 0.3)



def pickBoxes(pick, imagFilter):
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(imagFilter, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return imagFilter

# imag1Filter = pickBoxes(pick1, imag1Filter)
# imag2Filter = pickBoxes(pick2, imag2Filter)
# imag3Filter = pickBoxes(pick3, imag3Filter)
# imag4Filter = pickBoxes(pick4, imag4Filter)

# %matplotlib inline
# plt.figure(figsize=(12,6))
# plt.imshow(imag1Filter)

# plt.figure(figsize=(12,6))
# plt.imshow(imag2Filter)

# plt.figure(figsize=(12,6))
# plt.imshow(imag3Filter)

# plt.figure(figsize=(12,6))
# plt.imshow(imag4Filter)


# In[ ]:





# In[81]:


get_ipython().run_line_magic('matplotlib', 'inline')
for i in range(100,150):
    test_img = cv2.imread('C:/Users/maely/Downloads/data_odometry_gray/dataset/sequences/01/image_0/000'+str(i)+'.png')
    plt.figure(figsize=(12,6))
    plt.imshow(test_img)


# In[82]:


thresh = 100
blockSize = 2
apertureSize = 3
a = 0.04


for i in range(100,150):
    img = cv2.imread('C:/Users/maely/Downloads/data_odometry_gray/dataset/sequences/01/image_0/000'+str(i)+'.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    dst = cv2.cornerHarris(img, blockSize, apertureSize, a)
    dst_boxes=[]
    imagNorm = norm(dst,dst_boxes)
    plt.figure(figsize=(12,6))
    plt.imshow(imagNorm)


# In[89]:


for i in range(100,150):
    img = cv2.imread('C:/Users/maely/Downloads/data_odometry_gray/dataset/sequences/01/image_0/000'+str(i)+'.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    dst = cv2.cornerHarris(img, blockSize, apertureSize, a)
    imagFilter = norm2(dst)
    pick = non_max(np.array(dst_boxes), 0.3)
    imagFilter = pickBoxes(pick, imagFilter)
    plt.figure(figsize=(12,6))
    plt.imshow(imagFilter)


# In[77]:


# # Se tivermos mais de 100 features em cada bucket => QuickSelect

# def partition(arr, l, r):

#     x = arr[r]
#     i = l
#     for j in range(l, r):

#         if arr[j] <= x:
#             arr[i], arr[j] = arr[j], arr[i]
#             i += 1

#     arr[i], arr[r] = arr[r], arr[i]
#     return i

# def kthSmallest(arr, l, r, k):


#     if (k > 0 and k <= r - l + 1):


#     index = partition(arr, l, r)


#     if (index - l == k - 1):
#         return arr[index]


#         if (index - l > k - 1):
#             return kthSmallest(arr, l, index - 1, k)


#         return kthSmallest(arr, index + 1, r,
#                             k - index + l - 1)
#     print("Index out of bound")

# # Driver Code
# arr = [ 10, 4, 5, 8, 6, 11, 26 ]
# n = len(arr)
# k = 3
# print("K-th smallest element is ", end = "")
# print(kthSmallest(arr, 0, n - 1, k))



# In[78]:


# def match_features(des1, des2):
    
#     matches = matcher.knnMatch(des1, des2, k=2)
#     matches = sorted(matches, key=lambda x: x[0].distance)
        
#     return matches


# In[126]:





# In[161]:





# In[129]:





# In[130]:





# In[131]:





# In[132]:





# In[ ]:





# In[ ]:





# In[159]:





# In[ ]:





# In[ ]:





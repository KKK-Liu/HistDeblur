import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import time
from PIL import Image
from queue import Queue


def blur_one_image_BGR(ori_img):
    '''
        generate the mapping relation from pixel to kernel
    '''
    img = np.array(ori_img)

        
    kernel_mapping = np.random.randint(low = 0, high = 70, size=(224,224))
    kernel_mapping = np.array(kernel_mapping, dtype=np.uint8)
    
    kernel_mapping_smoothed = kernel_mapping.copy()
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping, sigmaX=10, ksize=(0,0))
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping_smoothed, sigmaX=8, ksize=(0,0))
    
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.int8)
    
    kernel_mapping_smoothed = kernel_mapping_smoothed - np.min(kernel_mapping_smoothed)
    
    tmp_m = 48 - np.max(kernel_mapping_smoothed)
    
    for i in range(tmp_m):
        kernel_mapping_smoothed = kernel_mapping_smoothed + np.random.choice([0,1],p=[0.6,0.4])

    kernel_mapping_smoothed = np.dstack([kernel_mapping_smoothed, kernel_mapping_smoothed, kernel_mapping_smoothed])
    
    result_image = np.zeros(img.shape, dtype=np.uint8)
    
    for i, name in zip(range(3), ['B', 'G', 'R']):
        z_min = np.min(kernel_mapping_smoothed[:,:,i])
        z_max = np.max(kernel_mapping_smoothed[:,:,i])
        
        for j in range(z_min, z_max + 1):

            kernel = np.load('./plots_npy/color-real-3um/{}-{:0>2}.npy'.format(name, j))
            
            kernel = kernel/np.sum(kernel)
            kernel = cv2.Mat(kernel*255)
            kernel = cv2.resize(kernel,(21,21))
            kernel = kernel/np.sum(kernel)
            
            blurred_one_channel = cv2.filter2D(img[:,:,i],ddepth=-1, kernel=kernel)
            blurred_one_channel = np.array(blurred_one_channel)
            
            result_image[kernel_mapping_smoothed[:,:,i]==j, i] = blurred_one_channel[kernel_mapping_smoothed[:,:,i]==j]
    
    result_image = np.array(result_image, dtype=np.uint8)
    

    
    degree = 4
    for i in range(8):
        degree = degree + np.random.choice([0,1],p=[0.5,0.5])
        
    result_image = motion_blur_copy(result_image, degree, 0)
    
    return result_image

def motion_blur_copy(image, degree=24, angle=30):
    image = np.array(image)

    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle + 45, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    
    # print(motion_blur_kernel)
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def blur_dataset_and_load(image_names, name, generated_dataset_name, data_root):
        root_clear = os.path.join(generated_dataset_name, '{}-clear'.format(name))
        root_blurred = os.path.join(generated_dataset_name, '{}-blurred'.format(name))
        
        os.makedirs(root_clear, exist_ok=True)
        os.makedirs(root_blurred, exist_ok=True)
            
        for image_name in image_names:
            img = cv2.imread(os.path.join(data_root, image_name))
            blured_img = blur_one_image_BGR(img)
            
            cv2.imwrite(os.path.join(root_clear, image_name), img)
            cv2.imwrite(os.path.join(root_blurred, image_name), blured_img)

            
def make_dataset_with_random_spf_advanced_multi_real():
    from multiprocessing import Process

    TRAIN_IMAGE_NUMBER = 1024
    VALIDATION_IMAGE_NUMBER = 256

    data_root = './data/CRC-224/raw'
    image_names = os.listdir(data_root)
    random.shuffle(image_names)
    
    image_names_train = image_names[:TRAIN_IMAGE_NUMBER]
    image_names_val = image_names[TRAIN_IMAGE_NUMBER:TRAIN_IMAGE_NUMBER+VALIDATION_IMAGE_NUMBER]
    
    generated_dataset_name = os.path.join('./data/CRC-224', 'CRC-'+time.strftime('%m-%d-%H-%M', time.localtime()))
    

    ts = [Process(target=blur_dataset_and_load, args=(image_names_train[256*i:256*(i+1)], 'train', generated_dataset_name, data_root, )) for i in range(4)] +\
        [Process(target=blur_dataset_and_load, args=(image_names_val, 'val', generated_dataset_name, data_root, ))]
    
    [t.start() for t in ts]
        
    [t.join() for t in ts]
    print('Dataset generation finished. Dataset Name:{}'.format(generated_dataset_name))
    
    
def sharp_height_generation(show=False):
    kernel_mapping = np.random.randint(low = 0, high = 70, size=(224,224))
    kernel_mapping = np.array(kernel_mapping, dtype=np.uint8)
    
    kernel_mapping_smoothed = kernel_mapping.copy()
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping, sigmaX=10, ksize=(0,0))
    kernel_mapping_smoothed = cv2.GaussianBlur(kernel_mapping_smoothed, sigmaX=13, ksize=(0,0))
    
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.uint8)
    
    kernel_mapping_smoothed = kernel_mapping_smoothed - np.min(kernel_mapping_smoothed)
    
    
    tmp_m = 48 - int(np.max(kernel_mapping_smoothed))
    v1 = np.random.randint(low=24,high=48)
    v2 = np.random.randint(low=0,high=int((48-v1)/2)+1)
    kernel_mapping_smoothed = kernel_mapping_smoothed * int(v1/np.max(kernel_mapping_smoothed)) + v2
    kernel_mapping_smoothed = np.array(kernel_mapping_smoothed, dtype=np.uint8)

    if show:
        plt.imshow(kernel_mapping_smoothed, vmin=0,vmax=1)
        plt.show()
    return kernel_mapping_smoothed

    
def blur_with_height_known(img, height_map):
    result_image = np.zeros(img.shape, dtype=np.uint8)
    height_map = np.dstack([height_map for i in range(3)])
    for i, name in zip(range(3), ['B', 'G', 'R']):
        z_min = np.min(height_map[:,:,i])
        z_max = np.max(height_map[:,:,i])
        
        for j in range(z_min, z_max + 1):

            kernel = np.load('./plots_npy/color-real-3um/{}-{:0>2}.npy'.format(name, j))
            
            kernel = kernel/np.sum(kernel)
            kernel = cv2.Mat(kernel*255)
            kernel = cv2.resize(kernel,(21,21))
            kernel = kernel/np.sum(kernel)
            
            blurred_one_channel = cv2.filter2D(img[:,:,i],ddepth=-1, kernel=kernel)
            blurred_one_channel = np.array(blurred_one_channel)
            
            result_image[height_map[:,:,i]==j, i] = blurred_one_channel[height_map[:,:,i]==j]
    
    result_image = np.array(result_image, dtype=np.uint8)
    
    return result_image
    

def blur_img(img):
    h = sharp_height_generation()
    return blur_with_height_known(img, h), h

def new_blur_dataset_generation():
    sharp_imgs_root = r'D:\desktop\de-OOF\data\CRC-224\CRC-03-05-21-40\val-clear'
    save_root = r'D:\desktop\de-OOF\data\later_dataset\1'
    
    img_save_root = os.path.join(save_root,'image')
    kernel_save_root = os.path.join(save_root, 'ZMap')
    
    os.makedirs(img_save_root, exist_ok=True)
    os.makedirs(kernel_save_root, exist_ok=True)
    sharp_img_names = os.listdir(sharp_imgs_root)
    
    for sharp_img_name in sharp_img_names:
        sharp_img_path = os.path.join(sharp_imgs_root, sharp_img_name)
        
        sharp_img = cv2.imread(sharp_img_path)
        blurry_img, height_map =blur_img(sharp_img)
        
        cv2.imwrite(os.path.join(img_save_root, sharp_img_name), blurry_img)
        cv2.imwrite(os.path.join(kernel_save_root, sharp_img_name), height_map*5)
        
        
    
    
        
        
if __name__ == '__main__':
    # make_dataset_with_random_spf_advanced_multi_real()
    # new_blur_dataset_generation()
    ...

    
    
    
import cv2
import numpy as np
import os
import albumentations as A

transforms = A.Compose([
                    A.OneOf([
                              A.CLAHE(clip_limit=2.5, tile_grid_size=(4, 4)),
                              A.IAASharpen(),
                              A.IAAEmboss(),
                              A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2),
                          ], p=0.3),
                    A.OneOf([
                              A.MotionBlur(p=0.2),
                              A.MedianBlur(blur_limit=3, p=0.1),
                              A.Blur(blur_limit=3, p=0.1),
                          ], p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, interpolation=cv2.INTER_AREA, border_mode=cv2.BORDER_WRAP, p=0.5)
])

def preprocess(img):
    dilated_img = cv2.dilate(img.copy(), np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)  #using median blur to remove the undesired shadow along with abs difference and normalization
    diff_img = 255 - cv2.absdiff(img.copy(), bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    norm_img = norm_img.astype("uint8")
    ret, thresh1 = cv2.threshold(norm_img, 90, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel = np.ones((2,2),np.uint8)

    erosion = cv2.erode(thresh1,np.ones((1,1),np.uint8),iterations = 1)
    dilation = cv2.erode(erosion,kernel,iterations = 1)
    
    dilation = cv2.copyMakeBorder(dilation, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = 255)

    return dilation

imgs = os.listdir('/home/mainak/Documents/Robotics/Mosiac/trains/main_imgs')
print(imgs)
imgs = sorted(imgs, key= lambda x: int(x[:2]))
print(imgs)
base_path = '/home/mainak/Documents/Robotics/Mosiac/trains/main_imgs/'
for im in range(len(imgs)):

    for data in os.walk(base_path+imgs[im]+'/'):
        os.makedirs('/home/mainak/Documents/Robotics/Mosiac/trains/'+imgs[im])
        os.chdir('/home/mainak/Documents/Robotics/Mosiac/trains/'+imgs[im])
        for j in range(len(data[2])):
            print(data[2][j])
            image = cv2.imread(data[0]+data[2][j])
            image = cv2.resize(image, (1000, 1000))
            print(image.shape)

            for i in range(20):
                r = cv2.selectROI(image, showCrosshair=False, fromCenter=False)
                imcrop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
                imcrop = cv2.resize(imcrop, (64, 64))
                imcrop = cv2.cvtColor(imcrop, cv2.COLOR_BGR2GRAY)
                _, imcrop = cv2.threshold(imcrop.copy(), 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = np.ones((2,2),np.uint8)

                imcrop = cv2.erode(imcrop,kernel,iterations = 1)
                cv2.imshow("Image", imcrop)
                cv2.waitKey(0)
                imcrop1 = transforms(image= imcrop)['image']
                imcrop2 = transforms(image = imcrop)['image']
                

                cv2.imwrite(str(3*i+0+j*60)+'.jpg', imcrop)
                cv2.imwrite(str(3*i+1+j*60)+'.jpg', imcrop1)
                cv2.imwrite(str(3*i+2+j*60)+'.jpg', imcrop2)
'''for img in imgs:
    print(img)
    image = cv2.imread(base_path+img)
    print(image.shape)
    os.makedirs('/home/mainak/Documents/Robotics/Mosiac/tests/'+img[:2])
    os.chdir('/home/mainak/Documents/Robotics/Mosiac/tests/'+img[:2])
    for i in range(5):
        r = cv2.selectROI(image, showCrosshair=False, fromCenter=False)
        imcrop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        imcrop = cv2.resize(imcrop, (64, 64))
        imcrop = cv2.cvtColor(imcrop, cv2.COLOR_BGR2GRAY)
        _, imcrop = cv2.threshold(imcrop.copy(), 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        imcrop1 = transforms(image= imcrop)['image']
        imcrop2 = transforms(image = imcrop)['image']
        

        cv2.imwrite(str(3*i+0)+'.jpg', imcrop)
        cv2.imwrite(str(3*i+1)+'.jpg', imcrop1)
        cv2.imwrite(str(3*i+2)+'.jpg', imcrop2)'''



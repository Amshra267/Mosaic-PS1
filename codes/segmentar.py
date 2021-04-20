import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

num_cycles = 0   # num_of_times_the_loop_must run_then_stops while detecting contour recursively if not found
def skew_correction(img, before = False):
    """
    this function corrects the input image skewness wrt horizontal and aligns it well
    img - for binary_image
    """
  #  cv2.imshow("masked",img )
    t_ero = cv2.dilate(np.uint8(np.logical_not(img.copy()))*255, kernel = np.ones((2,2), np.uint8), iterations = 1)
    contours, _ = cv2.findContours(t_ero, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
   # print(len(contours))
    # Find largest contour and surround in min area box
    largestContour = contours[0]
    rect = cv2.minAreaRect(largestContour)

    angle = rect[-1]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(t_ero, [box], 0, (255,255,255), 3)
  #  cv2.imshow("t_Ero", t_ero)
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
   # print(angle)
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle
    # rotate the image to deskew it
    #print(angle)
    #print("angle- ", angle)
    if (angle < 10 and angle>-10) and before==True:
        return img

    (h, w) = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def h_projection(img):
    """
    Calculate the horizontal projected pixeled image
    """
    return np.sum(img, axis=1)

def v_projection(img):
    """
    Calculate the vertical projected pixeled image
    """
    return np.sum(img, axis=0)

def thresh_after_resize(img):
    H, W = img.shape
    w = 400
    h = int(H*w/W)
    img = cv2.resize(img, (w,h))
    
    #print(img.shape)
    #############--------Preprocessing and cleaning along with shadow removal-----------------##
    """--------------------INNOVATION- Robust to good amount of blurness and shadow present in input image---------"""
    dilated_img = cv2.dilate(img.copy(), np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)  #using median blur to remove the undesired shadow along with abs difference and normalization
    diff_img = 255 - cv2.absdiff(img.copy(), bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    ret, thresh1 = cv2.threshold(norm_img, 90, 255, cv2.THRESH_BINARY + 
                                                cv2.THRESH_OTSU)
    
    kernel = np.ones((2,2),np.uint8)

    erosion = cv2.erode(thresh1,np.ones((1,1),np.uint8),iterations = 1)
    dilation = cv2.erode(erosion,kernel,iterations = 1)
    
    # cv2.imshow("dil_can", cv2.Canny(dilation, 60, 160))
    # print(v_projection(cv2.Canny(dilation, 60, 160)))
    # print(v_projection(cv2.Canny(dilation, 60, 160)))
    dilation = cv2.copyMakeBorder(dilation, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = 255)

    #---------------------------Finished cleaning----------#
    
    ## Firstly to segment the entire word from input image using contours
    #cv2.imshow("dila2", dilation)
    dilation = skew_correction(dilation, before = True)
    extract = contour_detection(w, h, dilation)
    extract = skew_correction(extract)
    inp_shape = dilation.shape
    #----------------------------
    return extract, inp_shape

def thresh_before_resize(img):

    h, w = img.shape
    #############--------Preprocessing and cleaning along with shadow removal-----------------##
    """--------------------INNOVATION- Robust to good amount of blurness and shadow present in input image---------"""
    dilated_img = cv2.dilate(img.copy(), np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)  #using median blur to remove the undesired shadow along with abs difference and normalization
    diff_img = 255 - cv2.absdiff(img.copy(), bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    ret, thresh1 = cv2.threshold(norm_img, 90, 255, cv2.THRESH_BINARY + 
                                                cv2.THRESH_OTSU)
    
    kernel = np.ones((2,2),np.uint8)

    erosion = cv2.erode(thresh1,np.ones((1,1),np.uint8),iterations = 1)
    dilation = cv2.erode(erosion,kernel,iterations = 1)
    
    # cv2.imshow("dil_can", cv2.Canny(dilation, 60, 160))
    # print(v_projection(cv2.Canny(dilation, 60, 160)))
    # print(v_projection(cv2.Canny(dilation, 60, 160)))
    dilation = cv2.copyMakeBorder(dilation, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = 255)


    #---------------------------Finished cleaning----------#
   # cv2.imshow("dilation", dilation)
    ## Firstly to segment the entire word from input image using contours
    dilation = skew_correction(dilation, before = True)
    extract = contour_detection(w, h, dilation)
   # cv2.imshow("ext", extract*255)
    #----------------------------
    inp_shape = dilation.shape
    """ ----------INNOVATION - we made the function (skew_correction) which made the input rotation invariant wrt to horizotal-------"""
    
    extract = skew_correction(extract)

    W = 400
    H = int(h*W/w)
    extract = cv2.resize(extract*255, (W,H))//255

    return extract, inp_shape

def contour_detection(w, h, binary_img = None):
    global num_cycles
    try:
        contours, _ = cv2.findContours(cv2.erode(binary_img, np.ones((4,1),np.uint8), iterations = 1),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
     #   print(len(contours))
        """-------INNOVATION - making robust to shirorekha breaks(upper line breaks) even sometimes for very large breaks------------"""
        parts = np.asarray([cv2.boundingRect(c) for c in contours])
     #   print(parts)
        thresh_indexes = [False if  (part[2]==w+2 and part[3]==h+2) or (part[2]*part[3]< 0.01*w*h) else True for part in parts]  ## extra 2 for previous added padding of 1
        filters = parts[thresh_indexes]
        
        # Apart from the maximum one from filters if the width/height of that contour is greater than a ceratin threshold
        # It implies that the contours contain a letter which is segregated due to some noise in user way of writing like line breakings, gaps etc,
        # then we will add the corresponding contour to larger one of filter.
       # print(filters)
        if len(filters)!=1:
            w_h_ratio = [(part[2]/part[3], part[3]) for part in list(filters)]
            #print(w_h_ratio)
            is_letters = [True if i[0]>0.7 and i[1]>0.7*filters[:,3].max() else False for i in w_h_ratio] # is aspect ratio >0.7 means contained letter else not
            overall_cnt = filters[is_letters]
            
           # print(overall_cnt)
            x_min =  overall_cnt[:,0].min()    # to find minimum x from overall cnt
            y_min =  overall_cnt[:,1].min()   # to find minimum y index from overall cnt
            h_max = overall_cnt[:, 1].max()+overall_cnt[np.argmax(overall_cnt[:, 1]),3]-y_min  # to find max height from overall cnt
            w_all = overall_cnt[:, 0].max()+overall_cnt[np.argmax(overall_cnt[:, 0]),2]-x_min    # calculating the cumulative width of all from overall cnt
        else:
            x_min, y_min, w_all, h_max  = filters[0]
     #   print(x_min, y_min, h_max, w_all)
        # print(overall_cnt.shape)
        # print(w_h_ratio)
        # print(parts)
        # print(thresh_indexes)
        # print(filters)
        # print(is_letters)
        
        #------------------------------
        imgs = binary_img.copy()
        
      #  print("h - ", imgs.shape)
        
        #      continue
        diff = y_min-0
        if diff>h//20:
            diff = h//20
        
        
        # now selecting final coordinates to crop
        x_f1 = x_min-w//40
        y_f1 = y_min-diff
        x_f2 = x_min+w_all+w//20
        y_f2 = y_min+h_max+h//100
        
        
        if x_f1<0:
            x_f1 = 0
        if x_f2>w:
            x_f2 = w
        if y_f1<0:
            y_f1 = 0
        if y_f2>h:
            y_f2 = h
        #print(extract)
       # print((x_f1, y_f1), (x_f2, y_f2))
        partition = binary_img.copy()[y_f1:y_f2, x_f1:x_f2]//255
      #  print(partition.shape)
        cv2.rectangle(imgs, (x_f1, y_f1), (x_f2, y_f2), (0,0,0), 2)
      #  cv2.imshow("imgs", imgs)
        num_cycles = 0
    except:
        """Slightly unique"""
        """------if a contour is not detected then insteading of getting error or fail one possible try can be done using
        recursive call on darkeing the black intensity, so that may be the contour will be detected.---------------"""
        # if contour is not detected once then we increase the black intensity in order to get good intensity for contour
        num_cycles+=1   #increasing counts if not detected
        if num_cycles>=5:
            print("Contour_not_detected, even while applying extreme morphology, chhose different image")
            sys.exit(0) # stop execution
        partition = contour_detection(w, h, cv2.erode(binary_img.copy(), np.ones((2,2), np.uint8),iterations = 2))
    return partition


def extraction(img):
    """
    this function will behave as a generator which yield the letters at run time to avoid necessary memory usage

    Arguments: 
    img - input handwritten hindi captcha image
    
    Yield an image by image for model to train
    """
    extract1, shape1 = thresh_before_resize(img)
    extract2, shape2 = thresh_after_resize(img)
    print("ratio is - ", extract1.shape, shape1, extract2.shape, shape2)

    if ((extract1.shape[1]*shape1[0])/(shape1[1]*extract1.shape[0]))<((extract2.shape[1]*shape1[0])/(shape1[1]*extract2.shape[0])):
        extract = extract2
      #  print("extract2")
    else:
        extract = extract1
     #   print("extract1")
  #  cv2.imshow("ext1", extract1*255)
   # cv2.imshow("ext2", extract2*255)
   # cv2.imshow("ext", extract*255)
    
  #  cv2.imshow("extract", extract*255)
    extract = cv2.copyMakeBorder(extract*255, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value = 255)//255
    
    # Calculating pixel count of white in each images and remove 'Shirorekha' 
    col_pix_count = h_projection(extract) 
   # print(col_pix_count)
    line_indexes = np.where((col_pix_count==col_pix_count.min()))
  #  print(line_indexes)
    
    min_index = line_indexes[0][-1]
    #divided images based on above parameter into two parts out of which lower one will be further used for characters segmentation
  #  print(min_index)

    upper = extract[0:min_index+extract.shape[0]//7,:].copy()
    lower = extract[min_index+extract.shape[0]//7:,:].copy()
   # print(lower.shape)
   # print(lower.shape)
    # using canny edge detection and subsequent vertical projection to segment characters from it
   # print(lower.max())
  #  cv2.imshow("low", lower*255)
    edge = cv2.Canny(lower*255, 60, 160)
  #  print(edge)
   # cv2.imshow("canny", edge)
    row_pix_count = v_projection(edge)
   # print(row_pix_count)
    sep_indexes = np.where(row_pix_count==0)[0]
    sep_indexes = np.array(sep_indexes).astype(float)
    count = 0
   # print(sep_indexes)

   #-------------- perparing width index of each character and store it in a list for utlising later---------------
    seprators= []
    for i in range(len(row_pix_count)-1):
       # print(i)
        if (row_pix_count[i]==0) and (row_pix_count[i+1]==0):
            count+=1
        elif (row_pix_count[i]==0) and (row_pix_count[i+1]!=0):
            seprators.append(i-count)
            count = 0
    
    margin = 2 if count>0 else 0
    seprators.append(len(row_pix_count)-count+margin)

    #-------------------------DONE FORMATION-------------
   # print(seprators)
    

    ##-----Now the above list will consider the matra of bada aa, danda of ga seprately which need to be taken into consideration 
    # so we had made another modified list
    
    
    modified_seprate = [0]  # 0 taken by default as starting
    i = 0
    while (i<len(seprators)-1):
        part_prev = i
        part_new = i
        for j in range(i+1, len(seprators)):
            part_col_pixs = v_projection(cv2.Canny(lower[:,seprators[part_new]:seprators[j]]*255, 60,160))
          #  print(np.sum(part_col_pixs!=0)/len(part_col_pixs ))
            #print(len(part_col_pixs)/lower.shape[0])
            if np.sum(part_col_pixs!=0)/len(part_col_pixs )<=0.35 or len(part_col_pixs)/lower.shape[0]<=0.35:# if the black pizels count in columns is below a certain limit then it will be added into previous
                part_new += 1
            else:
                if part_new!=part_prev:
                    modified_seprate.append(seprators[part_new])
                    break
                part_new += 1
                
        i=part_new
    if seprators[-1] not in modified_seprate: # if last is not present then append it else no
        modified_seprate.append(seprators[-1])
    #print(modified_seprate)
    #--------------DONE---------------
   
   #-------------------GENERATOR to yield images one by one to make memory efficient running-------------------###

    for i in range(len(modified_seprate)-1):
        part = extract[:,modified_seprate[i]:modified_seprate[i+1]]*255
        h_p = h_projection(cv2.Canny(part, 60, 160))
        part = part[np.where(h_p!=0)[0],:]
        
        h_prev, w_prev =  part.shape
       # print( h_prev, w_prev)

        # padding images to make it in a square form so that it will be resized to 32*32(square image) smoothly
        if h_prev>w_prev:
            ext  = (h_prev-w_prev)//2
            part = cv2.copyMakeBorder(part, 10, 10, 10+ext, 10+ext, cv2.BORDER_CONSTANT, value = 255)
        if w_prev>=h_prev:
            ext  = (w_prev-h_prev)//2

#         part = cv2.copyMakeBorder(part, 10+ext, 10+ext, 10, 10, cv2.BORDER_CONSTANT, value = 255)
#         type1 = cv2.erode(cv2.resize(part, (64,64), cv2.INTER_NEAREST), np.ones((2,2), np.uint8), iterations = 1)
#         type2 = cv2.erode(cv2.resize(part, (64,64), cv2.INTER_NEAREST), np.ones((1,1), np.uint8), iterations = 1)
#         type3 = cv2.resize(part, (64,64), cv2.INTER_NEAREST)
#         yield type1, type2, type3
        
        part = cv2.copyMakeBorder(part, 10+ext, 10+ext, 10, 10, cv2.BORDER_CONSTANT, value = 255)
        type1 = cv2.erode(cv2.resize(part, (64,64), cv2.INTER_NEAREST), np.ones((1, 1), np.uint8), iterations = 1)
        type2 = cv2.erode(cv2.resize(part, (64,64), cv2.INTER_NEAREST), np.ones((2, 2), np.uint8), iterations = 1)
        type3 = cv2.resize(cv2.erode(part, np.ones((3, 3), np.uint8), iterations = 1), (64,64), cv2.INTER_NEAREST)
        yield type1, type2, type3




#------------------------------------CODE FOR TESTING ABOVE FUNCTION--------------------########### 



if __name__ == "__main__":

    img = cv2.imread("sample_words/15.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## resizing image with proper aspect ratio
    
    image_gen  = extraction(img) ## GENERATOR CALLED
    i=0
    while True:
        try:
            image1, image2, image3 = next(image_gen)
            cv2.imshow("type1"+str(i), image1)
            #  cv2.imshow("type2"+str(i), image2)
            # cv2.imshow("type3"+str(i), image3)

            i+=1
        except Exception as e:
            break

    cv2.imshow("word", img)
    

    if cv2.waitKey(0)&0Xff ==27:
        cv2.destroyAllWindows()
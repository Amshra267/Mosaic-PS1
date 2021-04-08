import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

num_cycles = 0   # num_of_times_the_loop_must run_then_stops while detecting contour recursively if not found
def skew_correction(img):
    """
    this function corrects the input image skewness wrt horizontal and aligns it well
    img - for binary_image
    """
   # cv2.imshow("masked", )
    t_ero = cv2.erode(np.uint8(np.logical_not(img.copy()))*255, np.ones((2,2),np.uint8), iterations=1)
   # cv2.imshow("t_ero", t_ero)
    coords = np.column_stack(np.where(t_ero > 0))
    #print(coords.shape)
   # print(img)
    angle = cv2.minAreaRect(coords)[-1]
    #print(angle)
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle
    # rotate the image to deskew it
    #print(angle)
    (h, w) = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
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
    
def contour_detection(w, h, binary_img = None):
    global num_cycles
    try:
        contours, _ = cv2.findContours(binary_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        """-------INNOVATION - making robust to shirorekha breaks(upper line breaks) even sometimes for very large breaks------------"""

        parts = np.asarray([cv2.boundingRect(c) for c in contours])
        thresh_indexes = [False if  (part[2]==w+2 or part[3]==h+2) or (part[2]*part[3]< 0.01*w*h) else True for part in parts]  ## extra 2 for previous added padding of 1
        filters = parts[thresh_indexes]
        
        cv2.imshow("before_cnt", binary_img)
        # Apart from the maximum one from filters if the width/height of that contour is greater than a ceratin threshold
        # It implies that the contours contain a letter which is segregated due to some noise in user way of writing like line breakings, gaps etc,
        # then we will add the corresponding contour to larger one of filter.
        
        w_h_ratio = [(part[2]/part[3], part[3]) for part in list(filters)]
        #print(w_h_ratio)
        is_letters = [True if i[0]>0.7 and i[1]>0.7*filters[:,3].max() else False for i in w_h_ratio] # is aspect ratio >0.7 means contained letter else not
        overall_cnt = filters[is_letters]
        
       # print(overall_cnt)
        x_min =  overall_cnt[:,0].min()    # to find minimum x from overall cnt
        y_min =  overall_cnt[:,1].min()   # to find minimum y index from overall cnt
        h_max = overall_cnt[:, 1].max()+overall_cnt[np.argmax(overall_cnt[:, 1]),3]-y_min  # to find max height from overall cnt
        w_all = overall_cnt[:, 0].max()+overall_cnt[np.argmax(overall_cnt[:, 0]),2]-x_min    # calculating the cumulative width of all from overall cnt
        # print(x_min, y_min, h_max, w_all)
        # print(overall_cnt.shape)
        # print(w_h_ratio)
        # print(parts)
        # print(thresh_indexes)
        # print(filters)
        # print(is_letters)
        
        #------------------------------
        imgs = binary_img.copy()

        #      continue
        diff = y_min-0
        if diff>25:
            diff = 25
        
        partition = binary_img.copy()[y_min-diff:y_min+h_max+5, x_min-2:x_min+w_all+8]//255
        #print(extract)
        
      #  cv2.rectangle(imgs, (x_min, y_min), (x_min+w_all, y_min+h_max), (0,0,0), 2)
     #   cv2.imshow("imgs", imgs)
        num_cycles = 0
    except:
        partition = contour_detection(w, h, skew_correction(binary_img)) # one of the major reason may come for this type of error is that image is very rotated 
                                                 # so making it skew invariant and foloowing recurrsive policy

    """Slightly unique"""
    """------if a contour is not detected then insteading of getting error or fail one possible try can be done using
    recursive call on darkeing the black intensity, so that may be the contour will be detected.---------------"""

    if partition is None: # if contour is not detected once then we increase the black intensity in order to get good intensity for contour
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
    ## resizing image with proper aspect ratio
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
    
    dilation = cv2.copyMakeBorder(dilation, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = 255)


    #---------------------------Finished cleaning----------#
    
    ## Firstly to segment the entire word from input image using contours
    extract = contour_detection(w, h, dilation)
    #----------------------------

    """ ----------INNOVATION - we made the function (skew_correction) which made the input rotation invariant wrt to horizotal-------"""
    extract = skew_correction(extract)
    extract = cv2.copyMakeBorder(extract*255, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value = 255)//255
    
    # Calculating pixel count of white in each images and remove 'Shirorekha' 
    col_pix_count = h_projection(extract) 
   # print(col_pix_count)
    line_indexes = np.where((col_pix_count<col_pix_count.min()+10))
  #  print(line_indexes)
    min_index = line_indexes[0][-1]
    #divided images based on above parameter into two parts out of which lower one will be further used for characters segmentation

   # print(extract.shape)
    upper = extract[0:min_index+10,:].copy()
    lower = extract[min_index+10:,:].copy()
   # print(lower.shape)
    # using canny edge detection and subsequent vertical projection to segment characters from it
    edge = cv2.Canny(lower*255, 60, 160)
    #cv2.imshow("canny", edge)
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
    
    margin = 1 if count>0 else 0
    seprators.append(len(row_pix_count)-count+margin)

    #-------------------------DONE FORMATION-------------
   # print(seprators)
    

    ##-----Now the above list will consider the matra of bada aa, danda of ga seprately which need to be taken into consideration 
    # so we had made another modified list
    
    
    modified_seprate = []
    i = 0
    while (i<len(seprators)-1):
        part_col_pixs = v_projection(cv2.Canny(lower[:,seprators[i]:seprators[i+1]]*255, 60,160))
        print(np.sum(part_col_pixs!=0)/len(part_col_pixs ))
        if np.sum(part_col_pixs!=0)/len(part_col_pixs )<0.5:# if the black pizels count in columns is below a certain limit then it will be added into previous
            modified_seprate.append(seprators[i+1])
        else:
            if seprators[i] not in modified_seprate:
                modified_seprate.append(seprators[i])
        i+=1
    if seprators[-1] not in modified_seprate: # if last is not present then append it else no
        modified_seprate.append(seprators[-1])
  #  print(modified_seprate)
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

        part = cv2.copyMakeBorder(part, 10+ext, 10+ext, 10, 10, cv2.BORDER_CONSTANT, value = 255)
        type1 = cv2.erode(cv2.resize(part, (64,64), cv2.INTER_NEAREST), kernel, iterations = 1)
        type2 = cv2.erode(cv2.resize(part, (64,64), cv2.INTER_NEAREST), np.ones((1,1), np.uint8), iterations = 1)
        type3 = cv2.resize(part, (64,64), cv2.INTER_NEAREST)
        yield type1, type2, type3




#------------------------------------CODE FOR TESTING ABOVE FUNCTION--------------------########### 



if __name__ == "__main__":

    img = cv2.imread("sample_words/13.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## resizing image with proper aspect ratio
    
    image_gen  = extraction(img) ## GENERATOR CALLED
    i=0
    while True:
        try:
            image1, image2, image3 = next(image_gen)
            cv2.imshow("type1"+str(i), image1)
            cv2.imshow("type2"+str(i), image2)
            cv2.imshow("type3"+str(i), image3)

            i+=1
        except Exception as e:
            break

    cv2.imshow("word", img)
    

    if cv2.waitKey(0)&0Xff ==27:
        cv2.destroyAllWindows()
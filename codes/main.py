import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from segmentar import extraction
from tensorflow.keras.models import load_model

mapping= {
    0 : "क", 1 : "ग", 2 : "घ", 3 : "च", 4 : "छ", 
    5 : "ज", 6 : "झ", 7 : "ञ", 8 : "ट", 9 : "ठ", 
    10 : "ड", 11 : "ण", 12 : "त", 13 : "थ", 14 : "द", 
    15 : "न", 16 : "फ", 17 : "म", 18 : "र", 19 : "ल",
    20 : "व", 21 : "श", 22 :"त्र", 23: "ज्ञ", 24 : "अ",
    27 : "उ", 28 : "ऊ", 29 : "ए", 30 : "ऐ", 31 : "ओ"
}


def second_rank(vec):
    """
    This function calculates the second best class from a vector, since we trained on more classes and selected out of it based on acc,
    so if a prediction comes out to be a class not present in selected ones then it will output second best class
    """

    return np.argsort(vec)[-2]

##-------loading model with weights---
model = load_model("model.h5")


if __name__=="__main__":
    print("Enter the path of Input Image")

    while True:
        path = input()
        if not os.path.exists(path):
            print("File does not exist, please enter Correct Path")
            continue
        break

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## resizing image with proper aspect ratio------------
    image_gen  = extraction(img) ## GENERATOR CALLED
    i=0
    word = []
    while True:
        try:
            image1, image2, image3 = next(image_gen)
            # cv2.imshow("img"+str(i), part*255)
            image1 = image1.reshape(-1,64,64,1).astype("float32")/255
            image2 = image2.reshape(-1,64,64,1).astype("float32")/255
            image3 = image3.reshape(-1,64,64,1).astype("float32")/255
            #print(part.shape)
            prediction = model(image1) + model(image2) + model(image3)
            cls = np.argmax(prediction, axis = 1)
#             print(cls)
            if cls[0] not in list(mapping.keys()):
               # print("here")
                cls = [second_rank(prediction)]  # means if not in our final selected character set then move to second best
           # print(np.argmax( model(image1), axis = 1))
           # print(np.argmax( model(image2), axis = 1))
           # print(np.argmax( model(image3), axis = 1))
           # print(prediction.shape)
            
            word.append(mapping[cls[0]])
            cv2.imshow("imgs" + str(i), image1.reshape(64,64)*255)
            i+=1
        except Exception as e:
            print(e)
            break

    #------------Resizing_finished--------------
    print("Predicted Word : "," ".join(word))

    #cv2.imshow("extract", img)
     
    if cv2.waitKey(0)&0Xff ==27:
        cv2.destroyAllWindows()

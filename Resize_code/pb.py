import glob
import cv2
import os

#--------------- LOOPING FOR CREATE FILE AND RENAME THE FILE NAME--------------------------------------
img_pathss = 'resize_pb' # can put all the image into one file
os.mkdir('Public Bank') #Store ur cropped and rename image into this file
i=0 #counting for file name

# FOR LOOP for MKDIR into FOLDER
for img in glob.glob(img_pathss + "/*.jpg"):
    image = cv2.imread(img)
    if(image.shape[0]>image.shape[1]): #protait
        print("FIRST ",image.shape) #4000
        imgResized = image[1850:2700,700:4000] #please change the size if not suitable *Make sure is rectangle
        cv2.imwrite("Public Bank/3_%01i.jpg" % i, imgResized)
    else: #landscape
        print("second ",image.shape) #4000
        imgResizedd = image[1500:2250,1000:4000] #please change the size if not suitable *Make sure is rectangle
        cv2.imwrite("Public Bank/3_%01i.jpg" % i, imgResizedd)

    i+=1
cv2.destroyAllWindows()

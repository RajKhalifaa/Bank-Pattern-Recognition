import glob
import cv2
import os


#--------------- LOOPING FOR CREATE FILE AND RENAME THE FILE NAME--------------------------------------
i=0 #counting for file name
img_pathss = 'resize_maybank' #put all the image into one file
os.mkdir('Maybank') #Store ur cropped and rename image into this file

# FOR LOOP for MKDIR into FOLDER
for img in glob.glob(img_pathss + "/*.jpg"):
    image = cv2.imread(img)
    imgResized = image[18:500,124:1920] #please change the size if not suitable *Make sure is rectangle
    cv2.imwrite("Maybank/2_%01i.jpg" %i, imgResized)

    i+=1

cv2.destroyAllWindows()


from PIL import Image
import glob
import cv2
import os
import math
import numpy as np

i=95


img_pathss = 'resize_bsn'
os.mkdir('Resized Folder_2')


# FOR LOOP for MKDIR into FOLDER
for img in glob.glob(img_pathss + "/*.jpg"):
    image = cv2.imread(img)
    imgResized = image[1350:3500,400:7500]
    resized = cv2.resize(image, (200, 125))
    cv2.imwrite("Resized Folder_2/1_%01i.jpg" %i, imgResized)

    i+=1
#    cv2.imshow('imge',imgResized)
#    cv2.waitKey(0)
cv2.destroyAllWindows()

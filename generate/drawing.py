import os
import cv2

def drawing():
    parent=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_static = os.path.join(parent, "static")
    img_Images = os.path.join(img_static, "Images")
    img = os.path.join(img_Images,"ID.png")
    image=cv2.imread(img)

    gray_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    inverted=255-gray_img
    blurred=cv2.GaussianBlur(inverted,(21,21),0)
    invertedblur = 255-blurred
    pencilsketch = cv2.divide(gray_img, invertedblur,scale=256.0)

    cv2.imwrite(os.path.join(img_Images, "drawingID.png"),pencilsketch)
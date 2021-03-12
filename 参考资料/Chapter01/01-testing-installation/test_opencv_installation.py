"""
Testing OpenCV installation
"""

# import the necessary packages (in this case, OpenCV package)
import cv2
import numpy as np

# Use the function cv2.imread() to read an image.
# The image should be in the working directory or a full path of image should be provided.
# load OpenCV logo image: 

# def cv_imread(file_path = ""):
#     file_path_gbk = file_path.encode('gbk')        # unicode转gbk，字符串变为字节数组
#     img_mat = cv2.imread(file_path_gbk.decode())  # 字节数组直接转字符串，不解码
#     return img_mat

def cv_imread(file_path):
    cv_img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)  
    return cv_img  


image = cv2.imread("D:\\project\\Mastering-OpenCV-4-with-Python\\Chapter01\\01-testing-installation\\logo.png")


# Use cv2.cvtColor() to convert an image from one color format to another
# In this case we use cv2.cvtColor() to convert the loaded image to grayscale (BGR to GRAY):
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use the function cv2.imshow() to show an image in a window.
# The window automatically fits to the image size.
# First argument is the window name.
# Second argument is the image to be displayed.
# Each created window should have different window names.
# Show original image:
cv2.imshow("OpenCV logo", image)

# Show grayscale image:
cv2.imshow("OpenCV logo gray format", gray_image)

# cv2.waitKey() is a keyboard binding function.
# The argument is the time in milliseconds.
# The function waits for specified milliseconds for any keyboard event.
# If any key is pressed in that time, the program continues.
# If 0 is passed, it waits indefinitely for a key stroke.
# Wait indefinitely for a key stroke (in order to see the created window):
cv2.waitKey(1000)

# To destroy all the windows we created call cv2.destroyAllWindows()
cv2.destroyAllWindows()

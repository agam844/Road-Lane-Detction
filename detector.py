import matplotlib.pyplot as plt
import cv2
import numpy as np

def region_of(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    line_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image, (x1,y1), (x2,y2), (0, 255, 0), 3)
    img = cv2.addWeighted(img, 0.8, line_image, 1, 0.0)
    return img


#image = cv2.imread('road.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def process(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)

    cropped_image = region_of(canny_image,
                              np.array([region_of_interest], np.int32),)

    lines = cv2.HoughLinesP(cropped_image, rho=2,theta=np.pi/60,threshold=50,lines=np.array([]),minLineLength=40,maxLineGap= 100)

    image_with_line = draw_the_lines(image, lines)
    return image_with_line

cap = cv2.VideoCapture('test.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#plt.imshow(image_with_line)
#plt.show()

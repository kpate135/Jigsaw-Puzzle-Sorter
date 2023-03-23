import os
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def draw_houghp_lines(image, houghp_lines_image, houghp_lines, potentialMinLine):
    for line in houghp_lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(houghp_lines_image,(x1,y1), (x2,y2),(0,255,0), 2)
        Hori = np.concatenate((image, houghp_lines_image), axis=1)
        
        cv2.imshow('HORIZONTAL', Hori)
        #cv2.imshow("image", image)
        #cv2.imshow("houghp", houghp_lines_image)
        cv2.waitKey()

def get_houhgp_lines(image,width, c):
    #print ("Width:", width)
    rho_resolution = 1
    theta_resolution = np.pi/180
    thresh = 25
    
    potentialMinLine = (int)((.51)*width)
    #check_image = image.copy()
    #cv2.line(check_image, (0,0), (potentialMinLine, 0), (0,255,0), 2)
    #cv2.line(check_image, (0,0), (0, potentialMinLine), (0,255,0), 2)
    #cv2.imshow("image", check_image)
    #cv2.waitKey()
    #print("Potential min line: ", potentialMinLine)

    #process image
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    edges_image = cv2.Canny(blurred_image, 50, 120)

    houghp_lines = cv2.HoughLinesP(edges_image, rho_resolution, theta_resolution, thresh, minLineLength=potentialMinLine, maxLineGap = 5)

    #if houghp_lines is not None:
        #draw_houghp_lines(image, np.zeros_like(image), houghp_lines, potentialMinLine)       

    return houghp_lines

def label_houghp_lines(houghp_lines):
    if houghp_lines is not None:
        num_lines = len(houghp_lines)
        if num_lines == 1:
            return "edge piece"
        elif num_lines == 2:
            return "corner piece"
        else:
            return ("num_lines: ", 2)
    else:
        return "non-edge piece"

def scale(x,y,w,h):
    scale = (int)(0.2*min(w,h)) #add this percent to each side
    x_scaled = x - scale
    y_scaled = y - scale
    w_scaled = w + scale*2
    h_scaled = h + scale*2
    
    return x_scaled,y_scaled,w_scaled,h_scaled

def process_image(image):
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blurred_image, 65, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #cv2.imshow("thres,",thresh)
    # Find contours, obtain bounding box, extract and save ROI
    
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #cv2.drawContours(image, cnts, -1, (0,255,0),3)
    
    for c in cnts:
        
        
        x,y,w,h = cv2.boundingRect(c)
        x_scaled, y_scaled, w_scaled, h_scaled = scale(x,y,w,h)
        origin = (x_scaled,y_scaled)
        width = x_scaled + w_scaled
        height = y_scaled + h_scaled
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,155,0), 2)
        cv2.rectangle(image, origin, (width, height), (255,0,0), 2)
        #check_image = image[x_scaled:width, y_scaled:height]
        #cv2.imshow("check", check_image)
        ROI = original[y_scaled:height, x_scaled:width]
        
        smallest_edge = min(w,h)
        houghp_lines = get_houhgp_lines(ROI, smallest_edge, c)
        label = "not edge"
        if houghp_lines is not None:
            label = "edge"
            for i in range(len(houghp_lines)-1):
                line2 = houghp_lines[i+1]
                line1_x1,line1_y1,line1_x2,line1_y2 = houghp_lines[i][0]
                line2_x1,line2_y1,line2_x2,line2_y2 = houghp_lines[i+1][0]
                m1 = (line1_y2-line1_y1)/(line1_x2-line1_x1)
                
                m2 = (line2_y2-line2_y1)/(line2_x2-line2_x1)
                diff = abs(abs(m1)-abs(m2))
                print(diff)
                if diff > .05:
                    label = "corner"
                else: label = "edge"

        #label = label_houghp_lines(houghp_lines) #check for edge pieces
        cv2.putText(image, label, (x-5,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (255,100,100), 2)

        #cv2.putText(image, label, (x-5,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (255,100,100), 2)
        
    return image

def capture_video():
    cap = cv2.VideoCapture("photos/many_pieces_vid1.mp4")
    #while (cap.isOpened()):
    #for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    while (True):
        ret, frame = cap.read()
        resize = cv2.resize(frame, (800,800))
        cv2.imshow('frame', resize)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("gray", gray)

        processed_image = process_image(resize)
        cv2.imshow('processed_frame', processed_image)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    
    cap.release()
    return



if (0):
    #image = cv2.imread('photos/many_pieces4.jpg')
    #image = rotate_image(image, 45)
    #image = resize(image, 70)
    capture_video()
else:
    image = cv2.imread('photos/many_pieces4.jpg')
    image = cv2.resize(image, (800,800))
    
    #image = rotate_image(image, 45)
    #image = resize(image, 70)
    #capture_video()
    #cv2.imshow("original", image)
    processed_image = process_image(image)
    cv2.imshow("processed_image", image)

cv2.waitKey()
cv2.destroyAllWindows()




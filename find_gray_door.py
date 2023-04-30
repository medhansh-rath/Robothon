#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
    
def remove_similar_lines(lines, threshold1=0.5, threshold2=500):
    unique_lines = [lines[0]] # Add the first line to the list of unique lines
    for i in range(1, len(lines)):
        m, b = lines[i]
        # Check if there is an existing line that is similar to the current line
        is_similar = False
        for j in range(len(unique_lines)):
            m_existing, b_existing = unique_lines[j]
            if abs(m - m_existing) < threshold1 and abs(b - b_existing) < threshold2:
                is_similar = True
                break
        # Add the line to the list of unique lines if it is not similar to any existing line
        if not is_similar:
            unique_lines.append(lines[i])
    return unique_lines

def find_gray_door(color_img, a=0, b=0, c=82, d=205, e=50, f=150):
    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    lower_grey = np.array([a, b, c])
    upper_grey = np.array([d, e, f])
    mask = cv2.inRange(img, lower_grey, upper_grey)

    kernel = np.ones((21, 21), np.uint8)
    img_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel)

    # Blur
    img_close = cv2.GaussianBlur(img_close, (11, 11), 0)

    # Define the region of interest
    roi = img_close[500:3500, 500:4500]

    # Perform an opening operation to remove small objects or connections
    kernel = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)

    # Find the contours in the opened image
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Draw the largest contour on a new image
    result = np.zeros_like(img)
    cv2.drawContours(result, [max_contour], 0, 255, -1)

    # Find the edge of the contour using Canny edge detection
    edge = cv2.Canny(result, 100, 200)

    lines = cv2.HoughLinesP(edge, rho=1, theta=np.pi/180, threshold=20, minLineLength=400, maxLineGap=300)

    # Convert lines to slope-intercept form
    slopes = []
    intercepts = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x1 = x1 + 500
        x2 = x2 + 500
        y1 = y1 + 500
        y2 = y2 + 500
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        slopes.append(m)
        intercepts.append(b)

    slope_lines = list(zip(slopes, intercepts))
    unique_lines = remove_similar_lines(slope_lines)
    unique_lines = unique_lines[:4]
    
    # Calculate the intersection points of the Hough lines
    corners = []
    for i in range(len(unique_lines)):
        for j in range(i + 1, len(unique_lines)):
            m1, b1 = unique_lines[i]
            m2, b2 = unique_lines[j]
            if abs(m1 - m2) > 1e-6:
                x = (b2 - b1) / (m1 - m2)
                y = m1 * x + b1
                corners.append((x, y))
    
    return corners


# In[ ]:





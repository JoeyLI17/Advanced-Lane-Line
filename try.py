import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import pickle
from PIL import Image, ImageDraw, ImageFont

# Read saved pickle file for undistort (mtx and dixt)
openFileName = './camera_cal/PRJ2_JLI_mtx_dist_pickle.p'

dist_pickle = pickle.load( open( openFileName, "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read saved pickle file for bird eye transfer (m and minv)
openFileName = 'JLI_p_t_transfer.p'
m_pickle = pickle.load( open( openFileName, "rb" ) )
m = m_pickle["m"]
minv = m_pickle["minv"]
ym_per_pix = m_pickle["ym_per_pix"]
xm_per_pix = m_pickle["xm_per_pix"]

global left_fit_g
global rigth_fit_g

left_fit_g = []
rigth_fit_g = []

def imageProcess(img):
    # step 1: undistort the image
    undistImage = cal_undistort(img,mtx,dist)
    # step 2: use combain channel
    # combain S channel of HLS and sobel x
    hls_binary = hls_select(img, thresh=(100, 255))
    grad_binary_x = abs_sobel_thresh(img, orient='x', sobel_thresh=(20, 100))
    combined_binary = np.zeros_like(hls_binary)
    combined_binary[(hls_binary ==1)|(grad_binary_x==1)]=1
    # step 3: transform to bird eye view
    dirdEye = dirdEyeTransf(combined_binary,m)
    # step 4: histogram view
    hist = histBottom(dirdEye)
    #try:
    #    fit, left_fit_g, rigth_fit_g,curve = search_around_poly(dirdEye,left_fit_g,rigth_fit_g)
    #except NameError:
    if len(left_fit_g)==0:
        fit, left_fit, right_fit,curve = fit_polynomial(dirdEye,polyFunction=False,ployLine=False)
        left_fit_g = left_fit
        rigth_fit_g = right_fit
    else:
        fit, left_fit_g, rigth_fit_g,curve = search_around_poly(dirdEye,left_fit_g,rigth_fit_g)

    # step 6: back to perspective
    back = perspectiveTransf(fit,minv)
    # combine original with color lane
    final = cv2.addWeighted(undistImage, 1, back, 0.6, 0)
    # step 7: find curvature in the real world in side step 5
    
    
    # step 8: over lay infomration
    xCenter = fit.shape[1]/2 # car center
    ymax = fit.shape[0]
    leftXbottom = left_fit_g[0]*ymax**2 + left_fit_g[1]*ymax + left_fit_g[2]
    rightXbottom = rigth_fit_g[0]*ymax**2 + rigth_fit_g[1]*ymax + rigth_fit_g[2]
    
    display = np.zeros_like(fit)
    
    laneCenter = (leftXbottom+rightXbottom)/2
    pixDet = xCenter-laneCenter # negtive car is on the left of the lane center
    mDet = pixDet*xm_per_pix # convert to real world
    
    if mDet<0:
        side = "left"
    else:
        side = "right"
    
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(final,("%.3f" % np.abs(mDet)+" m "+side+" to the lane center"),(20,100), \
                font, 1.5, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(final,("Radius of the curveture is "+"%.3f" % curve[2]+" m "),(20,180), \
                font, 1.5, (255,255,255), 3, cv2.LINE_AA)
    output = final
    return output
    
# Step:1
# Undistort the image
def cal_undistort(img,mtx,dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
    
# Step 2: 1
# Gradient, sobel x or y
# Define a function that takes an image, gradient orientation,
def abs_sobel_thresh(img, orient='x', sobel_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1

    # Return the result
    return binary_output
    
    
# Step 2: 2
# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output
    
    
# Step 3:
# bird eye view
def dirdEyeTransf(img,m):
    imshape = img.shape
    xGridMax = imshape[1]; # x max
    yGridMax = imshape[0]; # Y max
    dShape = (imshape[1],imshape[0]) # x,y
    warped = cv2.warpPerspective(img, m, dShape, flags = cv2.INTER_LINEAR)
    return warped
    
    
# Step 4: 1
# If read the pic for the first time
# find histogram for the bottom 100 pixle
def histBottom(dirdEye):

    # Grab only the bottom 100 pixle
    bottom_100 = dirdEye[dirdEye.shape[0]//2:,:]
    # Lane lines are likely to be mostly vertical nearest to the car
    # Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_100, axis=0)

    return histogram
    
# Step 7:
# find real world datapoints

def realWorldfit(leftx,lefty,rightx,righty):
    '''
    Generates fake data to use for calculating lane curvature.
    In your own project, you'll ignore this function and instead
    feed in the output of your lane detection algorithm to
    the lane curvature calculation.
    '''
    # step 7 find curveture in the real world
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    left_y_eval = np.max(lefty)
    right_y_eval = np.max(righty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*left_y_eval*ym_per_pix + \
                           left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*right_y_eval*ym_per_pix + \
                            right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    average_curve = (left_curverad+right_curverad)/2
    
    
    
    return left_curverad,right_curverad,average_curve
    
# Step 5: 1
# boxing and find points for the first frame
def find_lane_pixels(binary_warped,box=True): # bird eye
    # Take a histogram of the bottom 100 pickle of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:],
                       axis=0) # 3D to 2D
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    #plt.imshow(out_img)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2) # 2D
    # find base x position left and right separately
    # according to the Max y in the region    
    leftx_base = np.argmax(histogram[:midpoint]) # x for highest y
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 25
    # Set the width of the windows +/- margin
    margin = 40
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows): # window starts from 0
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        if box == True:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        # define the rectangle zone
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        else:
            leftx_current = leftx_current+int((leftx_current-leftx_base)/10)
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        else:
            rightx_current = rightx_current+int((rightx_current-rightx_base)/10)

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    # put everything in one array
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


# step 5: 2
# fit curve
def fit_polynomial(binary_warped,polyFunction=True,ployLine=True):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped,box=True)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        #print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # step 7 find curveture in the real world
    curve = realWorldfit(leftx,lefty,rightx,righty)
    
    #print("curve is: ",curve)
    #plt.text(600-10,150,str(curve),color='w',size=20)
            
    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    if ployLine==True:
        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
    
    # plot the polyromial function
    if polyFunction == True:
        # left equation
        a = "{:.2e}".format(left_fit[0])
        b = "{:.2e}".format(left_fit[1])
        c = "{:.2e}".format(left_fit[2])
        leftPolyText = "$f_{left}$(y)=" + "(" + a +")" + "*$y^2$+" +\
                       "(" + b +")" + "*y+" + "(" + c +")"
        
        plt.text(binary_warped.shape[1]/12,binary_warped.shape[0]*3/4,
                 leftPolyText,color='w',size=12)
        
        # Right equation
        e = "{:.2e}".format(right_fit[0])
        f = "{:.2e}".format(right_fit[1])
        g = "{:.2e}".format(right_fit[2])
        RightPolyText = "$f_{right}$(y)=" + "(" + e +")" + "*$y^2$+" +\
                       "(" + f +")" + "*y+" + "(" + g +")"
        
        plt.text(binary_warped.shape[1]/12,binary_warped.shape[0]*1/4,
                 RightPolyText,color='w',size=12)
    
    window_img = np.zeros_like(out_img)
    testLeftLine = np.array([np.transpose(np.vstack([left_fitx,ploty]))])
    tsetRightLine = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
    test_pts = np.hstack((testLeftLine, tsetRightLine))
    
    cv2.fillPoly(window_img, np.int_([test_pts]),(0,230,230))
    result2 = cv2.addWeighted(out_img, 1, window_img, 1, 0)

    return result2, left_fit, right_fit,curve
    
# step 5: 3
def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped,left_fit,right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # step 7 find curveture in the real world    
    curve = realWorldfit(leftx,lefty,rightx,righty)
    #print("curve 2 is: ",curve)

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 0]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    window_img = np.zeros_like(out_img)
    testLeftLine = np.array([np.transpose(np.vstack([left_fitx,ploty]))])
    tsetRightLine = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
    test_pts = np.hstack((testLeftLine, tsetRightLine))
    
    cv2.fillPoly(window_img, np.int_([test_pts]),(0,230,230))
    result2 = cv2.addWeighted(out_img, 1, window_img, 1, 0)
    
    return result2, left_fit, right_fit,curve

# Step 6:
# from bird eye back to perspective view
def perspectiveTransf(img,minv):
    imshape = img.shape
    xGridMax = imshape[1]; # x max
    yGridMax = imshape[0]; # Y max
    dShape = (imshape[1],imshape[0]) # x,y
    warped = cv2.warpPerspective(img, minv, dShape, flags = cv2.INTER_LINEAR)
    return warped
    
# Make a video

from moviepy.editor import VideoFileClip  # move clip processing
vidoeName = "project_video.mp4"

left_fit = []
clipInput = VideoFileClip(vidoeName)

clipOutput = clipInput.fl_image(imageProcess)

clipOutputName = "project_video_color_lane_JLI.mp4"
clipOutput.write_videofile(clipOutputName, audio=False)

  
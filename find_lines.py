import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import pickle
import glob
import math
from moviepy.editor import VideoFileClip
import os


# camera calibraton files"
files_calibration='camera_cal/calibration*.jpg'
# distorsion paramaters are saved in :
file_distorsion="calibration_wide_dist_pickle.p"
# for perspective points :  file straight_lines1.jpg
# files for testing
# distortion, perspective
file_test1='test_images/test6.jpg'
#file_test1='test_images/straight_lines2.jpg'

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# run flag
run=True

# debug and test flag
debug_distorsion=False
debug_perspective=False
debug_thresholding=False
debug_lines_from_scratch=False

###########################################
# evaluation of distorsion coefficients
###########################################

def init_calibration_distorsion():
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
  objp = np.zeros((6*9,3), np.float32)
  objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)


  # Arrays to store object points and image points from all the images.
  objpoints = [] # 3d points in real world space
  imgpoints = [] # 2d points in image plane.

  # Make a list of calibration images
  images = glob.glob(files_calibration)

  # Step through the list and search for chessboard corners
  for idx, fname in enumerate(images):
      img = cv2.imread(fname)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
      # Find the chessboard corners
      ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
  
      # If found, add object points, image points
      if ret == True:
          objpoints.append(objp)
          imgpoints.append(corners)
  img_size = (img.shape[1], img.shape[0])
  # Do camera calibration given object points and image points
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

  # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
  dist_pickle = {}
  dist_pickle["mtx"] = mtx
  dist_pickle["dist"] = dist
  pickle.dump( dist_pickle, open( file_distorsion, "wb" ) )
  
  return mtx, dist

###########################################
# evaluation of perspective matrix
###########################################
def init_perspective():
  # estimated with straight_lines1.jpg
  src = np.float32([[268.762,676.967],[1037.94,676.967], [769.189, 505.01],[523.094, 505.01]])
  dst = np.float32([[268.762,676.967],[1037.94,676.967], [1037.94, 505.01],[268.762, 505.01]])
  M = cv2.getPerspectiveTransform(src, dst)
  Minv = cv2.getPerspectiveTransform(dst, src)
  return M, Minv


###########################################
# class with all algorithms to detect lines
###########################################
class Line():
  def __init__(self):
    # was the line detected in the last iteration?
    self.detected = False
    #previous values for curves
    self.prev_left_fit=None
    self.prev_right_fit= None
    # distorsion parameters
    self.mtx, self.dist = init_calibration_distorsion()
    # perspective matrix
    self.M, self.Minv = init_perspective()

  ###########################################
  # make undistorsion
  ###########################################
  def undistortion_image(self,img):
    undist= cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    return undist

  ###########################################
  # transform perspective
  ###########################################
  def change_perspective(self,img,img_size,M):
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

  ###########################################
  # thresholding
  ###########################################
  def colour_thresholding(self,img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
  
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary

  ###########################################
  # detect the line
  ###########################################
  def find_lane_pixels(self,img):
    binary_warped=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
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
    for window in range(nwindows):
      # Identify window boundaries in x and y (and right and left)
      win_y_low = binary_warped.shape[0] - (window+1)*window_height
      win_y_high = binary_warped.shape[0] - window*window_height
       
      # the four below boundaries of the window #
      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin
      win_xright_high = rightx_current + margin
    
      # Identify the nonzero pixels in x and y within the window #
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
      (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
      (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
       
      # Append these indices to the lists
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)

      # If you found > minpix pixels, recenter next window on their mean position
      if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
      if len(good_right_inds) > minpix:
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
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

    return leftx, lefty, rightx, righty

  ###########################################
  # find lines from scratch
  ###########################################
  def lines_from_scratch(self,binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty = self.find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
      left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
      right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
      # Avoids an error if `left` and `right_fit` are still none or incorrect
      print('The function failed to fit a line!')
      left_fitx = 1*ploty**2 + 1*ploty
      right_fitx = 1*ploty**2 + 1*ploty
    return left_fitx, right_fitx, ploty , left_fit, right_fit

  ###########################################
  # insert region
  ###########################################
  def insert_region(self,left_fitx, right_fitx, ploty,img_size,img,left_curverad, right_curverad):
  
    lines_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(len(ploty)-1):
      #cv2.line(lines_img,(int(left_fitx[i]),int(ploty[i])),(int(left_fitx[i+1]),int(ploty[i+1])), [255, 0, 0],4 )
      #cv2.line(lines_img,(int(right_fitx[i]),int(ploty[i])),(int(right_fitx[i+1]),int(ploty[i+1])), [255, 0, 0],4 )
      cv2.fillPoly(lines_img, np.array([[ (int(left_fitx[i]),int(ploty[i])),(int(left_fitx[i+1]),int(ploty[i+1])),(int(right_fitx[i+1]),int(ploty[i+1])), (int(right_fitx[i]),int(ploty[i]))]], dtype=np.int32), [255,0,0])

    linespersp_img= self.change_perspective(lines_img,img_size,self.Minv)
    
    draw_img=cv2.addWeighted(img, 0.8, linespersp_img, 0.4, 0.)
    curve=(left_curverad+right_curverad)/2
    strcurve='%.1f' % (curve)
    textcur="Radius of curvature left: "+strcurve+" m"
    realcenter=(left_fitx[len(ploty)-1]+right_fitx[len(ploty)-1])/2
    deltacenter=(img.shape[1]/2-realcenter)*xm_per_pix
    if (deltacenter>0 ):
      strpos='%.2f' % (deltacenter)
      textpos="Vehicle is "+strpos+" m left of the center"
    else:
      strpos='%.2f' % (-deltacenter)
      textpos="Vehicle is "+strpos+" m right of the center"
    cv2.putText(draw_img,textcur, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0, 255), 6)
    cv2.putText(draw_img,textpos, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0, 255), 6)
    return draw_img

  ###########################################
  # curvature
  ###########################################
  def measure_curvature_real(self,left_fit, right_fit, ploty):

    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
  
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    return left_curverad, right_curverad

  ###########################################
  # fit a polynome
  ###########################################
  def fit_poly(self,img_shape, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each with np.polyfit()
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    # Calc both polynomials using ploty, left_fit and right_fit
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty, left_fit , right_fit

  ###########################################
  # search around previous lines
  ###########################################
  def search_around_poly(self,binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 50

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # around
    left_lane_inds = ((nonzerox > (self.prev_left_fit[0]*(nonzeroy**2) + self.prev_left_fit[1]*nonzeroy +
                    self.prev_left_fit[2] - margin)) & (nonzerox < (self.prev_left_fit[0]*(nonzeroy**2) +
                    self.prev_left_fit[1]*nonzeroy + self.prev_left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (self.prev_right_fit[0]*(nonzeroy**2) + self.prev_right_fit[1]*nonzeroy +
                    self.prev_right_fit[2] - margin)) & (nonzerox < (self.prev_right_fit[0]*(nonzeroy**2) +
                    self.prev_right_fit[1]*nonzeroy + self.prev_right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit , right_fit = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    return left_fitx, right_fitx, ploty , left_fit , right_fit

  ###########################################
  # process image
  ###########################################
  def process_image(self,img):
    img_size = (img.shape[1], img.shape[0])
    dist_img= self.undistortion_image(img)
    thre_img = self.colour_thresholding(dist_img)
    persp_img= self.change_perspective(thre_img,img_size, self.M)
    
    if (self.detected==False):
      left_fitx, right_fitx, ploty, left_fit, right_fit=self.lines_from_scratch(persp_img)
      self.detected=True
    else:
      left_fitx, right_fitx, ploty, left_fit, right_fit=self.search_around_poly(persp_img)
    self.prev_left_fit=left_fit
    self.prev_right_fit=right_fit
    
    left_curverad, right_curverad = self.measure_curvature_real(left_fit, right_fit, ploty)
    draw_img=self.insert_region(left_fitx, right_fitx, ploty,img_size,img, left_curverad, right_curverad)
    return draw_img
  
  
  ###########################################
  # video pipeline for lines detection
  ###########################################
  def video_pipeline(self,video_name):
    white_output = "output_images/"+video_name
    #clip1= VideoFileClip(video_name).subclip(0,5)
    clip1 = VideoFileClip(video_name)
    white_clip = clip1.fl_image(self.process_image)
    white_clip.write_videofile(white_output, audio=False)
  
  ###########################################
  # test result of line detection form scratch
  ###########################################
  def test_find_lines_from_scratch(self):
    img = cv2.imread(file_test1)
    img_size = (img.shape[1], img.shape[0])
    dist_img= self.undistortion_image(img)
    thre_img = self.colour_thresholding(dist_img)
    persp_img= self.change_perspective(thre_img,img_size, self.M)
    left_fitx, right_fitx, ploty, left_fit, right_fit=self.lines_from_scratch(persp_img)
    left_curverad, right_curverad = self.measure_curvature_real(left_fit, right_fit, ploty)
    draw_img=self.insert_region(left_fitx, right_fitx, ploty,img_size,img, left_curverad, right_curverad)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(draw_img)
    ax2.set_title('lines detection', fontsize=30)
    # on screen
    plt.show()

  ###########################################
  # test result of thresholding
  ###########################################
  def test_thresholding(self):
    img = mpimg.imread(file_test1)
    img_size = (img.shape[1], img.shape[0])
    und_img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    thre_img = self.colour_thresholding(und_img)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=40)
    ax2.imshow(thre_img)
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # on screen
    plt.show()

  ###########################################
  # test result of changing perspective
  ###########################################
  def test_perspective(self):
    # undistortion of an image
    img = cv2.imread(file_test1)
    img_size = (img.shape[1], img.shape[0])
    und_img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    persp_img= self.change_perspective(und_img,img_size, self.M)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(persp_img)
    ax2.set_title('change perspective Image', fontsize=30)
    # on screen
    plt.show()
    
  ###########################################
  # test result of calibration
  ###########################################
  def test_calibration(self):
    # Test undistortion of an image
    img = cv2.imread(file_test1)
    img_size = (img.shape[1], img.shape[0])
    dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    # on screen
    plt.show()
  
  ###########################################
  # test result of undistorsion
  ###########################################
  def test_images(self):
    #reading in an image
    files=os.listdir("test_images/")
    for file in files:
        image=mpimg.imread('test_images/'+file)
        final_image=self.process_image(image)
        mpimg.imsave('output_images/'+file,final_image)


############################
# main
############################

mylines=Line()

if (run):
  mylines.test_images()
  mylines.video_pipeline("project_video.mp4")
#  mylines.video_pipeline("challenge_video.mp4")
#  mylines.video_pipeline("harder_challenge_video.mp4")

# debug
if (debug_distorsion):
  mylines.test_calibration()
if (debug_perspective):
  mylines.test_perspective()
if (debug_thresholding):
  mylines.test_thresholding()
if (debug_lines_from_scratch):
  mylines.test_find_lines_from_scratch()











  

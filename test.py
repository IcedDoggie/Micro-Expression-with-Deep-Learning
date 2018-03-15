import cv2 # Import relevant libraries
import numpy as np

img = cv2.imread('cameraman.png', 0) # Read in image

height = img.shape[0] # Get the dimensions
width = img.shape[1]

# Define mask
mask = 255*np.ones(img.shape, dtype='uint8')

# Draw circle at x = 100, y = 70 of radius 25 and fill this in with 0
cv2.circle(mask, (100, 70), 25, 0, -1)    

# Apply distance transform to mask
out = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

# Define scale factor
scale_factor = 10

# Create output image that is the same as the original
filtered = img.copy() 

# Create floating point copy for precision
img_float = img.copy().astype('float')

# Number of channels
if len(img_float.shape) == 3:
  num_chan = img_float.shape[2]
else:
  # If there is a single channel, make the images 3D with a singleton
  # dimension to allow for loop to work properly
  num_chan = 1
  img_float = img_float[:,:,None]
  filtered = filtered[:,:,None]

# For each pixel in the input...
for y in range(height):
  for x in range(width):

	# If distance transform is 0, skip
	if out[y,x] == 0.0:
	  continue

	# Calculate M = d / S
	mask_val = np.ceil(out[y,x] / scale_factor)

	# If M is too small, set the mask size to the smallest possible value
	if mask_val <= 3:
	  mask_val = 3

	# Get beginning and ending x and y coordinates for neighbourhood
	# and ensure they are within bounds
	beginx = x-int(mask_val/2)
	if beginx < 0:
	  beginx = 0

	beginy = y-int(mask_val/2)
	if beginy < 0:
	  beginy = 0

	endx = x+int(mask_val/2)
	if endx >= width:
	  endx = width-1

	endy = y+int(mask_val/2)
	if endy >= height:
	  endy = height-1

	# Get the coordinates of where we need to grab pixels
	xvals = np.arange(beginx, endx+1)
	yvals = np.arange(beginy, endy+1)
	(col_neigh,row_neigh) = np.meshgrid(xvals, yvals)
	col_neigh = col_neigh.astype('int')
	row_neigh = row_neigh.astype('int')

	# Get the pixels now
	# For each channel, do the foveation
	for ii in range(num_chan):
	  chan = img_float[:,:,ii]
	  pix = chan[row_neigh, col_neigh].ravel()

	  # Calculate the average and set it to be the output
	  filtered[y,x,ii] = int(np.mean(pix))

# Remove singleton dimension if required for display and saving
if num_chan == 1:
  filtered = filtered[:,:,0]

# Show the image
cv2.imshow('Output', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
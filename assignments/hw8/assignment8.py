# ASSIGNMENT 8
# Hieu Nguyen
# GTID: 903185448
# Email: hieu@gatech.edu

import numpy as np
import scipy as sp
import scipy.signal
import cv2

# Import ORB as SIFT to avoid confusion.
try:
    from cv2 import ORB as SIFT
except ImportError:
    try:
        from cv2 import SIFT
    except ImportError:
        try:
            SIFT = cv2.ORB_create
        except:
            raise AttributeError("Your OpenCV(%s) doesn't have SIFT / ORB."
                                 % cv2.__version__)


""" Assignment 8 - Panoramas

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file. (This is a problem
    for us when grading because running 200 files results a lot of images being
    saved to file and opened in dialogs, which is not ideal). Thanks.

    2. DO NOT import any other libraries aside from the three libraries that we
    provide. You may not import anything else, you should be able to complete
    the assignment with the given libraries (and in most cases without them).

    3. DO NOT change the format of this file. Do not put functions into classes,
    or your own infrastructure. This makes grading very difficult for us. Please
    only write code in the allotted region.
"""

def getImageCorners(image):
    """ For an input image, return its four corners.

    You should be able to do this correctly without instruction. If in doubt,
    resort to the testing framework. The order in which you store the corners
    does not matter.

    Note: The reasoning for the shape of the array can be explained if you look
    at the documentation for cv2.perspectiveTransform which we will use on the
    output of this function. Since we will apply the homography to the corners
    of the image, it needs to be in that format.

    Another note: When storing your corners, they are assumed to be in the form
    (X, Y) -- keep this in mind and make SURE you get it right.

    Args:
        image (numpy.ndarray): Input can be a grayscale or color image.

    Returns:
        corners (numpy.ndarray): Array of shape (4, 1, 2). Type of values in the
                                 array is np.float32.
    """
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    # WRITE YOUR CODE HERE
    corners[1] = [0, image.shape[0]]
    corners[2] = [image.shape[1], 0]
    corners[3] = [image.shape[1], image.shape[0]]
    return corners
    # END OF FUNCTION

def findMatchesBetweenImages(image_1, image_2, num_matches):
    """ Return the top list of matches between two input images.

    Note: You will not be graded for this function. This function is almost
    identical to the function in Assignment 7 (we just parametrized the number
    of matches). We expect you to use the function you wrote in A7 here. We will
    also release a solution for how to do this after A7 submission has closed.

    If your code from A7 was wrong, don't worry, you will not lose points in
    this assignment because your A7 code was wrong (hence why we will provide a
    solution for you after A7 closes).

    This function detects and computes SIFT (or ORB) from the input images, and
    returns the best matches using the normalized Hamming Distance through brute
    force matching.

    Args:
        image_1 (numpy.ndarray): The first image (grayscale).
        image_2 (numpy.ndarray): The second image. (grayscale).
        num_matches (int): The number of desired matches. If there are not
                           enough, return as many matches as you can.

    Returns:
        image_1_kp (list): The image_1 keypoints, the elements are of type
                           cv2.KeyPoint.
        image_2_kp (list): The image_2 keypoints, the elements are of type 
                           cv2.KeyPoint.
        matches (list): A list of matches, length 'num_matches'. Each item in 
                        the list is of type cv2.DMatch. If there are less 
                        matches than num_matches, this function will return as
                        many as it can.

    """
    # matches - type: list of cv2.DMath
    matches = None
    # image_1_kp - type: list of cv2.KeyPoint items.
    image_1_kp = None
    # image_1_desc - type: numpy.ndarray of numpy.uint8 values.
    image_1_desc = None
    # image_2_kp - type: list of cv2.KeyPoint items.
    image_2_kp = None
    # image_2_desc - type: numpy.ndarray of numpy.uint8 values.
    image_2_desc = None

    # COPY YOUR CODE FROM A7 HERE.
    orb = cv2.ORB()
    image_1_kp, image_1_desc = orb.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = orb.detectAndCompute(image_2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(image_1_desc, image_2_desc)
    matches = sorted(matches, key=lambda x:x.distance)
    matches = matches[:num_matches]

    # print "Image 1: {} keypoints found".format(len(image_1_kp))
    # print "Image 2: {} keypoints found".format(len(image_2_kp))
    # print "{} matches found".format(len(matches))
    # img1_kp = cv2.drawKeypoints(image_1, image_1_kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # img2_kp = cv2.drawKeypoints(image_2, image_2_kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite("images/output/img1_kp.jpg", img1_kp)
    # cv2.imwrite("images/output/img2_kp.jpg", img2_kp)

    return image_1_kp, image_2_kp, matches
  # END OF FUNCTION.

def findHomography(image_1_kp, image_2_kp, matches):
    """ Returns the homography between the keypoints of image 1, image 2, and
        its matches.

    Follow these steps:
        1. Iterate through matches and:
            1a. Get the x, y location of the keypoint for each match. Look up
                the documentation for cv2.DMatch. Image 1 is your query image,
                and Image 2 is your train image. Therefore, to find the correct
                x, y location, you index into image_1_kp using match.queryIdx,
                and index into image_2_kp using match.trainIdx. The x, y point
                is stored in each keypoint (look up documentation).
            1b. Set the keypoint 'pt' to image_1_points and image_2_points, it
                should look similar to this inside your loop:
                    image_1_points[match_idx] = image_1_kp[match.queryIdx].pt
                    # Do the same for image_2 points.

        2. Call cv2.findHomography and pass in image_1_points, image_2_points,
           use method=cv2.RANSAC and ransacReprojThreshold=5.0. I recommend
           you look up the documentation on cv2.findHomography to better
           understand what these parameters mean.
        3. cv2.findHomography returns two values, the homography and a mask.
           Ignore the mask, and simply return the homography.

    Args:
        image_1_kp (list): The image_1 keypoints, the elements are of type
                           cv2.KeyPoint.
        image_2_kp (list): The image_2 keypoints, the elements are of type 
                           cv2.KeyPoint.
        matches (list): A list of matches. Each item in the list is of type
                        cv2.DMatch.
    Returns:
        homography (numpy.ndarray): A 3x3 homography matrix. Each item in
                                    the matrix is of type numpy.float64.
    """
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

    # WRITE YOUR CODE HERE.
    i = 0
    for mat in matches:
        image_1_points[i] = np.float32(image_1_kp[mat.queryIdx].pt)
        image_2_points[i] = np.float32(image_2_kp[mat.trainIdx].pt)
        i += 1
    M_hom, mask = cv2.findHomography(image_1_points, image_2_points, \
                    method=cv2.RANSAC, ransacReprojThreshold=5.0)
    return M_hom
    # END OF FUNCTION

def blendImagePair(warped_image, image_2, point):
    """ This is the blending function. We provide a basic implementation of
    this function that we would like you to replace.

    This function takes in an image that has been warped and an image that needs
    to be inserted into the warped image. Lastly, it takes in a point where the
    new image will be inserted.

    The current method we provide is very simple, it pastes in the image at the
    point. We want you to replace this and blend between the images.

    We want you to be creative. The most common implementation would be to take
    the average between image 1 and image 2 only for the pixels that overlap.
    That is just a starting point / suggestion but you are encouraged to use
    other approaches.

    Args:
        warped_image (numpy.ndarray): The image provided by cv2.warpPerspective.
        image_2 (numpy.ndarray): The image to insert into the warped image.
        point (numpy.ndarray): The point (x, y) to insert the image at.

    Returns:
        image: The warped image with image_2 blended into it.
    """
    output_image = np.copy(warped_image)
    # REPLACE THIS WITH YOUR BLENDING CODE.
    output_image[point[1]:point[1] + image_2.shape[0],
                 point[0]:point[0] + image_2.shape[1]] = image_2

    # Create mask of overlapping region
    mask_1 = np.copy(warped_image)
    mask_2 = np.copy(warped_image)
    mask_1[mask_1 > 0] = 127
    mask_2[:,:,:] = 0
    mask_overlap = np.copy(mask_2)
    mask_2[point[1]:point[1]+image_2.shape[0], point[0]:point[0]+image_2.shape[1]] = 127
    mask_overlap = mask_1 + mask_2
    mask_overlap[mask_overlap > 127] = 255
    mask_overlap[mask_overlap < 128] = 0

    # Find corners of overlap mask
    max_col = 0
    min_col = 100000
    max_row = 0
    min_row = 100000
    for row in xrange(mask_overlap.shape[0]):
        for col in xrange(mask_overlap.shape[1]):
            if(col>max_col):
                if(mask_overlap[row,col,0]==255):
                    max_col = col
            if(col<min_col):
                if(mask_overlap[row,col,0]==255):
                    min_col = col
            if(row>max_row):
                if(mask_overlap[row,col,0]==255):
                    max_row = row
            if(row<min_row):
                if(mask_overlap[row,col,0]==255):
                    min_row = row 

    # Iterate over output image and replace overlap with weighted average
    row_top_threshold = 100
    row_bot_threshold = 150
    for row in xrange(output_image.shape[0]):
        for col in xrange(output_image.shape[1]):
            if (mask_overlap[row, col, 0] > 0):
                x_weight = (float(col)-min_col)/(max_col-min_col)
                output_image[row, col] = x_weight*output_image[row, col] + (1-x_weight)*warped_image[row, col]
                if (row>=min_row and row<=min_row+row_top_threshold):
                    y_weight_top = (float(row)-min_row)/row_top_threshold
                    output_image[row, col] = y_weight_top*output_image[row, col] + (1-y_weight_top)*warped_image[row, col]
                if (row<=max_row and row>=max_row-row_bot_threshold):
                    y_weight_bot = 1.0 - ((max_row-float(row))/row_bot_threshold)
                    output_image[row, col] = y_weight_bot*image_2[row-point[1], col-point[0]] + (1-y_weight_bot)*output_image[row, col]
    return output_image
    # END OF FUNCTION

def warpImagePair(image_1, image_2, homography):
    """ Warps image 1 so it can be blended with image 2 (stitched).

    Follow these steps:
        1. Obtain the corners for image 1 and image 2 using the function you
        wrote above.
        
        2. Transform the perspective of the corners of image 1 by using the
        image_1_corners and the homography to obtain the transformed corners.
        
        Note: Now we know the corners of image 1 and image 2. Out of these 8
        points (the transformed corners of image 1 and the corners of image 2),
        we want to find the minimum x, maximum x, minimum y, and maximum y. We
        will need this when warping the perspective of image 1.

        3. Join the two corner arrays together (the transformed image 1 corners,
        and the image 2 corners) into one array of size (8, 1, 2).

        4. For the first column of this array, find the min and max. This will
        be your minimum and maximum X values. Store into x_min, x_max.

        5. For the second column of this array, find the min and max. This will
        be your minimum and maximum Y values. Store into y_min, y_max.

        6. Create a translation matrix that will shift the image by the required
        x_min and y_min (should be a numpy.ndarray). This looks like this:
            [[1, 0, -1 * x_min],
             [0, 1, -1 * y_min],
             [0, 0, 1]]

        Note: We'd like you to explain the reasoning behind multiplying the
        x_min and y_min by negative 1 in your writeup.

        7. Compute the dot product of your translation matrix and the homography
        in order to obtain the homography matrix with a translation.

        8. Then call cv2.warpPerspective. Pass in image 1, the dot product of
        the matrix computed in step 6 and the passed in homography and a vector
        that will fit both images, since you have the corners and their max and
        min, you can calculate it as (x_max - x_min, y_max - y_min).

        9. To finish, you need to blend both images. We have coded the call to
        the blend function for you.

    Args:
        image_1 (numpy.ndarray): Left image.
        image_2 (numpy.ndarray): Right image.
        homography (numpy.ndarray): 3x3 matrix that represents the homography
                                    from image 1 to image 2.

    Returns:
        output_image (numpy.ndarray): The stitched images.
    """
    # Store the result of cv2.warpPerspective in this variable.
    warped_image = None
    # The minimum and maximum values of your corners.
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0

    # WRITE YOUR CODE HERE
    image_1_corners = getImageCorners(image_1)
    image_2_corners = getImageCorners(image_2)
    image_1_corners_t = cv2.perspectiveTransform(image_1_corners, homography)
    join_corners = np.append(image_1_corners_t, image_2_corners, axis=0)
    x_min = np.amin(join_corners[:,:,0])
    x_max = np.amax(join_corners[:,:,0])
    y_min = np.amin(join_corners[:,:,1])
    y_max = np.amax(join_corners[:,:,1])
    trans_mat = np.array([[1, 0, -1 * x_min], [0, 1, -1 * y_min], [0, 0, 1]])
    trans_hom_mat = np.dot(trans_mat, homography)
    warped_image = cv2.warpPerspective(image_1, trans_hom_mat, (x_max-x_min, y_max-y_min))

    # END OF CODING
#    cv2.imwrite("images/output/warped_image.jpg", warped_image)
    output_image = blendImagePair(warped_image, image_2, (-1 * x_min, -1 * y_min))
    return output_image

# Some simple testing.
# image_1 = cv2.imread("images/source/panorama_1/1.jpg")
# image_2 = cv2.imread("images/source/panorama_1/2.jpg")
# image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
# image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
# image_1_kp, image_2_kp, matches = findMatchesBetweenImages(image_1_gray, image_2_gray, 100)
# homography = findHomography(image_1_kp, image_2_kp, matches)
# result = warpImagePair(image_1, image_2, homography)
# cv2.imwrite("images/output/panorama_1_result.jpg", result)

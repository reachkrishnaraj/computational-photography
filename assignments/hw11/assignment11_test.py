import cv2
import numpy as np
import sys
import os

import assignment11

def test_videoVolume():
    image_list1 = [np.array([[[0,  0,  0], [0,  0,  0]]], dtype = np.uint8),
                   np.array([[[1,  1,  1], [1,  1,  1]]], dtype = np.uint8),
                   np.array([[[2,  2,  2], [2,  2,  2]]], dtype = np.uint8),
                   np.array([[[3,  3,  3], [3,  3,  3]]], dtype = np.uint8)]

    video_volume1 = np.array([[[[ 0,  0,  0], [ 0,  0,  0]]],
                              [[[ 1,  1,  1], [ 1,  1,  1]]],
                              [[[ 2,  2,  2], [ 2,  2,  2]]],
                              [[[ 3,  3,  3], [ 3,  3,  3]]]], dtype = np.uint8)

    image_list2 = [np.array([[[2, 2, 2],
                              [2, 2, 2],
                              [2, 2, 2],
                              [2, 2, 2]],
                             [[2, 2, 2],
                              [2, 2, 2],
                              [2, 2, 2],
                              [2, 2, 2]]], dtype = np.uint8),
                   np.array([[[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]]], dtype = np.uint8),
                   np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]], dtype = np.uint8)]

    video_volume2 = np.array([[[[2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2]],
                               [[2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2]]],
                              [[[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]],
                               [[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]]],
                              [[[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]],
                               [[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]]]], dtype = np.uint8)

    print "Evaluating videoVolume:"
    for img_list, true_out in zip((image_list1, image_list2),
                                  (video_volume1, video_volume2)):
        print "Input:\n{}\n".format(img_list)
        usr_out = assignment11.videoVolume(img_list)

        if type(usr_out) != type(true_out):
            print "Error: videoVolume has type {}. Expected type is {}.".format(
                type(usr_out), type(true_out))
            return False

        if usr_out.shape != true_out.shape:
            print ("Error: videoVolume has shape {}. " + 
                   "Expected shape is {}.").format(usr_out.shape,
                                                   true_out.shape)
            return False

        if usr_out.dtype != true_out.dtype:
            print ("Error: videoVolume has dtype {}." + 
                   " Expected dtype is {}.").format(usr_out.dtype,
                                                    true_out.dtype)
            return False

        if not np.all(usr_out == true_out):
            print ("Error: videoVolume has value:\n{}\n" +
                   "Expected value:\n{}").format(usr_out, true_out)
            return False
        print "Passed current input."
    print "videoVolume passed."
    return True

def test_sumSquaredDifferences():
    video_volume1 = np.array([[[[ 0,  0,  0], [ 0,  0,  0]]],
                              [[[ 1,  1,  1], [ 1,  1,  1]]],
                              [[[ 2,  2,  2], [ 2,  2,  2]]],
                              [[[ 3,  3,  3], [ 3,  3,  3]]]], dtype = np.uint8)

    video_volume2 = np.array([[[[2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2]],
                               [[2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2]]],
                              [[[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]],
                               [[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]]],
                              [[[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]],
                               [[255, 255, 255],
                                [255, 255, 255],
                                [255, 255, 255],
                                [255, 255, 255]]]], dtype = np.uint8)

    diff1 = np.array([[  0.,   6.,  24.,  54.],
                      [  6.,   0.,   6.,  24.],
                      [ 24.,   6.,   0.,   6.],
                      [ 54.,  24.,   6.,   0.]], dtype = np.float)

    diff2 = np.array([[      0.,     24.,  768156.],
                      [     24.,      0.,  774204.],
                      [ 768156., 774204.,       0.]], dtype = np.float)

    for vid_volume, true_out in zip((video_volume1, video_volume2),
                                    (diff1, diff2)):
        print "Input:\n{}\n".format(vid_volume)
        usr_out = assignment11.sumSquaredDifferences(vid_volume)

        if type(usr_out) != type(true_out):
            print ("Error: sumSquaredDifferences has type {}." + 
                   "Expected type is {}.").format(type(usr_out), type(true_out))
            return False

        if usr_out.shape != true_out.shape:
            print ("Error: sumSquaredDifferences has shape {}." + 
                   "Expected shape is {}.").format(usr_out.shape,
                                                   true_out.shape)
            return False

        if usr_out.dtype != true_out.dtype:
            print ("Error: sumSquaredDifferences has dtype {}." + 
                   "Expected dtype is {}.").format(usr_out.dtype,
                                                   true_out.dtype)
            return False

        if not np.all(np.abs(usr_out - true_out) < 1.):
            print ("Error: sumSquaredDifferences has value:\n{}\n" +
                   "Expected value:\n{}").format(usr_out, true_out)
            return False
        print "Passed current input."
    print "sumSquaredDifferences passed."
    return True

def test_transitionDifference():
    ssd1 = np.zeros((9,9), dtype = float) 
    ssd1[4,4] = 1

    ssd2 = np.eye(5, dtype = float)

    out1 = np.array([[0.0625, 0.  , 0.   , 0.  ,  0.    ],
                     [0.    , 0.25, 0.   , 0.  ,  0.    ],
                     [0.    , 0.  , 0.375, 0.  ,  0.    ],
                     [0.    , 0.  , 0.   , 0.25,  0.    ],
                     [0.    , 0.  , 0.   , 0.  ,  0.0625]], dtype = float)
    out2 = np.array([[1.]], dtype = float)

    for ssd, true_out in zip((ssd1, ssd2), (out1, out2)):
        print "Input:\n{}\n".format(ssd)
        usr_out = assignment11.transitionDifference(ssd)

        if type(usr_out) != type(true_out):
            print ("Error: transitionDifference output has type {}. " +
                   "Expected type is {}.").format(type(usr_out), type(true_out))
            return False

        if usr_out.shape != true_out.shape:
            print ("Error: transitionDifference output has shape {}. " +
                   "Expected shape is {}.").format(usr_out.shape,
                                                   true_out.shape)
            return False

        if usr_out.dtype != true_out.dtype:
            print ("Error: transitionDifference output has dtype {}. " +
                   "Expected dtype is {}.").format(usr_out.dtype,
                                                   true_out.dtype)
            return False

        if not np.all(np.abs(usr_out - true_out) < 0.10):
            print ("Error: transitionDifference output has value:\n{}\n" + 
                   "Expected value:\n{}").format(usr_out, true_out)
            return False
        print "Passed current input."
    print "transitionDifference passed."
    return True

def test_findBiggestLoop():
    diff1 = np.ones((5,5), dtype = float)
    alpha1 = 1
    out1 = (0,4)

    diff2 = np.array([[ 0.,  1.,  1.,  5.],
                      [ 1.,  0.,  3.,  4.],
                      [ 1.,  3.,  0.,  5.],
                      [ 5.,  4.,  5.,  0.]])
    alpha2 = 1 
    out2 = (0,2)

    diff3 = np.array([[ 0.,  1.,  4.],
                      [ 1.,  0.,  1.],
                      [ 4.,  1.,  0.]])   
    alpha3 = 2
    out3 = (0,1)

    for diff, alpha, true_out in zip((diff1, diff2, diff3),
                                     (alpha1, alpha2, alpha3),
                                     (out1, out2, out3)):
        print "Input:\n{}\n".format(diff)
        print "Alpha = {}".format(alpha)

        usr_out = assignment11.findBiggestLoop(diff, alpha)

        if type(usr_out) != type(true_out):
            print ("Error: findBiggestLoop has type {}. " +
                   "Expected type is {}.").format(type(usr_out), type(true_out))
            return False            

        if usr_out != true_out:
            print ("Error: findBiggestLoop is {}. " +
                   "Expected output is {}.").format(usr_out, true_out)
            return False

        print "Current input passed."
    print "findBiggestLoop passed."
    return True

def test_synthesizeLoop():
    video_volume1 = np.array([[[[ 0,  0,  0],
                                [ 0,  0,  0]]],
                              [[[ 1,  1,  1],
                                [ 1,  1,  1]]],
                              [[[ 2,  2,  2],
                                [ 2,  2,  2]]],
                              [[[ 3,  3,  3],
                                [ 3,  3,  3]]]], dtype = np.uint8) 

    video_volume2 = np.array([[[[2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2]],
                               [[2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2]]],
                              [[[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]],
                               [[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]]],
                              [[[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]],
                               [[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]]]], dtype = np.uint8)
    frames1 = (2,3)
    frames2 = (1,1)

    out1 = [np.array([[[ 2,  2,  2],
                       [ 2,  2,  2]]], dtype = np.uint8),
            np.array([[[ 3,  3,  3],
                       [ 3,  3,  3]]], dtype = np.uint8)]

    out2 = [np.array([[[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],
                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]], dtype = np.uint8)]

    for video_volume, frames, true_out in zip((video_volume1, video_volume2), 
                                              (frames1, frames2),
                                              (out1, out2)):
        print "Input:\n{}\n".format(video_volume)
        print "Input Frame Loop:\n{}\n".format(frames)

        usr_out = assignment11.synthesizeLoop(video_volume,
                                              frames[0], frames[1])

        if type(usr_out) != type(true_out):
            print ("Error: synthesizeLoop has type {}. " + 
                   "Expected type is {}.").format(type(usr_out), type(true_out))
            return False

        if len(usr_out) != len(true_out):
            print ("Error: synthesizeLoop has len {}. " + 
                   "Expected len is {}.").format(len(usr_out), len(true_out))
            return False

        # Test each individual image.
        for usr_img, true_img in zip(usr_out, true_out):
            if type(usr_img) != type(true_img):
                print ("Error: A frame in synthesizeLoop has type {}. " + 
                       "Expected type is {}.").format(type(usr_img),
                                                      type(true_img))
                return False

            if usr_img.shape != true_img.shape:
                print ("Error: A frame in synthesizeLoop has shape {}. " + 
                       "Expected shape is {}.").format(usr_img.shape,
                                                       true_img.shape)
                return False

            if usr_img.dtype != true_img.dtype:
                print ("Error: A frame in synthesizeLoop has dtype {}. " + 
                       "Expected dtype is {}.").format(usr_img.dtype,
                                                       true_img.dtype)
                return False

            if np.all(usr_img != true_img):
                print ("Error: synthesizeLoop has value:\n{}\n" +
                       "Expected value:\n{}").format(usr_img, true_img)
                return False
        print "Current input passed."
    print "synthesizeLoop passed."
    return True

def vizDifference(diff):
    return (((diff - diff.min()) / (diff.max() - diff.min())) * 255).astype( \
        np.uint8)

def runTexture(img_list):
    """ This function administrates the extraction of a video texture from the
      given frames.
    """
    video_volume = assignment11.videoVolume(img_list)
    ssd_diff = assignment11.sumSquaredDifferences(video_volume)
    transition_diff = assignment11.transitionDifference(ssd_diff)
    alpha = 1.5*10**6
    idxs = assignment11.findBiggestLoop(transition_diff, alpha)

    diff3 = np.zeros(transition_diff.shape, float)

    for i in range(transition_diff.shape[0]): 
        for j in range(transition_diff.shape[1]): 
            diff3[i,j] = alpha*(i-j) - transition_diff[i,j] 

    return vizDifference(ssd_diff), \
           vizDifference(transition_diff), \
           vizDifference(diff3), \
           assignment11.synthesizeLoop(video_volume, idxs[0]+2, idxs[1]+2)

if __name__ == "__main__":
    print "Performing Unit Tests"
    if not test_videoVolume():
        print "videoVolume function failed. Halting testing."
        sys.exit()
    if not test_sumSquaredDifferences():
        print "sumSquaredDifferences function failed. Halting testing."
        sys.exit()
    if not test_transitionDifference():
        print "transitionDifference function failed. Halting testing."
        sys.exit()
    if not test_findBiggestLoop():
        print "findBiggestLoop function failed. Halting testing."
        sys.exit()
    if not test_synthesizeLoop():
        print "synthesizeLoop function failed. Halting testing."
        sys.exit()
    
    sourcefolder = os.path.abspath(os.path.join(os.curdir, 'videos', 'source'))
    outfolder = os.path.abspath(os.path.join(os.curdir, 'videos', 'out'))

    print 'Searching for video folders in {} folder'.format(sourcefolder)

    # Extensions recognized by opencv
    exts = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg', 
            '.jpe', '.jp2', '.tiff', '.tif', '.png']

  # For every image in the source directory
    for video_dir in os.listdir(sourcefolder):
        print "Collecting images from directory {}".format(video_dir)
        img_list = []
        filenames = sorted(os.listdir(os.path.join(sourcefolder, video_dir)))

        for filename in filenames:
            name, ext = os.path.splitext(filename)
            if ext in exts:
                img_list.append(cv2.imread(os.path.join(sourcefolder, video_dir,
                                                        filename)))
    
        print "Extracting video texture frames."
        diff1, diff2, diff3, out_list = runTexture(img_list)

        cv2.imwrite(os.path.join(outfolder, '{}diff1.png'.format(video_dir)),
                    diff1)
        cv2.imwrite(os.path.join(outfolder, '{}diff2.png'.format(video_dir)),
                    diff2)
        cv2.imwrite(os.path.join(outfolder, '{}diff3.png'.format(video_dir)),
                    diff3)

        print "writing output to {}".format(os.path.join(outfolder, video_dir))
        if not os.path.exists(os.path.join(outfolder, video_dir)):
            os.mkdir(os.path.join(outfolder, video_dir))

        for idx, image in enumerate(out_list):
            cv2.imwrite(os.path.join(outfolder,video_dir,'frame{0:04d}.png'.format(idx)), image)


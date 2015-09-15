import numpy as np
import cv2
import unittest

from assignment4 import imageGradientX
from assignment4 import imageGradientY
from assignment4 import computeGradient

class Assignment4Test(unittest.TestCase):
    def setUp(self):
        self.testImage = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)

        if self.testImage == None:
            raise IOError("Error, image test_image.jpg not found.")
    
    def test_imageGradientX(self):
        test_output = imageGradientX(self.testImage)
        self.assertEqual(type(test_output),
                         type(self.testImage))

        self.assertEqual(test_output.shape[0], self.testImage.shape[0])
        self.assertEqual(test_output.shape[1], self.testImage.shape[1] - 1)

        print "\n\nSUCCESS: imageGradientX returns the correct output type.\n"

    def test_imageGradientY(self):
        test_output = imageGradientY(self.testImage)
        self.assertEqual(type(test_output),
                         type(self.testImage))
        self.assertEqual(test_output.shape[0], self.testImage.shape[0] - 1)
        self.assertEqual(test_output.shape[1], self.testImage.shape[1])
        print "\n\nSUCCESS: imageGradientY returns the correct output type.\n"

    def test_computeGradient(self):
        avg_kernel = np.ones((3, 3)) / 9

        gradient = computeGradient(self.testImage, avg_kernel)
        # Test the output.
        self.assertEqual(type(gradient), type(self.testImage))
        # Test the column / rows are two less than input.
        self.assertEqual(gradient.shape[:2], (self.testImage.shape[0] - 2,
                                              self.testImage.shape[1] - 2))

        print "\n\nSUCCESS: computeGradient returns the correct output type.\n"

if __name__ == '__main__':
	unittest.main()

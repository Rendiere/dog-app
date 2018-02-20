import unittest
from face_detector import FaceDetector

import cv2
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class FaceDetectorTests(unittest.TestCase):

    def test_load_img(self):
        """
        Makes sure that image gets loaded in
        :return:
        """
        img_path = 'data/test-images/me.jpg'

        facedetector = FaceDetector()

        # Check that image loads successfully
        facedetector.load_img(img_path)
        self.assertIsNotNone(facedetector.img)

        # Ensure error is raised with bad image path
        with self.assertRaises(FileNotFoundError):
            facedetector.load_img('')

    def test_set_img(self):
        img_path = 'data/test-images/me.jpg'
        img = plt.imread(img_path)

        facedetector = FaceDetector()

        facedetector.set_img(img)
        self.assertIsNotNone(facedetector.img)

        # Test some incorrect types
        with self.assertRaises(TypeError):
            facedetector.set_img('')
            facedetector.set_img([1, 2, 3])
            facedetector.set_img((1, 2, 3))

        with self.assertRaises(Warning):
            facedetector.set_img(np.array([]))
            facedetector.set_img(np.array([],[1,2,3,4]))


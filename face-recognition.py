"""
Prototyping the automated prediction
"""
import cv2
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DEFAULT_CLASSIFIER = 'data/haarcascades/haarcascade_frontalface_alt.xml'


class FaceDetector():

    def __init__(self, classifier_xml=DEFAULT_CLASSIFIER):
        self.face_cascade = cv2.CascadeClassifier(classifier_xml)

    def load_img(self, img_path):
        """
        Load image from disk
        :param img_path:
        :return:
        """
        self.img = cv2.imread(img_path)

    def set_img(self, img):
        """
        Set the image to already loaded image
        :param img:
        :return:
        """
        self.img = img

    def convert_img_bgr_to_grayscale(self, img):
        """
        Convert image to grayscale
        :param img:
        :return:
        """

        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def convert_img_bgr_to_rgb(self, img):
        """
        Convert image to RGB
        :return:
        """

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def detect_faces(self, image_path=None, img=None, show=True, return_img=True, return_bboxes=False):
        """
        Detect faces from grayscale image
        :param gray_image:
        :return:
        """

        assert (return_img != return_bboxes, 'Cannot set return_img and return_bboxes both to False or True')

        if image_path:
            self.load_img(image_path)

        elif img:
            self.set_img(img)
        else:
            raise FileNotFoundError('Please pass values to either image_path or img arguments')

        gray_img = self.convert_img_bgr_to_grayscale(self.img)

        faces = self.face_cascade.detectMultiScale(gray_img)

        print(f'Found {len(faces)} faces in the image')

        if show:
            self.plot_face_boxes(faces)
            self.plot_img()

        if return_img:
            return self.img

        elif return_bboxes:
            return faces

        return faces

    def plot_face_boxes(self, faces):
        """
        Plot bounding boxes on detected faces
        :param faces:
        :return:
        """
        # For each box in detected faces, draw the box on the original image
        for (x, y, w, h) in faces:
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def plot_img(self):
        """
        Plot the image in RGB
        """

        rgb_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgb_img)
        plt.show()


if __name__ == '__main__':
    facerec = FaceDetector()

    img = facerec.detect_faces(image_path='data/test-images/me.jpg', show=True)
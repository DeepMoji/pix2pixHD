import cv2 as cv
import dlib
import bz2
import numpy as np

# LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
LANMARKS_PATH = '/Users/michaelko/Code/backup/lib/shape_predictor_68_face_landmarks.dat'

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        landmarks = []
        for detection in dets:
            try:
                face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
                # yield face_landmarks
                landmarks.append(face_landmarks)
            except:
                print("Exception in get_landmarks()!")

        return landmarks

def triangulate_landmarks(landmarks):
    points = np.array(landmarks, np.int32)
    convexhull = cv.convexHull(points)
    rect = cv.boundingRect(convexhull)
    subdiv = cv.Subdiv2D(rect)
    subdiv.insert(landmarks)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        index_pt1 = np.where((points == pt1).all(axis=1))
        # index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        # index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        # index_pt3 = extract_index_nparray(index_pt3)
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    return triangles, indexes_triangles


def process_face(img_name):
    landmarks_model_path = LANMARKS_PATH
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    # detect landmarks
    landmarks = landmarks_detector.get_landmarks(img_name)
    # Create triangles o
    # Show landmarks on top of the image
    triangles, indexes_triangles = triangulate_landmarks(landmarks[0])

    # Draw the triangles on an image
    in_img = cv.imread(img_name)

    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        cv.line(in_img, pt1, pt2, (0, 0, 255), 2)
        cv.line(in_img, pt2, pt3, (0, 0, 255), 2)
        cv.line(in_img, pt1, pt3, (0, 0, 255), 2)

        cv.imwrite('/Users/michaelko/Downloads/test_photo.png', in_img)



def sharpen_face(in_img):
    blurred = cv.GaussianBlur(in_img, (9, 9), 0)
    # sharpened = original + (original − blurred) × amount = original*(1 + amount) - blurred*amount
    final_img = cv.addWeighted(in_img, 3.0, blurred, -2.0, 0)
    return final_img

def test_bilateral():
    img_name = '/Users/michaelko/Code/ngrok/res1/comparison/images/name_modle_100_512_q57.IMG_5878.jpgno.jpg'
    img = cv.imread(img_name)
    sharpen_face(img)
    pass


if __name__ == '__main__':
    test_bilateral()

    # img_name = '/Users/michaelko/Code/ngrok/images_test_res/47023.jpg'
    # out_img_name = '/Users/michaelko/Downloads/slow_toon.jpg'
    #
    # in_img = cv.imread(img_name)
    #
    # process_face(img_name)
    #
    # final_img = sharpen_face(in_img)
    # cv.imwrite(out_img_name, final_img)

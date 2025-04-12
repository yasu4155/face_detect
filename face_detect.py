import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2, math, os
import mediapipe as mp
import mediapipe.python.solutions.drawing_styles
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

def main():
    fupload = st.sidebar.file_uploader('Upload image file')
    if fupload is None:
        st.write('Please upload an image file.')
        return
    cv2.imwrite('./target.jpg', cv2.cvtColor(np.array(Image.open(fupload.name)), cv2.COLOR_BGR2RGB))

    TARGET_IMG = './target.jpg'
    img = cv2.imread(TARGET_IMG)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#   STEP1：モデルを読み込み、検知対象の設定を行う。特徴量を計算するオブジェクト作成
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

#   STEP2：イメージオブジェクトの作成
    image = mp.Image.create_from_file(TARGET_IMG)

#   STEP 3: 顔の検出
    detection_result = detector.detect(image)

#   STEP 4: 検出した部位がわかるようにマーカーを描く
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    st.image(annotated_image)

#   STEP 5: 検出した顔に点でマスクを描く
    img2 = draw_mask_on_image(img, detection_result, TARGET_IMG)
    st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    os.remove('./target.jpg')
    return

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
      face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
      face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      face_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks])

      solutions.drawing_utils.draw_landmarks(
          image = annotated_image,
          landmark_list = face_landmarks_proto,
          connections = mp.solutions.face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec = None,
          connection_drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
    return annotated_image

def draw_mask_on_image(img, detection_result, TARGET_IMG):
    img_height, img_width, _= img.shape
    landmark_pixel_coordinates = []

    for idx, face_landmark in enumerate(detection_result.face_landmarks):
        for idx, landmark in enumerate(face_landmark):
            x_pixcel = min(math.floor(landmark.x * img_width),  img_width-1)
            y_pixcel = min(math.floor(landmark.y * img_height), img_height-1)
            nornmalized_landmark = (x_pixcel, y_pixcel)
            landmark_pixel_coordinates.append(nornmalized_landmark)
    # print(landmark_pixel_coordinates)

    coordinates_tuple = np.array(tuple(landmark_pixel_coordinates))
    img2 = cv2.imread(TARGET_IMG)

    for idx, coordinates in enumerate(coordinates_tuple):
        cv2.circle(img2, coordinates, 1, (255, 255, 0), 1)
    return img2

if __name__ == '__main__':
    main()

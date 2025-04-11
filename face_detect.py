import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import mediapipe as mp
import mediapipe.python.solutions.drawing_styles
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

def main():
    TARGET_IMG = './sample2.jpg'
    img = cv2.imread(TARGET_IMG)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#   STEP1：モデルを読み込み、検知対象の設定を行う。特徴量を計算するオブジェクト作成
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

#   STEP2：イメージオブジェクトの作成
    image = mp.Image.create_from_file(TARGET_IMG)

#   STEP 4: 顔の検出
    detection_result = detector.detect(image)

#   STEP 5: 検出した部位がわかるようにマーカーを描く
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    # cv2.imshow('image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()
    st.image(annotated_image)

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
    img = cv2.imread(TARGET_IMG)
    for idx, coordinates in enumerate(coordinates_tuple):
        cv2.circle(img, coordinates, 1, (255, 255, 0), 1)

    # cv2.imshow('image', img)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
      ])

      solutions.drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks_proto,
          connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_contours_style())

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

  # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
      plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

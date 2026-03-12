import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_draw = mp.solutions.drawing_utils

def detect_pose(frame):

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        keypoints = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]

        return keypoints, frame

    return None, frame
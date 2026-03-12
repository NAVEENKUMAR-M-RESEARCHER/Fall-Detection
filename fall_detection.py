import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Fall detection parameters
FALL_ANGLE_THRESHOLD = 45      # degrees from vertical
FALL_RATIO_THRESHOLD = 1.2     # width/height bounding box ratio
CONSECUTIVE_FRAMES   = 5       # frames to confirm fall

fall_counter = 0
fall_detected = False


def get_angle_from_vertical(shoulder, hip):
    """Calculate torso angle from vertical axis."""
    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]
    angle = abs(np.degrees(np.arctan2(dx, dy)))
    return angle


def get_bounding_box_ratio(landmarks, frame_w, frame_h):
    """Calculate width/height ratio of person bounding box."""
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]
    width  = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if height == 0:
        return 0
    return width / height


def check_fall(landmarks, frame_w, frame_h):
    """Return True if pose suggests a fall."""
    lm = landmarks.landmark

    # Get key joints
    left_shoulder  = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x  * frame_w,
                      lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y  * frame_h)
    right_shoulder = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_w,
                      lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_h)
    left_hip       = (lm[mp_pose.PoseLandmark.LEFT_HIP].x       * frame_w,
                      lm[mp_pose.PoseLandmark.LEFT_HIP].y       * frame_h)
    right_hip      = (lm[mp_pose.PoseLandmark.RIGHT_HIP].x      * frame_w,
                      lm[mp_pose.PoseLandmark.RIGHT_HIP].y      * frame_h)

    # Midpoints
    mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2)
    mid_hip      = ((left_hip[0] + right_hip[0]) / 2,
                    (left_hip[1] + right_hip[1]) / 2)

    # Feature 1: Torso angle from vertical
    torso_angle = get_angle_from_vertical(mid_shoulder, mid_hip)

    # Feature 2: Bounding box width/height ratio
    bbox_ratio = get_bounding_box_ratio(lm, frame_w, frame_h)

    # Feature 3: Shoulder Y close to Hip Y (body is horizontal)
    vertical_collapse = abs(mid_shoulder[1] - mid_hip[1]) < (frame_h * 0.15)

    fall = (torso_angle > FALL_ANGLE_THRESHOLD or bbox_ratio > FALL_RATIO_THRESHOLD
            or vertical_collapse)

    return fall, torso_angle, bbox_ratio


# ── Main Loop ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)   # 0 = webcam, or pass a video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    status_color = (0, 255, 0)   # green = safe
    status_text  = "SAFE"

    if results.pose_landmarks:
        # Draw skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS)

        is_fall, angle, ratio = check_fall(results.pose_landmarks, w, h)

        if is_fall:
            fall_counter += 1
        else:
            fall_counter = max(0, fall_counter - 1)

        # Confirm fall only after N consecutive frames
        if fall_counter >= CONSECUTIVE_FRAMES:
            fall_detected  = True
            status_color   = (0, 0, 255)   # red
            status_text    = "!! FALL DETECTED !!"
        else:
            fall_detected  = False

        # HUD info
        cv2.putText(frame, f"Torso angle : {angle:.1f} deg", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"BBox ratio  : {ratio:.2f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Status banner
    cv2.rectangle(frame, (0, 0), (w, 40), status_color, -1)
    cv2.putText(frame, status_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
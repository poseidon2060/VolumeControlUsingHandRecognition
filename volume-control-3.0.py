import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    prev_distance = 0
    volume_level = 0.0

    # change these 2 parameters to tune the model according to your needs
    sensitivity = 0.01
    increment_volume = 0.1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                distance = math.sqrt(
                    (thumb_landmark.x - index_finger_landmark.x) ** 2 +
                    (thumb_landmark.y - index_finger_landmark.y) ** 2
                )

                volume_change = distance - prev_distance

                if abs(volume_change) >= sensitivity:
                    if volume_change >= 0:
                        volume_level = min(1.0, volume_level + increment_volume)
                    else:
                        volume_level = max(0.0, volume_level - increment_volume)

                    adjust_volume(volume_level)
                    prev_distance = distance

                    bar_height = int(volume_level * 100)
                    cv2.rectangle(frame, (10, 30), (30, 130), (255, 255, 255), -1)
                    cv2.rectangle(frame, (10, 130 - bar_height), (30, 130), (0, 255, 0), -1)

        cv2.imshow('Volume Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def adjust_volume(volume1):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None
    )
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    target_volume = volume1
    volume.SetMasterVolumeLevelScalar(target_volume, None)

if __name__ == "__main__":
    main()

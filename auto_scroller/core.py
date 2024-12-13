import cv2
import mediapipe as mp
import pyautogui


class GestureDetector:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=min_detection_confidence,
                                         min_tracking_confidence=min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.last_raised_fingers = 0

    def count_raised_fingers(self, hand_landmarks):
        finger_tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        ]
        finger_dips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        ]

        raised_count = 0
        for tip, dip in zip(finger_tips, finger_dips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y:
                raised_count += 1
        return raised_count

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                raised_fingers = self.count_raised_fingers(hand_landmarks)

                if raised_fingers != self.last_raised_fingers:
                    if raised_fingers == 2:
                        pyautogui.scroll(1)
                    elif raised_fingers == 1:
                        pyautogui.scroll(-1)

                    self.last_raised_fingers = raised_fingers
        else:
            self.last_raised_fingers = 0

        return frame

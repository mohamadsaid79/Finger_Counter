import cv2 as cv
import mediapipe as mp
import time

class handLandmarkDetector():
    def __init__(self, image_mode=False, max_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.5):
        self.image_mode = image_mode
        self.max_hands = max_hands
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.image_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detect_landmarks(self, image, draw=True, draw_connections=True, draw_default_style=False):
        imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        land_mark_data = []
        hand_classified_landmarks = [[], []]
        results = self.hands.process(imageRGB)
        landmarks = results.multi_hand_landmarks
        data = None
        if landmarks:
            for hand_landmarks in landmarks:
                for id, landmark in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    px, py = int(landmark.x * w), int(landmark.y * h)
                    data = (id, px, py)
                    land_mark_data.append(data)
                if draw and not draw_connections:
                    self.mpDraw.draw_landmarks(image, hand_landmarks)
                elif draw and draw_connections and not draw_default_style:
                    self.mpDraw.draw_landmarks(image, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
                elif draw and draw_connections and draw_default_style:
                    self.mpDraw.draw_landmarks(image, hand_landmarks, self.mpHands.HAND_CONNECTIONS,
                                               self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                               self.mp_drawing_styles.get_default_hand_connections_style())
            if land_mark_data[0][1] > land_mark_data[4][1]:
                if len(land_mark_data) > 20:
                    hand_classified_landmarks[1] = land_mark_data[0:21]
                    hand_classified_landmarks[0] = land_mark_data[21::]
                else:
                    hand_classified_landmarks[1] = land_mark_data[0:21]
            elif land_mark_data[4][1] > land_mark_data[0][1]:
                if len(land_mark_data) > 20:
                    hand_classified_landmarks[0] = land_mark_data[0:21]
                    hand_classified_landmarks[1] = land_mark_data[21::]
                else:
                    hand_classified_landmarks[0] = land_mark_data[0:21]
        return hand_classified_landmarks, image

    def count_up_fingers(self, data):
        fingers = [[], []]
        if len(data[1]) != 0:
            if (data[1][3][1] > data[1][4][1]):
                fingers[1].append(1)
            else:
                fingers[1].append(0)

            if (data[1][5][2] > data[1][8][2] and data[1][7][2] > data[1][8][2]):
                fingers[1].append(1)
            else:
                fingers[1].append(0)
            if (data[1][9][2] > data[1][12][2] and data[1][11][2] > data[1][12][2]):
                fingers[1].append(1)
            else:
                fingers[1].append(0)
            if (data[1][13][2] > data[1][16][2] and data[1][15][2] > data[1][16][2]):
                fingers[1].append(1)
            else:
                fingers[1].append(0)
            if (data[1][17][2] > data[1][20][2] and data[1][19][2] > data[1][20][2]):
                fingers[1].append(1)
            else:
                fingers[1].append(0)
        if len(data[0]) != 0:
            if (data[0][3][1] < data[0][4][1]):
                fingers[0].append(1)
            else:
                fingers[0].append(0)
            if (data[0][5][2] > data[0][8][2] and data[0][7][2] > data[0][8][2]):
                fingers[0].append(1)
            else:
                fingers[0].append(0)
            if (data[0][9][2] > data[0][12][2] and data[0][11][2] > data[0][12][2]):
                fingers[0].append(1)
            else:
                fingers[0].append(0)
            if (data[0][13][2] > data[0][16][2] and data[0][15][2] > data[0][16][2]):
                fingers[0].append(1)
            else:
                fingers[0].append(0)
            if (data[0][17][2] > data[0][20][2] and data[0][19][2] > data[0][20][2]):
                fingers[0].append(1)
            else:
                fingers[0].append(0)

        return fingers


def main():
    capture = cv.VideoCapture(0)

    if not capture.isOpened():
        print("Error: Camera not found or could not be accessed.")
        return

    hand_detector = handLandmarkDetector()

    while True:
        ret, image = capture.read()
        if not ret:
            print("Error: Failed to grab image.")
            break
        image = cv.flip(image, 1)
        image = cv.resize(image, (1280, 960))

        landmarks, image = hand_detector.detect_landmarks(image, draw_default_style=False)

        fingers = hand_detector.count_up_fingers(landmarks)
        fingers_left = int(fingers[0].count(1))
        fingers_right = int(fingers[1].count(1))

        total_fingers = fingers_left + fingers_right

        cv.putText(image, f"Left Hand: {fingers_left} + Right Hand: {fingers_right} = Total: {total_fingers}",
                   (30, 190), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv.imshow("Hand Tracker", image)

        if cv.waitKey(1) == 27:
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

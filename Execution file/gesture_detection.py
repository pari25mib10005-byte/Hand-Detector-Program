import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ---------------------------
#  Vector Utils
# ---------------------------
def vector_3d(p1, p2):
    return [p2.x - p1.x, p2.y - p1.y, p2.z - p1.z]


def angle_between(v1, v2):
    dot = sum(v1[i] * v2[i] for i in range(3))
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(x * x for x in v2))
    if mag1 * mag2 == 0:
        return 0
    return math.degrees(math.acos(dot / (mag1 * mag2)))


# ---------------------------
#  Hand Gesture + Finger Count Class
# ---------------------------
class HandGestureDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def get_angles(self, hand):
        joints = [
            (0, 1, 2, 3, 4),         # Thumb
            (0, 5, 6, 7, 8),         # Index
            (0, 9, 10, 11, 12),      # Middle
            (0, 13, 14, 15, 16),     # Ring
            (0, 17, 18, 19, 20)      # Pinky
        ]

        angles = []
        for j in joints:
            v1 = vector_3d(hand.landmark[j[0]], hand.landmark[j[2]])
            v2 = vector_3d(hand.landmark[j[1]], hand.landmark[j[3]])
            angles.append(angle_between(v1, v2))

        return angles

    def classify_gesture(self, angles):
        thumb, index, middle, ring, pinky = angles

        OPEN = 45
        CLOSED = 60

        fingers_open = [
            thumb < OPEN,
            index < OPEN,
            middle < OPEN,
            ring < OPEN,
            pinky < OPEN
        ]

        count = sum(fingers_open)

        # Gestures
        if count == 5:
            gesture = "Open Palm "
        elif count == 0:
            gesture = "Fist "
        elif thumb < OPEN and all(f >= CLOSED for f in [index, middle, ring, pinky]):
            gesture = "Thumbs Up "
        elif index < OPEN and all(f >= CLOSED for f in [middle, ring, pinky]):
            gesture = "Point "
        elif index < OPEN and middle < OPEN and ring > CLOSED and pinky > CLOSED:
            gesture = "Victory "
        else:
            gesture = "Unknown"

        return gesture, count

    # Draw results on frame
    def process(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                angles = self.get_angles(hand_landmarks)
                gesture, count = self.classify_gesture(angles)

                wrist = hand_landmarks.landmark[0]
                x, y = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])

                cv2.putText(frame, f"{gesture}", (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"Fingers: {count}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


# ---------------------------
#  Main App (Runs Camera)
# ---------------------------
def run():
    cap = cv2.VideoCapture(0)
    detector = HandGestureDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.hands.process(rgb)

        # Process hands
        detector.process(frame, results)

        # Add your name on screen
        cv2.putText(frame, "PARI JAIN REG NO: 25MIB10005", (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        cv2.imshow("Advanced Hand Detector", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()

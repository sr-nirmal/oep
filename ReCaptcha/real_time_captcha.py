import random
import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import threading
cap = cv2.VideoCapture(0)
cap.set(3, 800)  # Width
cap.set(4, 800)  # Height

detector = HandDetector(maxHands=1)

startGame = False
initialTime = 0
stateResult = False
requiredGesture = None
timeLimit = 5


def generate_required_gesture():
    return random.randint(1, 5)


while True:
    success, img = cap.read()

    if not success:
        break

    hands, img = detector.findHands(img)

    if startGame:

        timer = time.time() - initialTime
        remainingTime = timeLimit - timer

        cv2.putText(img, f"Time Left: {int(remainingTime)}", (150, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)

        if remainingTime <= 0:
            cv2.putText(img, "Rejected! Time Over", (100, 400), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)
            cv2.imshow("Hand Gesture CAPTCHA", img)
            cv2.waitKey(2000)
            break

        if hands and stateResult is False:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            fingersCount = fingers.count(1)

            cv2.putText(img, f'Fingers: {fingersCount}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

            if fingersCount == requiredGesture:
                # Correct gesture detected within the time
                cv2.putText(img, "Accepted!", (100, 400), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 4)
                cv2.imshow("Hand Gesture CAPTCHA", img)
                cv2.waitKey(2000)
                startGame = False

    else:
        cv2.putText(img, "Press SPACE to start", (150, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 4)

    if requiredGesture is not None and startGame:
        cv2.putText(img, f'Show {requiredGesture} fingers', (200, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 4)

    cv2.imshow("Hand Gesture CAPTCHA", img)

    key = cv2.waitKey(1)

    if key == ord(' '):
        startGame = True
        requiredGesture = generate_required_gesture()
        initialTime = time.time()
        stateResult = False

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

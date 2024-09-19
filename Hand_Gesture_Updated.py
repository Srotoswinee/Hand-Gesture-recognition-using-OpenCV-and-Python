import cv2
import numpy as np
import math
import webbrowser
import time

def perform_action(gesture):
    """Perform actions based on recognized gestures."""
    actions = {
        0: "Put hand in the box",
        1: "Open Google",
        2: "Open YouTube",
        3: "Open Facebook",
        4: "Open Twitter",
        5: "Open LinkedIn"
    }
    if gesture in actions:
        print(actions[gesture])
        if gesture == 1:
            webbrowser.open("https://www.google.com")
        elif gesture == 2:
            webbrowser.open("https://www.youtube.com")
        elif gesture == 3:
            webbrowser.open("https://www.facebook.com")
        elif gesture == 4:
            webbrowser.open("https://www.twitter.com")
        elif gesture == 5:
            webbrowser.open("https://www.linkedin.com")
    else:
        print("Unknown gesture")

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

gesture_last_action_time = time.time()
delay = 10  # Delay in seconds

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        kernel = np.ones((3, 3), np.uint8)
        # Increase the size of the ROI by adjusting coordinates
        roi = frame[80:340, 80:340]  # Adjusted coordinates for a larger ROI
        cv2.rectangle(frame, (80, 80), (340, 340), (0, 255, 0), 2)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=lambda x: cv2.contourArea(x))
            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            hull = cv2.convexHull(approx, returnPoints=False)
            areahull = cv2.contourArea(cv2.convexHull(cnt))
            areacnt = cv2.contourArea(cnt)
            arearatio = ((areahull - areacnt) / areacnt) * 100
            defects = cv2.convexityDefects(approx, hull)
            l = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    s = (a + b + c) / 2
                    ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
                    d = (2 * ar) / a
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
                    if angle <= 90 and d > 30:
                        l += 1
                        cv2.circle(roi, far, 3, [255, 0, 0], -1)
                    cv2.line(roi, start, end, [0, 255, 0], 2)

            # Gesture recognition
            font = cv2.FONT_HERSHEY_SIMPLEX
            if l == 1:
                if areacnt < 2000:
                    gesture_text = 'Put hand in the box'
                else:
                    if arearatio < 8:  # Adjusted threshold for gesture 1
                        gesture_text = '0'
                    elif arearatio < 15:  # Adjusted threshold for gesture 1
                        gesture_text = 'Best of luck'
                    else:
                        gesture_text = '1'
            elif l == 2:
                gesture_text = '2'
            elif l == 3:
                if arearatio < 12:  # Decreased threshold for gesture 3
                    gesture_text = '3'
                else:
                    gesture_text = 'ok'
            elif l == 4:
                if arearatio < 16:  # Decreased threshold for gesture 4
                    gesture_text = '4'
                else:
                    gesture_text = 'reposition'
            elif l == 5:
                if arearatio < 20:  # Further decreased threshold for gesture 5
                    gesture_text = '5'
                else:
                    gesture_text = 'reposition'
            else:
                gesture_text = 'reposition'

            cv2.putText(frame, gesture_text, (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

            # Perform action based on the recognized gesture
            current_time = time.time()
            if l in [1, 2, 3, 4, 5]:
                if current_time - gesture_last_action_time > delay:  # Check delay
                    perform_action(l)
                    gesture_last_action_time = current_time

        # Display the resulting frame and mask
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)

    except Exception as e:
        print(f"An error occurred: {e}")
        pass

    # Exit loop when 'Esc' key is pressed
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Release resources
cv2.destroyAllWindows()
cap.release()

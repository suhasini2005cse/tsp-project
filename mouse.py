import streamlit as st
import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Streamlit app UI
st.title("Gesture-Controlled Virtual Mouse")
st.write("Use your hand gestures to control the mouse!")

# Webcam feed setup
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image(["tap.png"])

# Screen dimensions
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Could not access webcam. Please check your settings.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks for the index finger (tip and base)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            # Map landmarks to screen dimensions
            cursor_x = int(index_tip.x * screen_width)
            cursor_y = int(index_tip.y * screen_height)

            # Move the mouse cursor
            pyautogui.moveTo(cursor_x, cursor_y)

            # Detect click gesture (distance between index_tip and index_base)
            distance = abs(index_tip.y - index_base.y)
            if distance < 0.05:  # Adjust threshold as needed
                pyautogui.click()

    FRAME_WINDOW.image(frame)

else:
    cap.release()
    cv2.destroyAllWindows()
    st.write("Webcam stopped.")

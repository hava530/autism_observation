import streamlit as st
import time
import cv2
import mediapipe as mp
import speech_recognition as sr
import csv
from datetime import datetime

st.title("🧠 Autism Observation Interface (Real-Time)")

if "results" not in st.session_state:
    st.session_state.results = {}

# CSV Writer
def write_csv(data):
    file_name = "autism_observation_results.csv"
    with open(file_name, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Date", "Measurement", "Result"])
        for key, value in data.items():
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), key, value])

# 1. Language Development
def count_words_from_speech(duration_seconds=10):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info(f"Speak for {duration_seconds} seconds...")
        audio = recognizer.listen(source, phrase_time_limit=duration_seconds)
        try:
            text = recognizer.recognize_google(audio, language="tr-TR")
            words = text.split()
            return len(words), text
        except:
            return 0, ""

# 2. Stereotypical Behavior
def detect_stereotypical_behavior(record_duration=10):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(0)
    movement_count = 0
    previous_y = None
    start_time = time.time()

    while time.time() - start_time < record_duration:
        ret, frame = cap.read()
        if not ret:
            break
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
            if previous_y and abs(y - previous_y) > 0.05:
                movement_count += 1
            previous_y = y

        cv2.imshow("Stereotypical Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return movement_count

# 3. Attention Span (Eye Tracking)
def measure_attention(duration=10):
    face_mesh = mp.solutions.face_mesh.FaceMesh()
    cap = cv2.VideoCapture(0)
    attentive_time = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            attentive_time += 1
        cv2.imshow("Attention Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return attentive_time

# 4. Motor Skills (Hand Tracking)
def detect_hands(duration=10):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    cap = cv2.VideoCapture(0)
    detection_count = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            detection_count += 1
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return detection_count

# 5. Eye Contact (Basic Face Detection)
def detect_eye_contact(duration=10):
    face_mesh = mp.solutions.face_mesh.FaceMesh()
    cap = cv2.VideoCapture(0)
    contact_duration = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            contact_duration += 1
        cv2.imshow("Eye Contact", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return contact_duration

# --- Streamlit Buttons ---

st.subheader("🎤 Language Development")
if st.button("Start Speech"):
    word_count, recognized_text = count_words_from_speech()
    st.success(f"Number of words detected: {word_count}")
    if recognized_text:
        st.write(f"📄 Recognized speech: {recognized_text}")
    st.session_state.results["Language Development (word count)"] = word_count

st.subheader("🤸‍♂️ Stereotypical Behavior")
if st.button("Start Movement Tracking"):
    movements = detect_stereotypical_behavior()
    st.success(f"Number of movements detected: {movements}")
    st.session_state.results["Stereotypical Movements"] = movements

st.subheader("👁️ Attention Span")
if st.button("Start Attention Measurement"):
    attention = measure_attention()
    st.success(f"Attention duration (seconds): {attention}")
    st.session_state.results["Attention Span"] = attention

st.subheader("🖐️ Motor Skills (Hand Tracking)")
if st.button("Start Hand Tracking"):
    hand_movements = detect_hands()
    st.success(f"Number of hand detections: {hand_movements}")
    st.session_state.results["Motor Skills (hand detected)"] = hand_movements

st.subheader("👀 Eye Contact")
if st.button("Measure Eye Contact"):
    eye_contact = detect_eye_contact()
    st.success(f"Eye contact duration (seconds): {eye_contact}")
    st.session_state.results["Eye Contact"] = eye_contact

st.markdown("---")
if st.button("📊 Show and Save Results"):
    if st.session_state.results:
        st.success("🔍 Measurement Results:")
        for key, value in st.session_state.results.items():
            st.write(f"• *{key}*: {value}")
        write_csv(st.session_state.results)
        st.success("💾 Saved to CSV.")
    else:
        st.warning("No measurements have been taken yet.")

import streamlit as st
import cv2
from ultralytics import YOLO
import time
import os

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# ---------------- PAGE SETUP ----------------
st.set_page_config(layout="wide")
st.title("🚦 Intelligent Traffic Management System 🚦")

# ---------------- WEATHER MODE ----------------
mode = st.selectbox(
    "🌦 Traffic Environment Mode",
    ["Day", "Evening", "Winter / Fog", "Night"]
)

mode_filter = {
    "Day": None,
    "Evening": "evening",
    "Winter / Fog": "fog",
    "Night": "night"
}

# ---------------- CSS ----------------
st.markdown("""
<style>
.video img {
    border: 3px solid white;
    border-radius: 10px;
}
.count-box {
    border: 2px solid #00e5ff;
    border-radius: 10px;
    padding: 8px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    color: #00e5ff;
    margin-top: 6px;
}
.signal-card {
    border-radius: 16px;
    padding: 14px;
    color: white;
    text-align: center;
    font-weight: bold;
    font-size: 20px;
    margin-top: 6px;
}
.green { background:#00c853; box-shadow:0 0 25px #00e676; }
.yellow { background:#ffc400; box-shadow:0 0 20px #ffd740; color:black; }
.red { background:#d50000; }
.timer { font-size:14px; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
st.session_state.setdefault("run", False)
st.session_state.setdefault("order", None)
st.session_state.setdefault("index", 0)
st.session_state.setdefault("start_time", time.time())

# ---------------- YOLO ----------------
model = YOLO("yolov8n.pt")
vehicle_classes = ["car", "bus", "truck", "motorcycle"]

# ---------------- GREEN TIME ----------------
def get_green_time(count):
    if count <= 5:
        return 10
    elif count <= 12:
        return 20
    else:
        return 30

# ---------------- VIDEO SOURCES ----------------
lanes = {
    "Lane 1": "traffic-videos/lane1.mp4",
    "Lane 2": "traffic-videos/lane2.mp4",
    "Lane 3": "traffic-videos/lane3.mp4",
    "Lane 4": "traffic-videos/lane4.mp4",
}
caps = {l: cv2.VideoCapture(p) for l, p in lanes.items()}

# ---------------- CONTROLS ----------------
c1, c2 = st.columns(2)
with c1:
    if st.button("▶ Start"):
        st.session_state.run = True
        st.session_state.order = None
        st.session_state.index = 0
        st.session_state.start_time = time.time()

with c2:
    if st.button("⏸ Pause"):
        st.session_state.run = False

# ---------------- UI LAYOUT ----------------
video_cols = st.columns(4)
count_cols = st.columns(4)
signal_cols = st.columns(4)

video_box, count_box, signal_box = {}, {}, {}

for i, lane in enumerate(lanes):
    video_box[lane] = video_cols[i].empty()
    count_box[lane] = count_cols[i].empty()
    signal_box[lane] = signal_cols[i].empty()

# ---------------- MAIN LOOP ----------------
if st.session_state.run:
    while st.session_state.run:
        counts = {}

        # ---- READ & DETECT ----
        for lane, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            frame = cv2.resize(frame, (400, 250))

            # Weather visual simulation
            if mode_filter[mode] == "night":
                frame = cv2.convertScaleAbs(frame, alpha=0.5, beta=-30)
            elif mode_filter[mode] == "fog":
                frame = cv2.GaussianBlur(frame, (15, 15), 0)
            elif mode_filter[mode] == "evening":
                frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=20)

            res = model(frame, verbose=False)[0]

            c = 0
            for box in res.boxes:
                if model.names[int(box.cls[0])] in vehicle_classes:
                    c += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            counts[lane] = c

            video_box[lane].image(frame, channels="BGR", use_container_width=True)
            count_box[lane].markdown(
                f"<div class='count-box'>🚗 Vehicles: {c}</div>",
                unsafe_allow_html=True
            )

        # ---- SIGNAL ORDER ----
        if st.session_state.order is None:
            st.session_state.order = sorted(counts, key=counts.get, reverse=True)

        order = st.session_state.order
        idx = st.session_state.index

        green_lane = order[idx]
        yellow_lane = order[(idx + 1) % len(order)]

        green_time = get_green_time(counts[green_lane])
        elapsed = int(time.time() - st.session_state.start_time)
        remaining = green_time - elapsed

        if remaining <= 0:
            st.session_state.index = (idx + 1) % len(order)
            st.session_state.start_time = time.time()
            if st.session_state.index == 0:
                st.session_state.order = None

        # ---- SIGNAL DISPLAY ----
        for lane in lanes:
            if lane == green_lane:
                signal_box[lane].markdown(
                    f"<div class='signal-card green'>🟢 GREEN"
                    f"<div class='timer'>{remaining}s / {green_time}s</div></div>",
                    unsafe_allow_html=True
                )
            elif lane == yellow_lane:
                signal_box[lane].markdown(
                    "<div class='signal-card yellow'>🟡 YELLOW</div>",
                    unsafe_allow_html=True
                )
            else:
                signal_box[lane].markdown(
                    "<div class='signal-card red'>🔴 RED</div>",
                    unsafe_allow_html=True
                )

        time.sleep(0.3)

# ---------------- CLEANUP ----------------
if not st.session_state.run:
    for cap in caps.values():
        cap.release()

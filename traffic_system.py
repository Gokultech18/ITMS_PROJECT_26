import streamlit as st
import cv2
import time
import os
from ultralytics import YOLO

# ---------------- SAFE ENV ----------------
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# ---------------- PAGE SETUP ----------------
st.set_page_config(layout="wide")
st.title("🚦 Intelligent Traffic Management System")

# ---------------- WEATHER MODE ----------------
mode = st.selectbox(
    "🌦 Traffic Environment Mode",
    ["Day", "Evening", "Winter / Fog", "Night"]
)

MODE_STYLE = {
    "Day": "☀️ Normal visibility",
    "Evening": "🌇 Medium visibility",
    "Winter / Fog": "🌫 Low visibility",
    "Night": "🌙 Very low visibility"
}

st.info(f"Current Mode: **{mode}** — {MODE_STYLE[mode]}")

# ---------------- CSS ----------------
st.markdown("""
<style>
.video-box img {
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
    border-radius: 14px;
    padding: 12px;
    color: white;
    text-align: center;
    font-weight: bold;
    font-size: 18px;
    margin-top: 6px;
}

.green { background:#00c853; }
.yellow { background:#ffc400; color:black; }
.red { background:#d50000; }
.timer { font-size:14px; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "run" not in st.session_state:
    st.session_state.run = False
if "index" not in st.session_state:
    st.session_state.index = 0
if "order" not in st.session_state:
    st.session_state.order = None
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# ---------------- YOLO (CACHED) ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
VEHICLES = ["car", "bus", "truck", "motorcycle"]

# ---------------- GREEN TIME LOGIC ----------------
def get_green_time(count):
    if count <= 5:
        return 10
    elif count <= 12:
        return 20
    else:
        return 30

# ---------------- VIDEO SOURCES (CLOUD SAFE) ----------------
lanes = {
    "Lane 1": "traffic-videos/lane1.mp4",
    "Lane 2": "traffic-videos/lane2.mp4",
    "Lane 3": "traffic-videos/lane3.mp4",
    "Lane 4": "traffic-videos/lane4.mp4",
}

caps = {lane: cv2.VideoCapture(path) for lane, path in lanes.items()}

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

# ---------------- UI GRID ----------------
video_cols = st.columns(4)
count_cols = st.columns(4)
signal_cols = st.columns(4)

video_box, count_box, signal_box = {}, {}, {}

for i, lane in enumerate(lanes):
    video_box[lane] = video_cols[i].empty()
    count_box[lane] = count_cols[i].empty()
    signal_box[lane] = signal_cols[i].empty()

# ---------------- MAIN LOGIC (NO INFINITE LOOP) ----------------
if st.session_state.run:
    counts = {}

    for lane, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        frame = cv2.resize(frame, (400, 250))
        results = model(frame, verbose=False)[0]

        count = 0
        for box in results.boxes:
            if model.names[int(box.cls[0])] in VEHICLES:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        counts[lane] = count

        video_box[lane].image(frame, channels="BGR", use_container_width=True)
        count_box[lane].markdown(
            f"<div class='count-box'>🚗 Vehicles: {count}</div>",
            unsafe_allow_html=True
        )

    # -------- SIGNAL ORDER --------
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

    # -------- SIGNAL DISPLAY --------
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
    st.experimental_rerun()

# ---------------- CLEANUP ----------------
if not st.session_state.run:
    for cap in caps.values():
        cap.release()
while st.session_state.run:
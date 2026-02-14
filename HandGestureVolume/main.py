import cv2
import mediapipe as mp
import numpy as np
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time

# ================= CAMERA =================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ================= MEDIAPIPE =================
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

# ================= AUDIO =================
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volMin, volMax, _ = volume.GetVolumeRange()

# ================= SETTINGS =================
MUTE_DISTANCE = 25
UNMUTE_DISTANCE = 40
MIN_VOL_PERCENT = 20
MAX_VOL_PERCENT = 100
MAX_HAND_DISTANCE = 150

HOLD_EXIT_TIME = 2.0
STABLE_TIME = 2.0
DISTANCE_TOLERANCE = 5

# ================= STATE VARIABLES =================
isMuted = False
lastAction = None
palm_start_time = None

last_length = None
stable_start_time = None
volume_locked = False

# ================= LOOP =================
while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                lmList.append([id, int(lm.x * w), int(lm.y * h)])
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if lmList:
        # -------- LANDMARKS --------
        x1, y1 = lmList[4][1], lmList[4][2]   # thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]   # index tip
        x3, y3 = lmList[6][1], lmList[6][2]   # index PIP

        cv2.circle(img, (x1, y1), 6, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 6, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = hypot(x2 - x1, y2 - y1)

        # -------- FINGER STATES --------
        thumb_open  = lmList[4][1]  > lmList[3][1]
        index_open  = lmList[8][2]  < lmList[6][2]
        middle_open = lmList[12][2] < lmList[10][2]
        ring_open   = lmList[16][2] < lmList[14][2]
        pinky_open  = lmList[20][2] < lmList[18][2]

        # -------- EXIT (5 FINGERS HOLD) --------
        if thumb_open and index_open and middle_open and ring_open and pinky_open:
            if palm_start_time is None:
                palm_start_time = time.time()
            elif time.time() - palm_start_time >= HOLD_EXIT_TIME:
                break
        else:
            palm_start_time = None

        # -------- UNLOCK (PINKY ONLY) --------
        if pinky_open and not index_open and not middle_open and not ring_open:
            volume_locked = False
            last_length = None
            stable_start_time = None

        # -------- MUTE / UNMUTE --------
        index_finger_open = y2 < y3
        if index_finger_open:
            if length < MUTE_DISTANCE and lastAction != "mute":
                volume.SetMute(1, None)
                isMuted = True
                lastAction = "mute"

            elif length > UNMUTE_DISTANCE and lastAction != "unmute":
                volume.SetMute(0, None)
                isMuted = False
                lastAction = "unmute"

        # -------- VOLUME CONTROL WITH LOCK --------
        if not isMuted and not volume_locked:
            length = np.clip(length, UNMUTE_DISTANCE, MAX_HAND_DISTANCE)

            volPercent = np.interp(
                length,
                [UNMUTE_DISTANCE, MAX_HAND_DISTANCE],
                [MIN_VOL_PERCENT, MAX_VOL_PERCENT]
            )

            volLevel = np.interp(volPercent, [0, 100], [volMin, volMax])
            volume.SetMasterVolumeLevel(volLevel, None)

            # Stability check
            if last_length is None:
                last_length = length
                stable_start_time = time.time()
            else:
                if abs(length - last_length) <= DISTANCE_TOLERANCE:
                    if time.time() - stable_start_time >= STABLE_TIME:
                        volume_locked = True
                else:
                    last_length = length
                    stable_start_time = time.time()

        # -------- UI --------
        status = (
            "VOLUME LOCKED 🔒" if volume_locked
            else "MUTED 🔇" if isMuted
            else "UNMUTED 🔊"
        )

        cv2.putText(
            img, status,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255) if volume_locked else ((0, 0, 255) if isMuted else (0, 255, 0)),
            3
        )

    cv2.imshow("Hand Gesture Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()

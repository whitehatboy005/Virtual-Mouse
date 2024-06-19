import cv2
import mediapipe as mp
import pyautogui
import time

# Constants for drawing
CIRCLE_RADIUS = 5
CIRCLE_COLOR = (0, 255, 0)  # Green
CIRCLE_THICKNESS = -1  # Filled circle
LINE_COLOR = (0, 255, 0)  # Green
LINE_THICKNESS = 2
SCALING_FACTOR = 5.0  # Factor to amplify the cursor movement


def init_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video device.")
    return cap


def process_frame(frame):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, rgb_frame


def draw_landmarks(frame, hands, drawing_utils):
    for hand in hands:
        drawing_utils.draw_landmarks(frame, hand)
        landmarks = hand.landmark
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)
        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_x = int(landmarks[start_idx].x * frame.shape[1])
            start_y = int(landmarks[start_idx].y * frame.shape[0])
            end_x = int(landmarks[end_idx].x * frame.shape[1])
            end_y = int(landmarks[end_idx].y * frame.shape[0])
            cv2.line(frame, (start_x, start_y), (end_x, end_y), LINE_COLOR, LINE_THICKNESS)
    return landmarks


def get_landmark_coordinates(landmarks, frame_width, frame_height):
    coords = {}
    for id, landmark in enumerate(landmarks):
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        coords[id] = (x, y)
    return coords


def map_to_screen(coords, screen_width, screen_height, frame_width, frame_height):
    mapped_coords = {}
    for id, (x, y) in coords.items():
        mapped_x = screen_width * x / frame_width
        mapped_y = screen_height * y / frame_height
        mapped_coords[id] = (mapped_x, mapped_y)
    return mapped_coords


def move_cursor(index_coords, plocx, plocy, smoothening):
    index_x, index_y = index_coords
    clocx = plocx + (index_x - plocx) / smoothening
    clocy = plocy + (index_y - plocy) / smoothening

    # Apply scaling factor for more sensitive movement
    clocx = plocx + (clocx - plocx) * SCALING_FACTOR
    clocy = plocy + (clocy - plocy) * SCALING_FACTOR

    pyautogui.moveTo(clocx, clocy)
    return clocx, clocy


def detect_gestures(coords, thumb_coords, click_time, click_threshold, single_click_flag, left_dragging):
    thumb_x, thumb_y = thumb_coords

    # Left click
    if abs(coords[8][1] - thumb_y) < 70:
        current_time = time.time()
        if current_time - click_time < click_threshold:
            pyautogui.doubleClick()
            click_time = 0
        else:
            if not single_click_flag:
                pyautogui.click()
                single_click_flag = True
            click_time = current_time
    else:
        single_click_flag = False

    # Drag
    if abs(coords[16][1] - thumb_y) < 70:
        if not left_dragging:
            pyautogui.mouseDown()
            left_dragging = True
    else:
        if left_dragging:
            pyautogui.mouseUp()
            left_dragging = False

    # Right click
    if abs(coords[12][1] - thumb_y) < 70:
        pyautogui.rightClick()
        pyautogui.sleep(1)

    # Scroll
    extended_fingers = [id for id in [8, 12, 16, 20] if coords[id][1] < coords[id - 2][1]]

    if len(extended_fingers) == 0:  # Only thumb is extended
        thumb_tip_y = coords[4][1]
        wrist_y = coords[0][1]

        if thumb_tip_y < wrist_y - 40:  # Thumbs up gesture
            pyautogui.scroll(200)  # Increase this value to scroll faster
        elif thumb_tip_y > wrist_y + 40:  # Thumbs down gesture
            pyautogui.scroll(-200)  # Increase this value to scroll faster

    return click_time, single_click_flag, left_dragging


def add_user_instructions(frame):
    instructions = [
        "Virtual Mouse Instructions:",
        "1. Move cursor: Use Index Finger",
        "2. Left Click: Bring Thumb close to Index Finger",
        "3. Right Click: Bring Thumb close to Middle Finger",
        "4. Drag: Hold Thumb close to Ring Finger",
        "5. Scroll: Thumbs Up to Scroll Up, Thumbs Down to Scroll Down"
    ]
    y0, dy = 20, 30
    for i, line in enumerate(instructions):
        y = y0 + i * dy
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    cap = init_webcam()
    hand_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    smoothening = 7
    plocx, plocy = 0, 0
    click_time = 0
    click_threshold = 0.3
    single_click_flag = False
    left_dragging = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame, rgb_frame = process_frame(frame)
        frame_height, frame_width, _ = frame.shape
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            landmarks = draw_landmarks(frame, hands, drawing_utils)
            coords = get_landmark_coordinates(landmarks, frame_width, frame_height)
            mapped_coords = map_to_screen(coords, screen_width, screen_height, frame_width, frame_height)
            clocx, clocy = move_cursor(mapped_coords[8], plocx, plocy, smoothening)
            plocx, plocy = clocx, clocy

            click_time, single_click_flag, left_dragging = detect_gestures(
                mapped_coords, mapped_coords[4], click_time, click_threshold, single_click_flag, left_dragging
            )

        add_user_instructions(frame)
        cv2.imshow('Virtual Mouse', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

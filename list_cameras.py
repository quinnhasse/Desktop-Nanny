import cv2

def find_available_cameras(max_test=5):
    available = []
    for index in range(max_test):
        cap = cv2.VideoCapture(index)
        if cap is not None and cap.isOpened():
            available.append(index)
            cap.release()
    return available

available_cameras = find_available_cameras()
print(f"Available camera indices: {available_cameras}")
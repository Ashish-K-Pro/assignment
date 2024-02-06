
import cv2

def SceneDetectionAlgo(video_path, threshold=0.1):
    scene_boundaries = []

    cap = cv2.VideoCapture(video_path)
    prev_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256])

        if prev_frame is not None:
            prev_frame_hist = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
            similarity = cv2.compareHist(frame_hist, prev_frame_hist, cv2.HISTCMP_CORREL)

            if similarity < threshold:
                scene_boundaries.append(cap.get(cv2.CAP_PROP_POS_MSEC))

        prev_frame = frame_gray

    cap.release()
    return scene_boundaries


# Example usage:
myVideoPath = '/Users/ashishkumar/Desktop/myVideo.mp4'
scene_boundaries = SceneDetectionAlgo(myVideoPath, threshold=2.1)
print(scene_boundaries)
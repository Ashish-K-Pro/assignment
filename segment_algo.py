import cv2


# the function for key frame segmentations.
def thekeyframesExtraction(video_path, threshold=0.1):
    # variable to store datasets
    keyframes = []
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    # checking till the video's end
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
                keyframes.append(frame)

        prev_frame = frame_gray

    cap.release()
    return keyframes


# Example usage for the function:
myVideoPath = '/Users/ashishkumar/Desktop/myVideo.mp4'
extracted_keyframes = thekeyframesExtraction(myVideoPath, threshold=2.1)
print(extracted_keyframes)
import cv2

def select_roi(video_path):
    """
    Opens the first frame of the video and allows the user to manually
    select the bounding box (ROI) using the mouse.
    """

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Frame 1
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    # ROI selection
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)

    cap.release()
    cv2.destroyAllWindows()

    # Print ROI coordinates (x, y, width, height)
    print(f"Selected bounding box: x, y, w, h = ",roi[0],roi[1],roi[2],roi[3])
    return roi


video_file = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT video\LFT_dry_SVG_test1.mp4" #Replace with current test path

bounding_box = select_roi(video_file)

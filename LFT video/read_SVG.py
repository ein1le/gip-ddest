import cv2
import pytesseract
import csv
import os

def extract_number_from_frame(frame):
    
    # frame preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    # gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1] # grayscale thresholding (if needed)

    # ROI definition
    x, y, w, h = 143,171, 542, 157 #replace with bounding_box values
    roi = gray[y:y+h, x:x+w]

    # OCR
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    custom_config = r'--psm 6'
    text = pytesseract.image_to_string(roi, config=custom_config)

    # string post-processing
    text = text.strip()
    return text

def record_video_numbers(video_path, output_csv, sample_frequency=1.0):

    # File check
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return

    # Fetch basic properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0

    # Frames to skip based on sampling frequency
    frames_to_skip = int(sample_frequency * fps) if fps else 0

    # Print video properties
    print(f"Video FPS: {fps}")
    print(f"Total Frames: {frame_count}")
    print(f"Video Duration (sec): {duration}")
    print(f"Sampling every {sample_frequency} second(s) => {frames_to_skip} frames")

    # Prepare CSV writing
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["index", "timestamp_sec", "strain_value"])

        current_frame_idx = 0
        sample_index = 0
        
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)

            ret, frame = cap.read()
            if not ret:
                break

            # Compute the exact timestamp (in seconds)
            timestamp = current_frame_idx / fps if fps else 0

            # Extract the displayed number from the frame
            number_str = extract_number_from_frame(frame)

            # Write to CSV
            writer.writerow([sample_index, f"{timestamp:.2f}", number_str])

            sample_index += 1
            current_frame_idx += frames_to_skip


            if current_frame_idx >= frame_count:
                break

    # Release the video capture
    cap.release()
    print(f"Completed. Saved to {output_csv}.")

# Usage example:
if __name__ == "__main__":
    video_file =  r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT video\LFT_dry_SVG_test1.mp4" #Replace with current test path
    output_csv_file = "test1.csv"
    sample_freq = 2.0  # in seconds; adjust as needed

    record_video_numbers(video_file, output_csv_file, sample_freq)

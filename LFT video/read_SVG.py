import cv2
import pytesseract
import csv
import os

# ─── CONFIGURABLE VARIABLES ─────────────────────────────────────────────────────
video_path = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT video\LFT_PTFE_1_vid.mp4"
x, y, w, h = 124, 228, 315, 111  
psm = 6 
sample_freq = 1.0 
output_csv_file = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT video\LFT_PTFE_1_vid.csv"
# ────────────────────────────────────────────────────────────────────────────────

def extract_number_from_frame(frame, roi_x, roi_y, roi_w, roi_h, psm_val):
    """
    Extracts the number from a video frame using the specified ROI and PSM mode.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Debug: Draw ROI on a copy of the frame
    debug_frame = frame.copy()
    cv2.rectangle(debug_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h),
                  (0, 255, 0), 2)  # Green box around ROI

    # Extract ROI
    roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # Tesseract configuration
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    custom_config = f'--oem 3 --psm {psm_val} -c tessedit_char_whitelist=-0123456789'

    text = pytesseract.image_to_string(roi, config=custom_config).strip()
    print(f"Extracted Text: '{text}'") 

    # Debug 
    script_directory = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT video debug"
    roi_debug_gray_path = os.path.join(script_directory, "roi_debug_gray.png")
    roi_debug_path = os.path.join(script_directory, "roi_debug.png")
    roi_drawn_path = os.path.join(script_directory, "roi_drawn.png")

    cv2.imwrite(roi_debug_gray_path, gray)
    cv2.imwrite(roi_debug_path, roi, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(roi_drawn_path, debug_frame)

    return text

def record_video_numbers(video_file_path, output_csv, roi_x, roi_y, roi_w, roi_h, psm_val, sample_frequency=1.0):
    """
    Records recognized numbers from a video at a given sampling frequency
    and writes them to a CSV file.
    """
    # Validate the video file
    if not os.path.exists(video_file_path):
        print(f"❌ Video file not found: {video_file_path}")
        return

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print(f"❌ Could not open video file: {video_file_path}")
        return

    # Fetch basic properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0

    # Calculate frames to skip based on sampling frequency
    frames_to_skip = int(sample_frequency * fps) if fps else 0

    # Print video properties for debugging
    print("─────────────────────────────────────────")
    print(f"Video File:      {video_file_path}")
    print(f"Video FPS:       {fps}")
    print(f"Total Frames:    {frame_count}")
    print(f"Video Duration:  {duration:.2f} sec")
    print(f"Sample Frequency {sample_frequency} sec")
    print(f"Frames to Skip:  {frames_to_skip}")
    print("─────────────────────────────────────────")

    # Prepare CSV writing
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["index", "timestamp_sec", "strain_value"])

        current_frame_idx = 0
        sample_index = 0

        while True:
            # Set the current frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()

            if not ret:
                break  # No more frames or read failure

            # Compute the exact timestamp (in seconds)
            timestamp = current_frame_idx / fps if fps else 0

            # Extract the displayed number from the frame
            number_str = extract_number_from_frame(frame, roi_x, roi_y, roi_w, roi_h, psm_val)

            # Write to CSV
            writer.writerow([sample_index, f"{timestamp:.2f}", number_str])

            sample_index += 1
            current_frame_idx += frames_to_skip

            if current_frame_idx >= frame_count:
                break

    cap.release()
    print(f"\n✅ Completed. Saved to {output_csv}")

# ────────────────────────────────────────────────────────────────────────────────
# SCRIPT ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    record_video_numbers(
        video_file_path=video_path,
        output_csv=output_csv_file,
        roi_x=x,
        roi_y=y,
        roi_w=w,
        roi_h=h,
        psm_val=psm,
        sample_frequency=sample_freq
    )

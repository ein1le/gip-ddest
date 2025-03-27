import cv2
import pytesseract
import pandas as pd
import numpy as np
import argparse
import os
from datetime import timedelta

def extract_numbers_from_video(video_path, interval_seconds=1.0, roi=None, output_csv="numbers_output.csv", 
                               display_preview=False, threshold_value=127, tesseract_config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.'):
    """
    Extract numbers displayed on a video at regular intervals and save to CSV
    
    Parameters:
    - video_path: Path to the MP4 video file
    - interval_seconds: Time interval between frame captures (in seconds)
    - roi: Region of interest [x, y, width, height] where numbers appear
    - output_csv: Path to save the output CSV file
    - display_preview: Whether to display processing preview window
    - threshold_value: Binary threshold value for image preprocessing (0-255)
    - tesseract_config: Configuration parameters for Tesseract OCR engine
    
    Returns:
    - DataFrame containing timestamps and extracted numbers
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"Video FPS: {fps}")
    print(f"Video duration: {timedelta(seconds=duration)}")
    
    # Calculate frame interval
    frame_interval = int(fps * interval_seconds)
    
    # Initialize results list
    results = []
    current_frame = 0
    
    while True:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate timestamp
        timestamp = current_frame / fps
        timestamp_str = str(timedelta(seconds=timestamp))
        
        # Extract region of interest if specified
        if roi:
            x, y, w, h = roi
            roi_frame = frame[y:y+h, x:x+w]
        else:
            roi_frame = frame
        
        # Preprocess the image for better OCR
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to make text more visible
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # Use Tesseract to extract text
        extracted_text = pytesseract.image_to_string(thresh, config=tesseract_config).strip()
        
        # Try to convert to number if possible
        try:
            extracted_number = float(extracted_text)
        except ValueError:
            extracted_number = None
        
        # Add to results
        results.append({
            'timestamp': timestamp,
            'timestamp_str': timestamp_str,
            'frame': current_frame,
            'raw_text': extracted_text,
            'number': extracted_number
        })
        
        # Display preview if requested
        if display_preview:
            # Draw rectangle on original frame
            if roi:
                frame_copy = frame.copy()
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
                preview_frame = frame_copy
            else:
                preview_frame = frame
                
            # Add text overlay
            cv2.putText(preview_frame, f"Time: {timestamp_str}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(preview_frame, f"Detected: {extracted_text}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show ROI image processing
            cv2.imshow("Original ROI", roi_frame)
            cv2.imshow("Thresholded ROI", thresh)
            cv2.imshow("Preview", preview_frame)
            
            # Break if 'q' key is pressed
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        # Print progress
        print(f"Processing frame {current_frame}/{frame_count} - Time: {timestamp_str} - Detected: {extracted_text}")
        
        # Move to next interval
        current_frame += frame_interval
        if current_frame >= frame_count:
            break
    
    # Release resources
    cap.release()
    if display_preview:
        cv2.destroyAllWindows()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    return df

def select_roi(video_path):
    """
    Interactive tool to select region of interest from the first frame of the video
    
    Parameters:
    - video_path: Path to the video file
    
    Returns:
    - ROI coordinates [x, y, width, height] or None if canceled
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read the first frame of the video")
        return None
    
    # Instructions
    print("Select the region where numbers appear, then press ENTER or SPACE to confirm")
    print("Press C to cancel selection")
    
    # Select ROI
    roi = cv2.selectROI("Select Number Region", frame, False)
    cv2.destroyAllWindows()
    
    if roi == (0, 0, 0, 0):
        return None
    
    return roi

def interactive_threshold_tuning(video_path, roi):
    """
    Interactive tool to tune the threshold value for optimal OCR
    
    Parameters:
    - video_path: Path to the video file
    - roi: Region of interest [x, y, width, height]
    
    Returns:
    - Selected threshold value
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read the first frame of the video")
        return 127
    
    x, y, w, h = roi
    roi_frame = frame[y:y+h, x:x+w]
    
    def update_threshold(val):
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("Threshold Preview", thresh)
        
        # OCR result with current threshold
        text = pytesseract.image_to_string(thresh, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.').strip()
        preview = np.zeros((100, 400), dtype=np.uint8)
        cv2.putText(preview, f"Detected: {text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.imshow("OCR Result", preview)
    
    # Create window with trackbar
    cv2.namedWindow("Threshold Preview")
    cv2.createTrackbar("Threshold", "Threshold Preview", 127, 255, update_threshold)
    
    # Initial update
    update_threshold(127)
    
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == 27 or key == ord('q') or key == 13:  # ESC, q, or Enter
            break
    
    threshold_value = cv2.getTrackbarPos("Threshold", "Threshold Preview")
    cv2.destroyAllWindows()
    
    return threshold_value

def main():
    parser = argparse.ArgumentParser(description="Extract numbers from video at regular intervals")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--interval", type=float, default=1.0, help="Time interval between captures in seconds")
    parser.add_argument("--output", default="numbers_output.csv", help="Output CSV file path")
    parser.add_argument("--preview", action="store_true", help="Display preview windows")
    parser.add_argument("--select-roi", action="store_true", help="Interactively select ROI")
    parser.add_argument("--tune-threshold", action="store_true", help="Interactively tune threshold")
    parser.add_argument("--x", type=int, help="ROI x coordinate")
    parser.add_argument("--y", type=int, help="ROI y coordinate")
    parser.add_argument("--width", type=int, help="ROI width")
    parser.add_argument("--height", type=int, help="ROI height")
    parser.add_argument("--threshold", type=int, default=127, help="Binary threshold value (0-255)")
    
    args = parser.parse_args()
    
    # Check if tesseract is installed
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract OCR is not installed or not in PATH")
        print("Please install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
        return
    
    # Select ROI if requested
    roi = None
    if args.select_roi:
        roi = select_roi(args.video_path)
        if roi:
            print(f"Selected ROI: {roi}")
    elif all(v is not None for v in [args.x, args.y, args.width, args.height]):
        roi = (args.x, args.y, args.width, args.height)
    
    # Tune threshold if requested
    threshold = args.threshold
    if args.tune_threshold and roi:
        threshold = interactive_threshold_tuning(args.video_path, roi)
        print(f"Selected threshold: {threshold}")
    
    # Process video
    extract_numbers_from_video(
        args.video_path,
        interval_seconds=args.interval,
        roi=roi,
        output_csv=args.output,
        display_preview=args.preview,
        threshold_value=threshold
    )

if __name__ == "__main__":
    main()
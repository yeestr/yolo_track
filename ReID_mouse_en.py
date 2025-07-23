from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import time
import argparse
from matplotlib.widgets import Button

# Command line arguments
parser = argparse.ArgumentParser(description="Cross-Camera Person Re-Identification (Matplotlib Version)")
parser.add_argument("--video1", type=str, default="video/cam1.mp4", help="Path to first video")
parser.add_argument("--video2", type=str, default="video/cam4.mp4", help="Path to second video")
parser.add_argument("--output", type=str, default="", help="Output video path (optional)")
parser.add_argument("--conf", type=float, default=0.3, help="Object detection confidence threshold")
parser.add_argument("--sim_thresh", type=float, default=0.6, help="Re-identification similarity threshold")
parser.add_argument("--skip_frames", type=int, default=3, help="Process every n-th frame (for speed)")
args = parser.parse_args()

# Initialize YOLO model (for person detection)
print("Loading YOLO model...")
model = YOLO('yolov8m.pt')

# Initialize ReID model
print("Loading ReID model...")
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Global variables
target_feature = None  # Store target person's feature
target_selected = False  # Whether a target has been selected
processed_video1_frame = None
processed_video2_frame = None
current_bboxes1 = []
current_bboxes2 = []
current_crops1 = []
best_match_idx = None
best_match_sim = 0.0
tracking_id = None
paused = False
frame_count = 0

# Detection function
def process_detections(frame, conf_thresh=0.3):
    """Process person detections in a single frame"""
    results = model(frame, conf=conf_thresh)[0]
    bboxes = []
    crops = []
    
    for box in results.boxes:
        cls = int(box.cls.item())
        if cls != 0:  # Only focus on persons
            continue
        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Extract person image
        crop_img = frame[y1:y2, x1:x2]
        if crop_img.size == 0:  # Ensure crop area is valid
            continue
            
        bboxes.append((x1, y1, x2, y2, conf))
        crops.append(crop_img)
    
    return bboxes, crops

# Feature extraction
def extract_features(crops):
    """Extract feature vectors"""
    if not crops:
        return None
    # Convert to PIL image list
    pil_crops = []
    for crop in crops:
        pil_crops.append(np.array(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
    # Extract features
    features = extractor(pil_crops)
    return features

# Similarity calculation
def compute_similarity(feat1, feat2):
    """Compute cosine similarity between feature sets"""
    feat1 = torch.nn.functional.normalize(feat1, dim=1)
    feat2 = torch.nn.functional.normalize(feat2, dim=1)
    return torch.mm(feat1, feat2.T)

# Draw bounding boxes
def draw_boxes_on_frame(frame, bboxes, highlight_idx=None, is_matched=False):
    """Draw bounding boxes on frame"""
    img = frame.copy()
    
    # Draw all detection boxes
    for i, (x1, y1, x2, y2, conf) in enumerate(bboxes):
        # Determine box color and thickness
        if highlight_idx is not None and i == highlight_idx:
            if is_matched:
                # Blue: matched target
                color = (255, 0, 0)
            else:
                # Green: selected target
                color = (0, 255, 0)
            thickness = 3
        else:
            # White: other persons
            color = (255, 255, 255)
            thickness = 2
            
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Show ID number
        cv2.putText(img, f"#{i+1}", (x1, y1+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # For matched target, show similarity
        if is_matched and i == highlight_idx and best_match_sim > 0:
            cv2.putText(img, f"{best_match_sim:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return img

# Prepare frame for Matplotlib display
def prepare_for_display(frame):
    """Convert OpenCV frame to RGB for matplotlib display"""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Click event handler
def on_click(event):
    global target_selected, target_feature, tracking_id, best_match_idx, best_match_sim
    
    if not event.inaxes:
        return
    
    # Check if click is in the left subplot (first video)
    if event.inaxes == ax1 and not target_selected and current_bboxes1:
        x, y = event.xdata, event.ydata
        
        # Check if click is inside any detection box
        for i, (x1, y1, x2, y2, _) in enumerate(current_bboxes1):
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Select person
                tracking_id = i
                print(f"Selected person #{i+1} as tracking target")
                
                # Extract features
                if current_crops1:
                    selected_features = extract_features([current_crops1[i]])
                    if selected_features is not None:
                        target_feature = selected_features
                        target_selected = True
                        
                        # Update display
                        update_display()
                        
                        # If we already have detections for the second video, calculate matches immediately
                        if current_bboxes2 and len(current_bboxes2) > 0:
                            find_best_match()
                return

# Find best match
def find_best_match():
    global best_match_idx, best_match_sim, processed_video2_frame
    
    if target_feature is None or not current_bboxes2:
        return
    
    # Extract features for all persons in the second video
    crops2 = []
    for x1, y1, x2, y2, _ in current_bboxes2:
        crop = processed_video2_frame[y1:y2, x1:x2]
        crops2.append(crop)
    
    if not crops2:
        return
        
    features2 = extract_features(crops2)
    if features2 is None:
        return
    
    # Calculate similarity
    sim = compute_similarity(target_feature, features2)
    
    # Find the most similar person
    best_idx = torch.argmax(sim).item()
    best_sim = sim[0, best_idx].item()
    
    if best_sim >= args.sim_thresh:
        best_match_idx = best_idx
        best_match_sim = best_sim
        print(f"Found best match: Person #{best_idx+1}, similarity: {best_sim:.3f}")
    else:
        best_match_idx = None
        best_match_sim = best_sim
        print(f"No high-similarity match found (highest: {best_sim:.3f}, threshold: {args.sim_thresh})")

# Reset selection
def reset_selection(event):
    global target_selected, target_feature, tracking_id, best_match_idx, best_match_sim
    
    target_selected = False
    target_feature = None
    tracking_id = None
    best_match_idx = None
    best_match_sim = 0.0
    print("Selection reset")
    
    # Update display
    update_display()

# Pause/resume video
def toggle_pause(event):
    global paused
    paused = not paused
    print("Video " + ("paused" if paused else "resumed"))
    btn_pause.label.set_text("Resume" if paused else "Pause")
    plt.draw()

# Exit program
def exit_program(event):
    plt.close()

# Update display
def update_display():
    global fig, ax1, ax2, processed_video1_frame, processed_video2_frame
    
    # Clear current axes
    ax1.clear()
    ax2.clear()
    
    # Prepare display frames
    if processed_video1_frame is not None:
        # Prepare first video frame (left)
        if target_selected and tracking_id is not None:
            # Mark selected target
            display_frame1 = draw_boxes_on_frame(processed_video1_frame, current_bboxes1, tracking_id)
        else:
            # Show all detection boxes
            display_frame1 = draw_boxes_on_frame(processed_video1_frame, current_bboxes1)
            
        ax1.imshow(prepare_for_display(display_frame1))
        ax1.set_title("Video 1 - Click to select person to track")
    
    if processed_video2_frame is not None:
        # Prepare second video frame (right)
        if target_selected and best_match_idx is not None:
            # Mark matched target
            display_frame2 = draw_boxes_on_frame(processed_video2_frame, current_bboxes2, best_match_idx, True)
        else:
            # Show all detection boxes
            display_frame2 = draw_boxes_on_frame(processed_video2_frame, current_bboxes2)
            
        ax2.imshow(prepare_for_display(display_frame2))
        if target_selected:
            if best_match_idx is not None:
                ax2.set_title(f"Video 2 - Best match: #{best_match_idx+1}, similarity: {best_match_sim:.3f}")
            else:
                ax2.set_title(f"Video 2 - No match found (highest similarity: {best_match_sim:.3f})")
        else:
            ax2.set_title("Video 2")
    
    # Remove axes
    ax1.set_axis_off()
    ax2.set_axis_off()
    
    # Add global title
    fig.suptitle(f"Cross-Camera Person Tracking - Frame: {frame_count}" + (" (Paused)" if paused else ""), fontsize=12)
    
    # Redraw
    fig.canvas.draw_idle()

# Main function
def main():
    global processed_video1_frame, processed_video2_frame, current_bboxes1, current_bboxes2, current_crops1, frame_count
    global fig, ax1, ax2, btn_reset, btn_pause, btn_exit
    
    # Open videos
    cap1 = cv2.VideoCapture(args.video1)
    cap2 = cv2.VideoCapture(args.video2)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Cannot open video files")
        return
    
    # Get video properties
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare output video
    output_writer = None
    if args.output:
        output_writer = cv2.VideoWriter(
            args.output, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps1,  # Use first video's frame rate
            (width1 + width2, max(height1, height2))
        )
    
    # Create matplotlib figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Add control buttons
    plt.subplots_adjust(bottom=0.15)
    
    ax_reset = plt.axes([0.3, 0.05, 0.1, 0.075])
    ax_pause = plt.axes([0.45, 0.05, 0.1, 0.075])
    ax_exit = plt.axes([0.6, 0.05, 0.1, 0.075])
    
    btn_reset = Button(ax_reset, 'Reset')
    btn_pause = Button(ax_pause, 'Pause')
    btn_exit = Button(ax_exit, 'Exit')
    
    btn_reset.on_clicked(reset_selection)
    btn_pause.on_clicked(toggle_pause)
    btn_exit.on_clicked(exit_program)
    
    # Show initial empty images
    ax1.set_title("Video 1 - Click to select person to track")
    ax2.set_title("Video 2")
    ax1.set_axis_off()
    ax2.set_axis_off()
    
    plt.ion()  # Enable interactive mode
    plt.show(block=False)
    
    # Video processing loop
    try:
        while plt.fignum_exists(fig.number):  # Check if figure is still open
            if not paused:
                # Read frames
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    print("Video ended")
                    break
                
                frame_count += 1
                
                # Save original frames
                processed_video1_frame = frame1.copy()
                processed_video2_frame = frame2.copy()
                
                # Process every n-th frame
                if frame_count % args.skip_frames == 0:
                    # Process video 1 detections
                    bboxes1, crops1 = process_detections(frame1, args.conf)
                    current_bboxes1 = bboxes1
                    current_crops1 = crops1
                    
                    # Process video 2 detections
                    bboxes2, _ = process_detections(frame2, args.conf)
                    current_bboxes2 = bboxes2
                    
                    # If target is selected, find matches
                    if target_selected:
                        find_best_match()
                    
                    # Update display
                    update_display()
                
                # Write output video
                if output_writer is not None:
                    # Prepare output frame
                    output_height = max(height1, height2)
                    output_width = width1 + width2
                    output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                    
                    # Combine two video frames
                    if target_selected and tracking_id is not None:
                        # Video 1 - Mark selected target
                        display_frame1 = draw_boxes_on_frame(frame1, current_bboxes1, tracking_id)
                    else:
                        display_frame1 = draw_boxes_on_frame(frame1, current_bboxes1)
                        
                    if target_selected and best_match_idx is not None:
                        # Video 2 - Mark matched target
                        display_frame2 = draw_boxes_on_frame(frame2, current_bboxes2, best_match_idx, True)
                    else:
                        display_frame2 = draw_boxes_on_frame(frame2, current_bboxes2)
                    
                    output_frame[:height1, :width1] = display_frame1
                    output_frame[:height2, width1:width1+width2] = display_frame2
                    
                    output_writer.write(output_frame)
            
            # Process UI events
            plt.pause(0.01)  # Give matplotlib time to process events
    
    except KeyboardInterrupt:
        print("User interrupted")
    
    finally:
        # Clean up resources
        cap1.release()
        cap2.release()
        if output_writer is not None:
            output_writer.release()
        plt.close()
        print("Program ended")

if __name__ == "__main__":
    main()
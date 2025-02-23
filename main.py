import cv2
import time
import threading
from ultralytics import YOLO
from utills.deep_sort import DeepSort
import torch
import os
import yaml

with open(f'config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#--------------main configutrations ---------
input_video_file_name = config.get("input_video_file_name","supermarket_trimmed.mp4")
draw = config.get("draw", False)
show_output = config.get("show_output",False)
save_output = config.get("save_output",False)
if_save_output_true_TimePeriod = config.get("if_save_output_true_TimePeriod",1000)

# -----yolo model configurations ------------
yolo_confidence_score = config.get("yolo_confidence_score",0.3)
yolo_required_class_ids = config.get("yolo_required_class_ids",[0])
yolo_input_img_size = config.get("yolo_input_img_size",640)

# ------- deepsort configurations -----------
max_dist = config.get("max_cosine_dist",0.2)
nms_max_overlap = config.get("nms_max_overlap",1.0)
max_iou_distance = config.get("max_iou_distance",0.7)
max_age = config.get("max_age",70)
n_init = config.get("n_init",3)
nn_budget = config.get("nn_budget",100)
use_cuda_for_deepsort = config.get("use_cuda_for_deepsort",False)

# YOLO and DeepSORT initialization
model = YOLO('models/yolov8n.pt')
deepsort = DeepSort(
            model_path="utills/deep_sort/deep/checkpoint/ckpt.t7",
            max_dist=max_dist,
            min_confidence=yolo_confidence_score,
            nms_max_overlap=nms_max_overlap,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget,
            use_cuda=use_cuda_for_deepsort)

out_video_path = "output.mp4"

cap = cv2.VideoCapture(input_video_file_name)

# Define video writer
if save_output:
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_width, frame_height))

# Define variables
unique_person_ids = set()
lock = threading.Lock()  # Lock for thread-safe counter reset

frame_count = 0

system_start_time = time.time()
try:
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 Detection
        results = model.predict(source=frame, conf=yolo_confidence_score, classes=yolo_required_class_ids, save=False, verbose=False, imgsz=yolo_input_img_size)
        if results[0].boxes is None:
            continue
        xywhs = results[0].boxes.xywh.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        if len(xywhs) == 0:
            continue

        # DeepSORT tracking
        tracks = deepsort.update(torch.Tensor(xywhs), scores, class_ids, frame)
        if tracks is None or len(tracks) == 0:
            continue
        bboxes = tracks[:, :4]
        identities = tracks[:, -2]

        with lock:
            for i, box in enumerate(bboxes):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                track_id = int(identities[i]) if identities is not None else 0
                unique_person_ids.add(track_id)
                if draw:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                #TODO

        # Calculate FPS
        if draw:
            fps = 1 / (time.time() - start_time)


        # Display FPS and counts
        if draw:
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        frame_count += 1
        print(f"Frame count = {frame_count}")

        # save the output
        if save_output:
            out.write(frame)
            remaining_time_to_save_video = time.time() - system_start_time
            if remaining_time_to_save_video >= if_save_output_true_TimePeriod:
                out.release()
                save_output= False
                print("vdeo saved successfully")


        if show_output:
            cv2.imshow('Object Tracking', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("\n Keyboard interrupt detected! Stopping gracefully...")

finally:
    # Release resources
    cap.release()
    if show_output:
        cv2.destroyAllWindows()
    print("Resources released successfully.")
    print (unique_person_ids)
    print (f"Total Person found = {len(unique_person_ids)}")
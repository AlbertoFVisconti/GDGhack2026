import cv2
import depthai as dai
import numpy as np
import time

# Trigger distance in millimeters (10000mm = 10 meters)
trigger_range = 10000

# 1. Initialize Pipeline using the new v3 context manager
with dai.Pipeline() as pipeline:

    # 2. Define the Center Camera (for YOLO)
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    # 3. Define the Left and Right Cameras (for Depth)
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # 4. Define the Stereo Depth Node
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    # Link Left/Right lenses to the Stereo Node
    monoLeftOut = monoLeft.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8)
    monoRightOut = monoRight.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8)
    monoLeftOut.link(stereo.left)
    monoRightOut.link(stereo.right)

    # Create the depth queue
    qDepth = stereo.depth.createOutputQueue()

    # 5. Define YOLOv10 Node
    print("Initializing YOLOv10...")
    model_description = dai.NNModelDescription("luxonis/yolov10-nano:coco-512x288")
    yolo = pipeline.create(dai.node.DetectionNetwork).build(camRgb, model_description)

    # Create the YOLO queue
    qYolo = yolo.out.createOutputQueue()

    # Standard COCO dataset labels
    labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    # 6. Start the Pipeline
    pipeline.start()
    
    print("Pipeline running. Monitoring central path for objects...")

    cooldown_dict = {}
    cooldown_seconds = 5.0

    while pipeline.isRunning():
        inDepth = qDepth.get()
        inYolo = qYolo.tryGet()

        depthFrame = inDepth.getFrame()
        height, width = depthFrame.shape
        
        # --- PATH VISUALIZATION SETTINGS ---
        # The ratio of the screen width to consider as "directly ahead"
        path_width_ratio = 0.40 
        
        left_boundary = int(width * (0.5 - (path_width_ratio / 2)))
        right_boundary = int(width * (0.5 + (path_width_ratio / 2)))
        
        # Create the visual depth map
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthFrame, alpha=0.03), cv2.COLORMAP_JET)

        # Plot the path boundary lines (Green lines)
        cv2.line(depth_colormap, (left_boundary, 0), (left_boundary, height), (0, 255, 0), 2)
        cv2.line(depth_colormap, (right_boundary, 0), (right_boundary, height), (0, 255, 0), 2)

        if inYolo is not None:
            detections = inYolo.detections
            
            for detection in detections:
                label_name = labels[detection.label]
                
                # Get the bounding box of the object
                x1 = int(detection.xmin * width)
                y1 = int(detection.ymin * height)
                x2 = int(detection.xmax * width)
                y2 = int(detection.ymax * height)
                
                # Calculate the center of the object
                object_center_x = (x1 + x2) // 2
                
                # Check if the object is within the path boundaries
                if left_boundary <= object_center_x <= right_boundary:
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    object_roi = depthFrame[y1:y2, x1:x2]
                    valid_depths = object_roi[object_roi > 0]
                    
                    if len(valid_depths) > 0:
                        avg_depth = np.mean(valid_depths)
                        
                        # Draw tracking box for objects in the path (White box)
                        cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        cv2.putText(depth_colormap, f"{label_name}: {avg_depth/1000:.1f}m", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        if avg_depth < trigger_range:
                            current_time = time.time()
                            
                            # Debounce notification
                            if label_name not in cooldown_dict or (current_time - cooldown_dict[label_name]) > cooldown_seconds:
                                print(f"NOTIFY USER: '{label_name}' directly ahead at {avg_depth/1000:.1f} meters!")
                                cooldown_dict[label_name] = current_time

        cv2.imshow("Spatial Vision - Path Detection", depth_colormap)

        if cv2.waitKey(1) == ord('q'):
            break
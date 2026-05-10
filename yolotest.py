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

    # Create the depth queue directly from the node
    qDepth = stereo.depth.createOutputQueue()

    # 5. Define YOLOv10 Node
    print("Initializing YOLOv10...")
    model_description = dai.NNModelDescription("luxonis/yolov10-nano:coco-512x288")
    yolo = pipeline.create(dai.node.DetectionNetwork).build(camRgb, model_description)

    # Create the YOLO queue directly
    qYolo = yolo.out.createOutputQueue()

    # Standard COCO dataset labels
    labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    # 6. Start the Pipeline
    pipeline.start()
    
    print("Pipeline running. YOLO-First Architecture active...")

    cooldown_dict = {}
    cooldown_seconds = 5.0

    # The loop now runs as long as the pipeline is active
    while pipeline.isRunning():
        inDepth = qDepth.get()
        inYolo = qYolo.tryGet()

        depthFrame = inDepth.getFrame()
        height, width = depthFrame.shape
        
        # Create the visual map right away
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthFrame, alpha=0.03), cv2.COLORMAP_JET)

        # YOLO-FIRST LOGIC: Only check depth if YOLO actually sees something
        if inYolo is not None:
            detections = inYolo.detections
            
            for detection in detections:
                label_name = labels[detection.label]
                
                # 1. Get the exact bounding box of the object (scaled to the screen size)
                x1 = int(detection.xmin * width)
                y1 = int(detection.ymin * height)
                x2 = int(detection.xmax * width)
                y2 = int(detection.ymax * height)
                
                # Protect against out-of-bounds coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # 2. Extract ONLY the depth pixels inside the YOLO bounding box
                object_roi = depthFrame[y1:y2, x1:x2]
                valid_depths = object_roi[object_roi > 0] # Filter out blind spots (zeros)
                
                if len(valid_depths) > 0:
                    # 3. Calculate the actual distance to this specific object
                    avg_depth = np.mean(valid_depths)
                    
                    # Draw a box on the screen so you can see it working
                    cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(depth_colormap, f"{label_name}: {avg_depth/1000:.1f}m", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # 4. Check if it's within your trigger range
                    if avg_depth < trigger_range:
                        current_time = time.time()
                        
                        # 5. DEBOUNCE (Stop the spam)
                        # We ONLY print if it's new, or if 5 seconds have passed for THIS specific label
                        if label_name not in cooldown_dict or (current_time - cooldown_dict[label_name]) > cooldown_seconds:
                            
                            print(f"NOTIFY USER: '{label_name}' detected at {avg_depth/1000:.1f} meters!")
                            
                            # Reset the clock for this object
                            cooldown_dict[label_name] = current_time

        cv2.imshow("Spatial Vision (YOLO-First)", depth_colormap)

        if cv2.waitKey(1) == ord('q'):
            break
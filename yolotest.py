import cv2
import depthai as dai
import numpy as np
import time

trigger_range=10000

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

    # V3 MAGIC: No more XLinkOut! Create the queue directly from the node
    qDepth = stereo.depth.createOutputQueue()

    # 5. Define YOLOv10 Node
    print("Initializing YOLOv10...")
    model_description = dai.NNModelDescription("luxonis/yolov10-nano:coco-512x288")
    yolo = pipeline.create(dai.node.DetectionNetwork).build(camRgb, model_description)

    # V3 MAGIC: Create the YOLO queue directly
    qYolo = yolo.out.createOutputQueue()

    # Standard COCO dataset labels
    labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    # 6. Start the Pipeline (v3 no longer uses dai.Device)
    pipeline.start()
    
    print("Pipeline running. Monitoring depth for triggers...")


    cooldown_dict = {}
    cooldown_seconds = 5.0

    # The loop now runs as long as the pipeline is active
    while pipeline.isRunning():
        inDepth = qDepth.get()
        inYolo = qYolo.tryGet()

        depthFrame = inDepth.getFrame()
        
        height, width = depthFrame.shape
        center_x, center_y = width // 2, height // 2
        roi_size = 30
        
        center_roi = depthFrame[center_y - roi_size : center_y + roi_size, 
                                center_x - roi_size : center_x + roi_size]
        valid_depths = center_roi[center_roi > 0]
        
        if len(valid_depths) > 0:
            avg_depth = np.mean(valid_depths)
            
            # TRIGGER CONDITION: Object is closer than 1000mm (1 meter)
            if avg_depth < trigger_range:
                print(f"Trigger Active: Object detected at {avg_depth:.2f} mm.")
                
                if inYolo is not None:
                    detections = inYolo.detections
                    if detections:
                        for detection in detections:
                            label_name = labels[detection.label]
                            current_time = time.time()
                            
                            # Only notify if we've never seen it, OR if 5 seconds have passed
                            if label_name not in cooldown_dict or (current_time - cooldown_dict[label_name]) > cooldown_seconds:
                                print(f"NOTIFY USER: Identified '{label_name}' at {avg_depth:.0f}mm")
                                
                                # Record the time we saw it
                                cooldown_dict[label_name] = current_time
                    else:
                        print("Analysis: YOLO active, but object class not recognized.")
                print("-" * 40)

        # Display the depth map visually
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthFrame, alpha=0.03), cv2.COLORMAP_JET)
        
        cv2.rectangle(depth_colormap, 
                     (center_x - roi_size, center_y - roi_size), 
                     (center_x + roi_size, center_y + roi_size), 
                     (255, 255, 255), 2)
                     
        cv2.imshow("Depth Trigger Architecture", depth_colormap)

        if cv2.waitKey(1) == ord('q'):
            break
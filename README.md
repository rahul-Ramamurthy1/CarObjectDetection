# CarObjectDetection

**Objectives** 
1.   Understanding YOLO
2.   Performing Bounding Box Prediction
3.   Performing Object Detection on videos

**Description** <br />
This project will detect objects in a video, using “You Only Look Once,” or YOLO, family of models and Darknet, a deep convolutional neural network model.<br />
A video contains multiple frames (images) and is just a sequence of frames (images). <br />

The “You Only Look Once,” or YOLO, family of models are a series of end-to-end deep learning models designed for fast object detection on images, developed by Joseph Redmon, et al. and first proposed in the 2015 paper titled “You Only Look Once: Unified, Real-Time Object Detection.” YOLO can also be easily extended to detect objects on videos.

The approach involves a single deep convolutional neural network (DarkNet which is based on the VGG, a deep convolutional neural network model) that splits the input
into a grid of cells and each cell directly predicts a bounding box and object classification. The result is a large number of candidate bounding boxes that are 
consolidated into a final prediction by a post-processing step. <br />
As multiple bounding boxes may be detecting the same object, the post-processing step will remove the overlapping bounding boxes and leave the best ones, by applying a threshold to filter the result. <br/>
Our final goal is to detect objects in a video, which contains multiple frames (images). <br />



# Object Detection
This is my summer project about Object Detection.

## Goals:
- Learn and Build some popular End-to-End Object Detection models that need to be fast, accurate, and able to deploy.
- (Optional) Deploy to web-sever using AWS.

## Times:
- 2 months: Jul 2024 - Aug 2024

## Dataset & Environments:
- Dataset: Pascal VOC 2007 and 2012.
- GPU: GPU P100 of Kaggle.
- Framework: Pytorch.

## Approach:
- For the purpose of deployment, the models should be not only accurate but also fast which is possible for deployment in real-time scenarios. Because of this reason, I chose 2 families of architectures which are YOLOs and SSDs.
- Resnet-18 and Mobilenet-v3 are chosen as the backbone in YOLO and SSD respectively instead of the DarkNet and VGG in the original paper for the flexible purpose. Another reason is I want small-footprint models which can be adapted to a wide range of platforms.

## Results:
I am temporarily stopping the project for personal reasons. Although my time on this project was small, I still got some encouraging results.
- I have done building SSD, YOLO version 1, YOLO version 2.
- For prediction, I have already built for YOLO v1 and YOLO v2. SSD will be updated soon.
- Other works are spent for the future (i.e. Deployment).
Some inferences were made on a few samples from the evaluation dataset and I got some positive results. You can find it in the notebooks folder. 

## Limitations:
SSD is better in theory, so good or bad performance depends on how good the model's backbone is. 2 versions of YOLO I built based on the original papers, but it wasn't as good as it had been described. 
Specifically, in YOLO v1, the loss function I used was taken from the original paper which seems everything is regression problems and uses only the MSE loss function for calculation.
As a result, the reliable score, which is the production of object probability and the class probability, was very low, then it was hard to choose the right threshold and make decisions.
In YOLO v2, I used BCE instead and got higher scores for making decisions, but YOLO was still bad at predicting the class which may be because of the aggregate of multiple functions.
Honestly, YOLO families' sources are not published, so everything I have done is just my subjective perspective.

## References:
Papers:
- Zaidi, Syed Sahil Abbas, et al. "A survey of modern deep learning based object detection models." Digital Signal Processing 126 (2022): 103514.
- Redmon, J. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- Redmon, Joseph, and Ali Farhadi. "YOLO9000: better, faster, stronger." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
- Liu, Wei, et al. "Ssd: Single shot multibox detector." Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14. Springer International Publishing, 2016.

Other works:
- <https://d2l.ai/chapter_computer-vision/ssd.html>
- <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py>
- <https://github.com/motokimura/yolo_v1_pytorch/tree/master>

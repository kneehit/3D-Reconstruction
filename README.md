# 3D Reconstruction

## Description
This repository implements 3D Reconstruction (of Flash figure) using Structure from Motion Pipeline and RGB images. 
The pipeline consists of 
1. Feature Detection (using SURF / SIFT / Hessian Affine Region Detector)
2. Feature Matching (using FLANN)
3. Reconstruction (using Patch Match)



## Pipeline
![alt text](assets/flowchart.png)

## Results
![alt text](assets/sfm.gif)

## Closing Thoughts
Although we get decent reconstruction of the figure, it can be further improved
1. Statistical or Radial Outlier Removal can further help in noise removal from the dense reconstruction.
2. The head in front view has not been reconstructed inspite of many images in same pose. This is due glare from the lights directly above it. Image processing algorithms can help fix this issue.

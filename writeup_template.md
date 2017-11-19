## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
As an initial step, I just played around with the lecture topics and tried out histograms.

Initially I just worked out taking a small set of data (nReduction = 1000).

Then, I trained a linear SVM using feature and label vectors as defined in my code (Notebook.ipynb). 
For the topic of training test data split 
- Following the normal shuffle and split
- First splitting the data and shuffle
With the discussion forum, I also feel that before shuffling the data,
It is better to split the data into train and test. Even a random train-test split will be subject to overfitting since the images will be identical in both training and test set. Accuracy of the classifier was acceptable in my case. The splitting and shuffling is accompalished from the function split_n_shuffle()

#### 2. Explain how you settled on your final choice of HOG parameters.

I explored different set of parameters (orientations, pixes per cell, cells per block). The final set of parameters is as defined below:



The accuracy with these set of parameters was above 99%

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

As explained above, I trained with linear SVC. Few important points to be considered here are:
- The number of images for training on the SVC.This will increase the accuracy of detection
- How good is the window size
I used different color spaces, HLS yields good results due to Brightness, saturation and color. Saturation provides good gradients and color helps identifying color of cars. But finally settled with YCrCb w.r.t correct detections and false detections (my opinion)

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The image shows there are wrong identification.

As in the lecture, it is good to search in the lower half part of the image where the possibility of car would be too high and eliminate the sky and other parts. And also, since in the video input, total lower half also could be split into kind of trapezium shape area for searching (to be specific, the other side of lane is not our region of interest, but depends if you also want to detect vehicles on other side. But i do not find it useful)
- Cars that are closer to the camera appear to be larger and far from camera appear to be smaller
- Looking for cars on the ride side of our camera is also good practice (oops! But not the best in all cases)
- Assumption : The camera position is fixed. Else the 'MainWindow' which is the region of interest gets altered

To further improve the searching, I refined the search with global search window.
The area that I am looking is selected as in the figure below:


I played around with this region of interest by changing the shape and size. But got satisfied with the one present in my notebook. 

H – height of the frame, w – width of the frame


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on five scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. decision_function method of LinearSVC is used to eliminate false positives.

Refining the pipeline :
Some parts of code are taken from lecture and Mr.Alexander braun suggestion

- The class Scan() has two methods for scan_local and scan_global. As the name suggests, the scan_global() scans the whole region of interest, taking different scales corresponding to perspective scaling. This is used for video processing for the first frame. Later, the scan_local is used. Local scan scans the horizon since the cars enter the region of interest from there. The right side of our region of interest (as for project), a list of bounding boxes where there was a previous frame car detection.
The result is as shown below:


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
- It is present in github repo with name project_video_output.mp4
- The positions of positive detections in each frame are recorded.The heat map is created and then threshold that map to identify the vehicle positions. The individual blobs are identified in the heat map. Each blob is considered as a car(assumption). Bounding boxes are constructed to cover the area of each blob detected.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.


An example showing the heat map


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- The speed of the car also plays an important role here. If the cars are travelling very high speed, the slow filter that has been used here to smoothen the bound boxes and to suppress false positives. We know that if there was already a bound box, then the probability to find a car is high (position is looked in scan_local in class Scan())
- Different weights could be given to detect in the regions where there is probability to find a car and less weightage to the regions where there is less probability to find a car.
- Different distance to decision surface could improve the problems
- If processing of images is done faster (using a high GPU), then the pipeline will run very faster and will be able to detect cars quickly including cars which are overtaking at very fast speed.
- Although the pipeline is running bit slow in my PC, you can see the time taken to run the whole ProcessImage() pipeline is approximately 45 minutes. This could be the reason for small boxes appearing there (I am not sure for these false detections, but they do not appear somewhere else though), but if the pipeline is run on very high processing GPU, things would definitely improve I guess. The pipeline seems to be ok.
- Few parts of my code are based on the support of Mr.Alexander Braun who graduated at Udacity Nano-degree program. I should admit that the mentor support for my projects has been almost zero.


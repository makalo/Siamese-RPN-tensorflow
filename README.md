# Siamese-RPN-tensorflow
Code for reproducing the results in the following paper:
- [**High Performance Visual Tracking with Siamese Region Proposal Network**](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf)  
- [Pytorch version](https://github.com/songdejia/siamese-RPN.git) has been available by my classmates.
- Anthor version [zkisthebest/Siamese-RPN](https://github.com/zkisthebest/Siamese-RPN.git) have lots of bugs.So I have to re-implementation by myself
## Environment
python=3.6
tensorflow=1.10
cuda=9.0

## Downloading VOT2013 Data
- Enter http://data.votchallenge.net/vot2013/vot2013.zip in your browser
- Unzip the file and move to `./data`
## Downloading YouTube-bb Data
- git clone https://github.com/mbuckler/youtube-bb.git
- python3 download.py ./data 12
## Training and Evaluation
If your data format is the same as VOT 2013, you can run the code directly. If not, you need to change the utils/image_reader.py or convert the data format to VOT format.

### To train Siamese-RPN:
```
python train.py
```
If you want to see if the training is reasonable in the course of training, you can choose to turn on debug.Just change the __init__() in train.py
```
self.is_debug=True
```
This will result in a debug folder where you can see pictures of the training process, with groundtruth in red and box in top 10 scores in other colors.
### To test Siamese-RPN:

**To test series of images like VOT format**

If you want to test a series of images captured from the video, you need to assign new values `img_path`and `img_label` in config.py, which are the files of your image's path and label, respectively. Then execute the following commands
```
python test.py
```
This command will automatically synthesize videos from image sequences, and also synthesize videos from processed images, which are saved in. / data / vedio

**To test a vedio**

If you are testing a video, you need to put the video in./data/vedio. You can run the following command and select the object you want to track in the first frame according to the program prompt at the beginning.
```
python vedio_test.py test.mp4
```
The 'test.mp4' is the name of your vedio

### the model 
Since the YouTube-bb data set is very large, I will provide a model for training on the YouTube-bb data set in the next period of time.




# human_face_detection
Python API that processes a video file, detects human faces within each frame using an apt library, ensuring accuracy in distinguishing real human faces from animated or fictional characters.

Two models have been applied and tested, CascadeClassifier using cv2 and MTCNN using Pytorch. In case a user does not have a GPU, CascadeClassifier is a good option with decent performance. For better accuracy MTCNN is more suitable. 

To further enhance performance (reducing false positive, and false negative), a custom dataset (~120,000 samples containing ~60,000 from cartoon class and real human class) was uploaded on google drive with scripts to download the dataset, train, save the weights, and use the fine-tuned MTCNN for better accuracy in distinguishing between real and animated faces.

**Dataset Link:**
https://drive.google.com/file/d/1OeOiGsOrXTL_5NJ6GZYiD35_c8Zcy2NN/view?usp=sharing

**output videos can be downloaded from the following google drive link**: https://drive.google.com/drive/folders/1iVfHMxvIg1S8ULUUmWQ0izegXX8L9fRB?usp=sharing


****To run the face detection program use the following command :****

python face_detection.py --video_path /path/to/video.mp4 --output_path /path/to/save/output.avi --log_file /path/to/save/log.txt --detection_model 'haarcascade', 'mtcnn' --mtcnn_weight /path/to/trained_weights.pth (optional)

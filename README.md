# human_face_detection
Python API that processes a video file, detects human faces within each frame using an apt library, ensuring accuracy in distinguishing real human faces from animated or fictional characters.

**output videos can be downloaded from the following google drive link**: https://drive.google.com/drive/folders/1iVfHMxvIg1S8ULUUmWQ0izegXX8L9fRB?usp=sharing


****To run the face detection program use the following command :****

python face_detection.py --video_path /path/to/video.mp4 --output_path /path/to/save/output.avi --log_file /path/to/save/log.txt --detection_model 'haarcascade', 'mtcnn' --mtcnn_weight /path/to/trained_weights.pth (optional)

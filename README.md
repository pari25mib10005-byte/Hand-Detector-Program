# Hand-Detector-Program
Hand Detector Program By PARI JAIN

People often mention advanced hand gesture detection that uses MediaPipe and OpenCV. This setup manages real time detection of hand gestures through MediaPipe Hands. It combines that with OpenCV and some analysis of 3D vector angles. The program identifies hand landmarks first. Then it calculates finger angles to classify various gestures. It displays the finger count directly on the screen for easy viewing.

Features cover real time hand tracking that runs smoothly without lag. It calculates 3D joint vectors and angles to improve accuracy in detection. The system recognizes several gestures such as an open palm or a fist. It also picks up thumbs up along with pointing or victory signs. Finger counting works right out of the box for straightforward use. It handles tracking of two hands simultaneously without any problems. The name of the gesture appears right on the video feed. You can see it clearly as things happen.

Requirements include OpenCV Python and MediaPipe. You need the math library too. Install them with pip install opencv python mediapipe. That takes care of the basics for setup.

How it works begins with angle detection for each finger. The openness gets determined by angles between 3D landmark vectors. Gesture classification follows based on the count of open fingers and angle thresholds. An open palm requires all five fingers to stay open. A fist means all fingers closed tight together. Thumbs up keeps only the thumb open. Pointing uses just the index finger extended. Victory has the index and middle fingers open together. Anything else gets classified as unknown.

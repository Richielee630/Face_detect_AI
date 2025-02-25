# Face_detect_AI
### Overview: The main purpose of this project is to detect human face in a real-time video capture and provide prediction on its gender and ages.

## Datasets choosed:
* [CelebA 200K](https://www.kaggle.com/ashishjangra27/gender-recognition-200k-images-celeba)
* [Facial Age_0~99](https://www.kaggle.com/frabbisw/facial-age)

## Methods and Tools being used in the project:
- Libraries:
  - Tensorflow + keras
  - Sklearn
  - numpy
  - OpenCV 
  - Matplotlib
  - Pandas
  - Pickle
- feature extraction / dimensionality reduction:
  - RGB
  - HOG (Histogram of Oriented Gradient)
  - PCA (Principal Component Analysis)
- Models:
  - Haar Cascade Classifier
  - SVC (support vector classifier)
  - RF (random forest classifier)
  - CNN (convolutional neural network)

## software level diagram
<img src=https://user-images.githubusercontent.com/60235970/143810071-177731b6-0052-4056-b1b7-d41a04dea028.jpg alt="face_detect_ai_sw_level_diagram" width = "1000"/>

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Face_detect_AI.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Face_detect_AI
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the face detection script:
   ```bash
   python face_detect.py
   ```
2. Follow the on-screen instructions to start the real-time video capture and face detection.

## Contributing
Feel free to submit issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.

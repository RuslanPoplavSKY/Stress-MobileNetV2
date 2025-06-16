# Stress-MobileNetV2

This project implements a neural network model based on MobileNetV2 to detect stress on a person's face. In addition, a simple graphical user interface (GUI) using Tkinter is provided to select an image, classify it as "stressed" or "not stressed", and view the result.


⚙️ Requirements
Python 3.7+

TensorFlow 2.x

numpy

matplotlib

sklearn

Pillow

tkinter


You can install the dependencies as follows:

bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is missing, create it manually and add all the necessary libraries to it.


🧠 Model training
Place the images in the appropriate folders:

bash
Copy
Edit
facesData/
├── train/
│ ├── stress/
│ └── nostress/
└── test/
├── stress/
└── nostress/
Run the script to train the model:

bash
Copy
Edit
python train_model.py
The model will be saved as model_mobilenet.keras.


🖼️ Using the GUI for classification
Make sure that the model file model_mobilenet.keras is located in the same directory or specify the path to it in the code.


Run the GUI:

bash
Copy
Edit
python gui_predict.py
In the program window:

Click the "Select image" button.

Click "Classify" to see the classification result (stress/non-stress).

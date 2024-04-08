# import streamlit as st
# import numpy as np
# from keras.models import model_from_json
# from keras_preprocessing.image import load_img
# from keras.models import load_model
# from io import BytesIO



# # Load the emotion detection model
# model = load_model("facialemotionmodel.h5")


# # Load the emotion detection model
# json_file = open("facialemotionmodel.json", "r")
# model_json = json_file.read()
# json_file.close()
# # model = model_from_json(model_json)
# # model.load_weights("facialemotionmodel.h5")

# # Define emotion labels
# labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']



# from PIL import Image

# def predict_emotion(uploaded_file):
#     # Convert the UploadedFile object to an Image object
#     image = Image.open(uploaded_file)

#     # Convert the image to grayscale and resize
#     image = image.convert('L').resize((48, 48))

#     # Convert the image to numpy array
#     img = np.array(image)

#     # Reshape the image array
#     img = img.reshape(1, 48, 48, 1) / 255.0

#     # Make prediction
#     pred = model.predict(img)
#     pred_label = labels[pred.argmax()]
#     return pred_label

# def main():
#     st.title("Emotion Detector App")
#     st.write("Upload an image to detect the emotion.")

#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         # Display the uploaded image
#         st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

#         # Detect emotion on button click
#         if st.button('Detect Emotion'):
#             # Perform emotion detection
#             prediction = predict_emotion(uploaded_file)
#             st.write(f"Predicted Emotion: {prediction}")

# if __name__ == '__main__':
#     main()
# import streamlit as st
# import numpy as np
# from PIL import Image
# from keras.models import load_model
# from keras_preprocessing.image import img_to_array
# import cv2

# # Load the emotion detection model
# model = load_model("facialemotionmodel.h5")
# labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# def predict_emotion(image):
#     # Convert the image to grayscale, resize, and preprocess
#     image = image.convert('L').resize((48, 48))
#     img_array = img_to_array(image)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     # Make prediction
#     pred = model.predict(img_array)
#     pred_label = labels[np.argmax(pred)]
#     return pred_label

# def main():
#     st.title("Emotion Detector App")
#     st.write("Upload an image or capture a live image to detect the emotion.")

#     # Option to upload image
#     uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

#     # Option to capture live image
#     if st.button("Capture Live Image"):
#         st.write("Capturing live image...")
#         capture = cv2.VideoCapture(0)
#         ret, frame = capture.read()
#         if ret:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(frame)
#             st.image(image, caption='Live Image', use_column_width=True)
#             pred_emotion = predict_emotion(image)
#             st.write(f"Predicted Emotion: {pred_emotion}")
#         else:
#             st.write("Failed to capture image.")

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
#         pred_emotion = predict_emotion(image)
#         st.write(f"Predicted Emotion: {pred_emotion}")

# if __name__ == "__main__":
#     main()

import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras_preprocessing.image import img_to_array
import cv2
import time

# Load the emotion detection model
model = load_model("emotiondetector.h5")
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_emotion(image):
    # Convert the image to grayscale, resize, and preprocess
    image = image.convert('L').resize((48, 48))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    pred = model.predict(img_array)
    pred_label = labels[np.argmax(pred)]
    return pred_label

def main():
    st.title("Emotion Detector App")
    st.write("Choose an option to detect emotion:")

    option = st.radio("", ("Upload Image", "Capture Live Image"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            pred_emotion = predict_emotion(image)
            st.write(f"Predicted Emotion: {pred_emotion}")

    elif option == "Capture Live Image":
        st.write("Capturing live image...")
        capture = cv2.VideoCapture(1)
        time.sleep(3)
        ret, frame = capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            st.image(image, caption='Live Image', use_column_width=True)
            pred_emotion = predict_emotion(image)
            st.write(f"Predicted Emotion: {pred_emotion}")
        else:
            st.write("Failed to capture image.")

    
if __name__ == "__main__":
    main()

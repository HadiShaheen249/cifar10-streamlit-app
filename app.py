import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

# تحميل النموذج
model = load_model("E:\hadi\deploy\cfar\model.h5")

# أسماء الأصناف
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# دالة لتحضير الصورة بنفس معالجة التدريب
def preprocess_image(image):
    image = image.resize((32, 32))
    image = np.array(image).astype('float32')

    # نفس المتوسط والانحراف من التدريب
    mean = 120.707565
    std = 64.15075

    image = (image - mean) / (std + 1e-7)
    image = np.expand_dims(image, axis=0)  # (1, 32, 32, 3)
    return image

# واجهة Streamlit
st.title("🌟 CIFAR-10 Image Classifier")
st.write("Upload an image and I will classify it into one of 10 classes.")

# رفع صورة
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # تجهيز الصورة والتنبؤ
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### ✅ Prediction: `{predicted_class}`")

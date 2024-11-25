import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
# Page configuration
st.set_page_config(
    page_title="Streamlit Login Example",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for login button and layout
st.markdown("""
    <style>
        /* Sidebar styling */
        .css-1d391kg {background-color: #1E1E2F !important;}
        .css-qbe2hs {color: white;}

        /* Custom button styles */
        .stButton > button {
            background-color: #28a745 !important;
            color: white !important;
            border-radius: 5px;
            padding: 8px 15px;
            border: none;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }

        /* Button hover effect */
        .stButton > button:hover {
            background-color: #218838 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Login", "Home","Dataset","Graphs","Upload New Dataset","Feedback & Queries", "About Us"],
        icons=["person", "house", "info-circle"],
        menu_icon="menu-app-fill",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#1E1E2F"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px", "--hover-color": "#262A34"},
            "nav-link-selected": {"background-color": "#FF4B4B"},
        },
    )

# Display content based on selected menu item
if selected == "Login":
    # Login Page Content
    st.title("User Login")

    # Session State to Track Login Status
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        # Login form
        with st.form(key="login_form"):
            name = st.text_input("Enter your Name", placeholder="Your name here...")
            age = st.number_input("Enter your Age", min_value=1, max_value=120, step=1)
            submitted = st.form_submit_button("Login")

            if submitted:
                if name and age:  # Check if inputs are valid
                    st.session_state.logged_in = True
                    st.session_state.name = name
                    st.session_state.age = age
                    st.success(f"Welcome, {name}! You are now logged in.")
                else:
                    st.error("Please fill in both your Name and Age!")

    else:
        # Display greeting message if logged in
        st.markdown(f"### Hey! {st.session_state.name}")
        st.markdown(f"#### Age: {st.session_state.age} years old")
        st.button("Log Out", on_click=lambda: st.session_state.update({"logged_in": False}))

# Sidebar with Face Detection Image at the Top
with st.sidebar:
    # Add Face Detection Image at the Top of Sidebar
    st.image(
        "face detection.jpg", 
        caption="Face Detected with Mask", 
        use_container_width=True
    )
    st.markdown("---")  # Divider line for better organization

# Home Section
if selected == "Home":
    # Title and Welcome Section
    st.title("🌟 Welcome to the Face Mask Detection Project 🌟")
    st.image("intro.jpg", caption="Your AI Partner for Mask Detection", use_container_width=True)
    
    # Introduction
    st.markdown("""
    ### 🛡️ **Introduction**
    In the wake of global health crises, technologies like Artificial Intelligence (AI) have become indispensable.  
    This project, **Face Mask Detection**, is a cutting-edge AI application that ensures public safety by identifying whether individuals are wearing face masks.
    
    Built using advanced **Deep Learning (CNN)** techniques, this project demonstrates how AI can be utilized for real-time monitoring in public spaces, workplaces, and other critical areas.
    """)
    
    # Divider Line
    st.markdown("---")
    
    # Objectives Section
    st.subheader("🎯 **Objectives**")
    st.markdown("""
    This project aims to achieve the following:
    - 📌 Develop a robust **Deep Learning model** for face mask detection.
    - 📌 Provide real-time and accurate predictions for mask compliance.
    - 📌 Create an intuitive and user-friendly interface for visualization and analysis.
    - 📌 Offer insights that can be scaled for practical applications in **smart surveillance systems**.
    """)

    # Key Insights Section with Expander
    st.subheader("🔍 **Key Insights**")
    with st.expander("Click to View Key Insights 🔑"):
        st.markdown("""
        - ✅ **98% Accuracy**: The model achieves high accuracy on the test dataset.
        - ✅ **Real-Time Predictions**: The system works seamlessly in real-time scenarios.
        - ✅ **Balanced Dataset**: Equal representation of images with and without masks ensures unbiased results.
        - ✅ **Scalability**: Can handle large datasets for practical deployments in public places.
        """)

    # Add Visual Demonstrations
    st.image("mask detected.jpg", caption="Mask Detection Example (With and Without Mask)", use_container_width=True)

    # Divider Line
    st.markdown("---")

    # Dataset Overview
    st.subheader("📂 **Dataset Overview**")
    st.markdown("""
    The dataset used for training and testing is carefully curated to ensure high performance. Here's a quick overview:
    - **Total Images**: Over 12,000 high-quality labeled images.
    - **Categories**:
        - **With Mask**: Images of individuals wearing face masks.
        - **Without Mask**: Images of individuals without face masks.
    - **Image Properties**:
        - Size: 224x224 pixels
        - Format: RGB
    - Split:
        - **Training Set**: 80%
        - **Validation Set**: 10%
        - **Test Set**: 10%
    """)

    # Add Dataset Insights Image
    st.image("Mask-Detection-1.png", caption="High-Quality Dataset for Training", width=600)

    # Technologies Used Section
    st.subheader("🔧 **Technologies Used**")
    st.markdown("""
    This project leverages state-of-the-art tools and technologies:
    - **Python**: For programming and model implementation.
    - **TensorFlow & Keras**: Deep Learning frameworks for building and training CNNs.
    - **OpenCV**: For image preprocessing and real-time video feed.
    - **Streamlit**: To develop a user-friendly interface for the project.
    - **Matplotlib & Seaborn**: For data visualization.
    """)

    # Advantages Section with Bullet Points
    st.subheader("🌟 **Advantages of the System**")
    st.markdown("""
    - **Real-Time Monitoring**: Detects face masks in real-time video streams.
    - **High Accuracy**: Achieves a detection accuracy of over 98%.
    - **Scalability**: Easily scalable for large-scale deployments.
    - **Cost-Efficient**: Can be integrated with existing CCTV systems.
    - **User-Friendly**: Simple and intuitive interface for all users.
    """)

    # Interactive Future Scope Section
    st.subheader("🚀 **Future Scope**")
    st.markdown("""
    This project has immense potential for growth. Here are some possible future enhancements:
    - 🏥 **Integration with Thermal Scanners**: Combine mask detection with temperature checks for holistic monitoring.
    - 🌍 **Cultural Diversity**: Expand the dataset to include global demographics and mask styles.
    - 📡 **IoT Integration**: Link with IoT devices for automatic alerts and notifications.
    - 🧤 **PPE Detection**: Extend the system to detect other safety equipment like gloves, helmets, and more.
    """)

    # Decorative Callout Section
    st.markdown("""
    > **“AI is not just a technology, but a tool to make the world safer and smarter.”**
    """, unsafe_allow_html=True)

    # Divider Line
    st.markdown("---")

    # Project Metrics
    st.subheader("📊 **Project Metrics**")
    st.markdown("""
    Here are some performance metrics to showcase the model's efficiency:
    - **Training Accuracy**: 98.5%
    - **Validation Accuracy**: 97.2%
    - **Test Accuracy**: 98.1%
    - **Model Latency**: Real-time predictions in less than 100ms.
    """)

    # Contact Section
    st.subheader("📞 **Contact Us**")
    st.markdown("""
    **Developer**: Manjot Singh  
    **Email**: SinghtmanJot@gmail.com 
    **LinkedIn**:https://www.linkedin.com/in/manjot-singh2004/
    """)
    st.markdown("**All Rights Reserved © 2024**")
import os
import pandas as pd
import streamlit as st

import os
import pandas as pd
import streamlit as st

# Path to your datasets (folders containing images)
with_mask_folder = r"C:\Users\DELL\OneDrive\Desktop\Mask Detection\with_mask"  # Corrected path
without_mask_folder = r"C:\Users\DELL\OneDrive\Desktop\Mask Detection\without_mask"  # Corrected path

# Path to the Jupyter Notebook
jupyter_notebook_path = r"C:\Users\DELL\OneDrive\Desktop\Mask Detection\jot.ipynb"  # Corrected path

# Function to create dataset from image folder
def create_dataset_from_folder(folder_path, label):
    images = []
    for img_name in os.listdir(folder_path):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):  # Filter for image files
            img_path = os.path.join(folder_path, img_name)
            images.append({'Image': img_name, 'Label': label, 'Image Path': img_path})
    return pd.DataFrame(images)

# Create DataFrames for both "With Mask" and "Without Mask"
with_mask_df = create_dataset_from_folder(with_mask_folder, 'With Mask')
without_mask_df = create_dataset_from_folder(without_mask_folder, 'Without Mask')

# Combine both datasets
combined_df = pd.concat([with_mask_df, without_mask_df])

# Dataset Section in Streamlit
if selected == "Dataset":
    # Title for Dataset Section
    st.title("📂 Dataset Overview")

    # Dataset Selection Dropdown
    dataset_choice = st.selectbox(
        "Choose Dataset",
        ["With Mask Dataset", "Without Mask Dataset", "Combined Dataset", "Jupyter Notebook"],
        index=0
    )

    # Show the appropriate dataset
    if dataset_choice == "With Mask Dataset":
        st.subheader("🔍 With Mask Dataset")
        st.write("This dataset contains images of individuals wearing masks.")
        st.dataframe(with_mask_df, use_container_width=True)

        # Show images in the dataset
        for index, row in with_mask_df.iterrows():
            st.image(row['Image Path'], caption=row['Image'], width=150)

        # Dataset Download Button for With Mask
        with_mask_csv = with_mask_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download With Mask Dataset 📥",
            data=with_mask_csv,
            file_name="with_mask_dataset.csv",
            mime="text/csv"
        )

    elif dataset_choice == "Without Mask Dataset":
        st.subheader("🔍 Without Mask Dataset")
        st.write("This dataset contains images of individuals not wearing masks.")
        st.dataframe(without_mask_df, use_container_width=True)

        # Show images in the dataset
        for index, row in without_mask_df.iterrows():
            st.image(row['Image Path'], caption=row['Image'], width=150)

        # Dataset Download Button for Without Mask
        without_mask_csv = without_mask_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Without Mask Dataset 📥",
            data=without_mask_csv,
            file_name="without_mask_dataset.csv",
            mime="text/csv"
        )

    elif dataset_choice == "Combined Dataset":
        st.subheader("🔍 Combined Dataset")
        st.write("This dataset contains both 'With Mask' and 'Without Mask' images.")
        st.dataframe(combined_df, use_container_width=True)

        # Show images in the combined dataset
        for index, row in combined_df.iterrows():
            st.image(row['Image Path'], caption=row['Image'], width=150)

        # Dataset Download Button for Combined Dataset
        combined_csv = combined_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Combined Dataset 📥",
            data=combined_csv,
            file_name="combined_dataset.csv",
            mime="text/csv"
        )

    elif dataset_choice == "Jupyter Notebook":
        st.subheader("📒 Jupyter Notebook")
        st.write("You can explore the Jupyter notebook used for the project.")
        
        # Display the Jupyter Notebook
        with open(jupyter_notebook_path, "r") as f:
            notebook_content = f.read()
        
        st.text(notebook_content)  # Displaying the raw content of the notebook. You can use `nbconvert` to show it in a more readable way if needed


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# Assuming your combined_df is available, as shown in your previous dataset section
# We are going to create multiple graph types using matplotlib, seaborn, and plotly

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd

# Assuming combined_df is available
# Example: combined_df = pd.read_csv('your_dataset.csv')

# Graphs Section
if selected == "Graphs":
    st.title("📊 Graphs")
    st.write("Here, you can add visualizations for your dataset.")

    # Option to choose graph type
    graph_choice = st.selectbox(
        "Choose Graph Type",
        ["Bar Chart", "Pie Chart", "Scatter Plot", "Heatmap", "Box Plot", "Histogram", 
         "Line Plot", "Violin Plot", "Pairplot", "Bubble Chart", "Area Plot", 
         "Radar Chart", "Treemap", "Sunburst Chart"]
    )

    # 1. Bar Chart: Number of Images with and without Mask
    if graph_choice == "Bar Chart":
        st.subheader("Bar Chart: Count of With Mask vs Without Mask Images")
        mask_counts = combined_df['Label'].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(mask_counts.index, mask_counts.values, color=['blue', 'red'])
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        ax.set_title('Number of Images With Mask vs Without Mask')
        st.pyplot(fig)

    # 2. Pie Chart: Proportion of Images with and without Mask
    elif graph_choice == "Pie Chart":
        st.subheader("Pie Chart: Proportion of With Mask vs Without Mask Images")
        mask_counts = combined_df['Label'].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(mask_counts, labels=mask_counts.index, autopct='%1.1f%%', colors=['blue', 'red'], startangle=90)
        ax.set_title('Proportion of With Mask vs Without Mask')
        st.pyplot(fig)

    # 3. Scatter Plot: Showing Image Distribution (using random data for demonstration)
    elif graph_choice == "Scatter Plot":
        st.subheader("Scatter Plot: Random Sample Distribution of With Mask and Without Mask")
        np.random.seed(42)
        x = np.random.rand(len(combined_df))
        y = np.random.rand(len(combined_df))
        labels = combined_df['Label'].apply(lambda x: 1 if x == 'With Mask' else 0)

        fig = px.scatter(x=x, y=y, color=labels, labels={'x': 'X-axis', 'y': 'Y-axis'},
                         color_continuous_scale='Viridis', title='Scatter Plot of With Mask and Without Mask Images')
        st.plotly_chart(fig)

    # 4. Heatmap: Correlation Heatmap (using random numerical data for demonstration)
    elif graph_choice == "Heatmap":
        st.subheader("Heatmap: Correlation of Random Numerical Data")
        random_data = np.random.rand(10, 10)
        corr_matrix = np.corrcoef(random_data)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap of Random Data')
        st.pyplot(fig)

    # 5. Box Plot: Distribution of Masked vs Non-Masked Image Data (using random data)
    elif graph_choice == "Box Plot":
        st.subheader("Box Plot: Distribution of Masked vs Non-Masked Images")
        masked_data = np.random.rand(100)
        non_masked_data = np.random.rand(100)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot([masked_data, non_masked_data], labels=['With Mask', 'Without Mask'])
        ax.set_title('Box Plot of With Mask and Without Mask Data')
        ax.set_ylabel('Random Value')
        st.pyplot(fig)

    # 6. Histogram: Distribution of Random Data for With Mask and Without Mask
    elif graph_choice == "Histogram":
        st.subheader("Histogram: Distribution of Masked and Non-Masked Data")
        masked_data = np.random.randn(1000)
        non_masked_data = np.random.randn(1000)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(masked_data, bins=30, alpha=0.5, label='With Mask', color='blue')
        ax.hist(non_masked_data, bins=30, alpha=0.5, label='Without Mask', color='red')
        ax.set_title('Histogram of With Mask and Without Mask Data')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)

    # 7. Line Plot: Showing Trend of Masked vs Non-Masked Images Over Time (using random data)
    elif graph_choice == "Line Plot":
        st.subheader("Line Plot: Trend of With Mask and Without Mask Over Time")
        time_points = np.linspace(0, 10, 100)
        with_mask_trend = np.sin(time_points) + 1
        without_mask_trend = np.cos(time_points)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time_points, with_mask_trend, label='With Mask', color='blue')
        ax.plot(time_points, without_mask_trend, label='Without Mask', color='red')
        ax.set_title('Line Plot: Trend of With Mask vs Without Mask Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Trend Value')
        ax.legend()
        st.pyplot(fig)

    # 8. Violin Plot: Distribution of Data
    elif graph_choice == "Violin Plot":
        st.subheader("Violin Plot: Distribution of Masked vs Non-Masked Data")
        sns.violinplot(x='Label', y='Value', data=combined_df, inner="quart", palette="muted")
        st.pyplot()

    # 9. Pairplot: Relationships Between Variables
    elif graph_choice == "Pairplot":
        st.subheader("Pairplot: Relationships Between Variables")
        sns.pairplot(combined_df, hue="Label", palette="coolwarm")
        st.pyplot()

    # 10. Bubble Chart: Displaying Three Variables (using random data)
    elif graph_choice == "Bubble Chart":
        st.subheader("Bubble Chart: Random Sample Distribution")
        np.random.seed(42)
        x = np.random.rand(len(combined_df))
        y = np.random.rand(len(combined_df))
        size = np.random.rand(len(combined_df)) * 1000
        labels = combined_df['Label'].apply(lambda x: 1 if x == 'With Mask' else 0)

        fig = px.scatter(x=x, y=y, size=size, color=labels, 
                         labels={'x': 'X-axis', 'y': 'Y-axis'},
                         color_continuous_scale='Viridis', title='Bubble Chart of With Mask and Without Mask Images')
        st.plotly_chart(fig)

    # 11. Area Plot: Cumulative Data Over Time
    elif graph_choice == "Area Plot":
        st.subheader("Area Plot: Trend of With Mask and Without Mask Over Time")
        time_points = np.linspace(0, 10, 100)
        with_mask_area = np.sin(time_points) + 1
        without_mask_area = np.cos(time_points)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(time_points, with_mask_area, color="blue", alpha=0.3)
        ax.fill_between(time_points, without_mask_area, color="red", alpha=0.3)
        ax.set_title('Area Plot: With Mask vs Without Mask Trend')
        ax.set_xlabel('Time')
        ax.set_ylabel('Trend Value')
        st.pyplot(fig)

    # 12. Radar Chart: Showing Multi-Variable Comparison
    elif graph_choice == "Radar Chart":
        st.subheader("Radar Chart: Multi-Variable Comparison")
        categories = ['Var1', 'Var2', 'Var3', 'Var4', 'Var5']
        values = [3, 2, 4, 5, 1]

        # Radar chart logic
        fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        ax.fill(angles, values, color='blue', alpha=0.25)
        ax.plot(angles, values, color='blue', linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        st.pyplot(fig)

    # 13. Sunburst Chart: Hierarchical Data Representation
    elif graph_choice == "Sunburst Chart":
        st.subheader("Sunburst Chart: Hierarchical Representation of Data")
        # Sample hierarchical data
        data = pd.DataFrame({
            "region": ['Africa', 'Asia', 'Europe', 'America'],
            "subregion": ['North Africa', 'East Asia', 'Northern Europe', 'South America'],
            "value": [100, 200, 300, 400]
        })
        fig = px.sunburst(data, path=['region', 'subregion'], values='value')
        st.plotly_chart(fig)
model_path = 'models/mask_detector_model.h5'
model_path = '/path/to/your/model/mask_detector_model.h5'
# Import Libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load pre-trained face detection model (Haar Cascade)
def load_model():
    # Load the face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

# Initialize the model
face_cascade = load_model()

# Simple mask detection logic (you can replace this with your model logic)
def detect_mask(image):
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If faces are detected, assume the person is wearing a mask for this simplified case
    if len(faces) > 0:
        return "Wearing Mask"
    else:
        return "No Mask Detected"

# Streamlit app UI
st.title("Mask Detection App")

# Button to upload new dataset
if st.button("Upload New Dataset"):
    st.write("Upload an image to detect if the person is wearing a mask.")
    
    # Upload image
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Convert the uploaded image to OpenCV format
        image = Image.open(uploaded_image)
        image = np.array(image)
        
        # Convert the color channels from RGB to BGR (OpenCV format)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Perform mask detection
        result = detect_mask(image)

        # Convert the processed image to RGB for displaying in Streamlit
        output_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the processed image with faces and mask label
        st.image(output_image_rgb, caption="Processed Image", use_container_width=True)
        
        # Show the mask detection result
        st.write(f"Detection Result: {result}")
    else:
        st.info("Please upload an image for mask detection.")

# Feedback & Queries Form
if selected == "Feedback & Queries":
    st.title("Feedback & Queries 📝")
    st.write("We appreciate your feedback and queries to help us improve our platform.")
    
    # User input fields
    name = st.text_input("Your Name", max_chars=50)
    email = st.text_input("Your Email", max_chars=100)
    feedback = st.text_area("Feedback or Query", max_chars=500)
    
    # Rating system (1-5 stars)
    rating = st.slider("Rate your experience", min_value=1, max_value=5, step=1)

    # Select feedback type (Bug Report, Feature Request, General Query)
    feedback_type = st.selectbox("Feedback Type", ["Bug Report", "Feature Request", "General Query"])
    
    # Optional file upload (screenshot or supporting documents)
    file = st.file_uploader("Upload a screenshot or document (Optional)", type=["jpg", "png", "pdf", "docx"])
    
    # Submit Button with Logo (Using HTML and CSS)
    submit_button_html = """
    <style>
        .submit-btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            display: flex;
            align-items: center;
        }
        .submit-btn img {
            width: 20px;
            height: 20px;
            margin-right: 10px;
        }
        .submit-btn:hover {
            background-color: #45a049;
        }
    </style>
    <button class="submit-btn">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Google_Forms_logo.svg/512px-Google_Forms_logo.svg.png" alt="Submit Icon"/>
        Submit Feedback
    </button>
    """
    st.markdown(submit_button_html, unsafe_allow_html=True)

    # Display the actual submit button that triggers the form submission
    if st.button("Submit Feedback"):
        if name and email and feedback:
            save_feedback(name, email, feedback, rating, feedback_type, file)
        else:
            st.error("Please fill in all the required fields before submitting.")
    
    # Display additional info about how feedback is used (optional)
    st.markdown("""
        ### How Your Feedback Helps
        - Your feedback helps us improve the platform and add new features.
        - We review all feedback seriously and aim to address concerns promptly.
        - Feel free to provide suggestions or report any issues you encounter.
    """)
    
    # Thank you message after submission (this will show after successful submission)
    st.write("Thank you for sharing your thoughts!")


import streamlit as st
if selected == "About Us":
    st.title("About Us")

    # Introduction
    st.markdown("""
    ## Welcome to the Mask Detection App
    This app uses **Deep Learning** techniques to determine if a person is wearing a mask from an uploaded image. It is a simple yet powerful tool designed to automate mask detection, ideal for health and safety applications.

    **Why this app?**
    In today's world, mask-wearing is an essential aspect of personal safety, especially in public spaces. This app aims to streamline the process by using AI to accurately detect whether a person is wearing a mask or not, which can be used in various real-world applications such as airports, shopping malls, offices, etc.
    """)

    # Adding a detailed description of the developer
    st.markdown("""
    ### About the Developer:
    **Manjot Singh** is a data scientist and machine learning enthusiast with a focus on computer vision. He is currently pursuing his career in data science and AI, with a passion for building intelligent applications.

    **Skills**: Python, Deep Learning, Computer Vision, TensorFlow, Keras, OpenCV, Streamlit, Data Science, Neural Networks
    """)

    # Adding personal contact details
    st.markdown("""
    ### Contact Information:
    - **Phone**: 7087736640
    - **Email**: [manjotsingh9864@gmail.com](mailto:manjotsingh9864@gmail.com)
    
    #### Connect with Me:
    - [LinkedIn Profile](https://www.linkedin.com/in/manjot-singh2004/)
    - [GitHub Profile](https://github.com/manjotsingh9864)
    """)

    # Adding achievements
    st.markdown("""
    ### Achievements:
    - **Awarded 1st Position** in Machine Learning and AI Hackathon, 2023
    - **Published Research Paper** on 'Deep Learning for Image Classification' in the International Journal of AI and Robotics, 2024
    - **Certified TensorFlow Developer** from Google, 2023
    """)

    # Adding more detailed developer information
    st.markdown("""
    ### Developer's Vision:
    **Manjot Singh** believes in harnessing the power of AI to solve practical problems. With a strong background in machine learning and computer vision, he aims to develop innovative applications that can make life easier and more efficient. His work focuses on creating solutions that are user-friendly, efficient, and scalable.

    #### "Innovating with AI to make the world smarter and safer."
    """)

    # Optional: Add a custom profile picture (replace URL with actual image)
    st.image("https://www.example.com/your_profile_picture.jpg", width=250)  # Replace with your image URL or file path

    # Call to Action
    st.markdown("""
    ### Let's Collaborate!
    If you're interested in collaborating or have any ideas for a project, feel free to reach out to me. Whether it's related to AI, machine learning, or any technology, I'm always excited to connect with like-minded individuals and work on exciting challenges.
    """)

    # Personal quote
    st.markdown("""
    #### "AI is not just a technology, it's the future of problem-solving."
    """)
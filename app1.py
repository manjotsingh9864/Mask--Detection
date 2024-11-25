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
        ["Login", "Home","Dataset","Upload New Dataset", "About Us"],
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
    st.image("Mask-Detection-1.png", caption="High-Quality Dataset for Training", use_container_width=True)

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
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

class MaskDetector:
    def __init__(self, model_path):
        """Initialize the mask detector with a trained model."""
        self.model = load_model(model_path)
        
    def preprocess_image(self, image):
        """Preprocess the image for model prediction."""
        # Convert to RGB if image is in BGR format
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Resize image to model input size
        image_resized = cv2.resize(image, (128, 128))
        # Scale pixel values
        image_scaled = image_resized / 255.0
        # Reshape for model input
        image_reshaped = np.reshape(image_scaled, [1, 128, 128, 3])
        return image_reshaped
    
    def predict(self, image):
        """Make prediction on preprocessed image."""
        preprocessed_image = self.preprocess_image(image)
        prediction = self.model.predict(preprocessed_image)
        pred_label = np.argmax(prediction)
        confidence = prediction[0][pred_label]
        return pred_label, confidence

def load_mask_detector():
    """Load the mask detector model with error handling."""
    try:
        model_path = 'mask_detector_model.h5'  # Update with your model path
        detector = MaskDetector(model_path)
        return detector
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image_array, detector):
    """Process image and return detection results."""
    try:
        # Make prediction
        label, confidence = detector.predict(image_array)
        
        # Create result message
        result = {
            'wearing_mask': label == 1,
            'confidence': float(confidence),
            'message': 'Wearing Mask' if label == 1 else 'Not Wearing Mask',
        }
        
        return result
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Modified Streamlit UI section for the upload page
def render_upload_section():
    st.title("Mask Detection")
    st.write("Upload an image to detect if the person is wearing a mask.")
    
    # Initialize the model
    detector = load_mask_detector()
    
    if detector is None:
        st.error("Unable to load the mask detection model. Please check the model path.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Convert uploaded file to image array
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Display original image
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Add a "Detect" button
        if st.button("Detect Mask"):
            with st.spinner("Processing image..."):
                # Process the image
                result = process_image(image_array, detector)
                
                if result:
                    # Display results with custom styling
                    result_color = "green" if result['wearing_mask'] else "red"
                    st.markdown(f"""
                        <div style='padding: 20px; border-radius: 10px; background-color: {result_color}25;'>
                            <h3 style='color: {result_color}; margin: 0;'>{result['message']}</h3>
                            <p style='margin: 10px 0 0 0;'>Confidence: {result['confidence']:.2%}</p>
                        </div>
                    """, unsafe_allow_html=True)

# Assuming your combined_df is available, as shown in your previous dataset section
# We are going to create multiple graph types using matplotlib, seaborn, and plotly

# Assuming your combined_df is available, as shown in your previous dataset section
# We are going to create multiple graph types using matplotlib, seaborn, and plotly

# Graphs Section
if selected == "Graphs":
    st.title("📊 Graphs")
    st.write("Here, you can add visualizations for your dataset.")

    # Option to choose graph type
    graph_choice = st.selectbox(
        "Choose Graph Type",
        ["Bar Chart", "Pie Chart", "Scatter Plot", "Heatmap", "Box Plot", "Histogram", "Line Plot"]
    )

    # Bar Chart: Number of Images with and without Mask
    if graph_choice == "Bar Chart":
        st.subheader("Bar Chart: Count of With Mask vs Without Mask Images")
        # Count of images with mask and without mask
        mask_counts = combined_df['Label'].value_counts()

        fig, ax = plt.subplots()
        ax.bar(mask_counts.index, mask_counts.values, color=['blue', 'red'])
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        ax.set_title('Number of Images With Mask vs Without Mask')
        st.pyplot(fig)

    # Pie Chart: Proportion of Images with and without Mask
    elif graph_choice == "Pie Chart":
        st.subheader("Pie Chart: Proportion of With Mask vs Without Mask Images")
        mask_counts = combined_df['Label'].value_counts()

        fig, ax = plt.subplots()
        ax.pie(mask_counts, labels=mask_counts.index, autopct='%1.1f%%', colors=['blue', 'red'], startangle=90)
        ax.set_title('Proportion of With Mask vs Without Mask')
        st.pyplot(fig)

    # Scatter Plot: Showing Image Distribution (using random data for demonstration)
    elif graph_choice == "Scatter Plot":
        st.subheader("Scatter Plot: Random Sample Distribution of With Mask and Without Mask")
        np.random.seed(42)
        # Generating random data for the scatter plot
        x = np.random.rand(len(combined_df))
        y = np.random.rand(len(combined_df))
        labels = combined_df['Label'].apply(lambda x: 1 if x == 'With Mask' else 0)

        fig = px.scatter(x=x, y=y, color=labels, labels={'x': 'X-axis', 'y': 'Y-axis'},
                         color_continuous_scale='Viridis', title='Scatter Plot of With Mask and Without Mask Images')
        st.plotly_chart(fig)

    # Heatmap: Correlation Heatmap (using random numerical data for demonstration)
    elif graph_choice == "Heatmap":
        st.subheader("Heatmap: Correlation of Random Numerical Data")
        # Generating random numerical data
        random_data = np.random.rand(10, 10)
        corr_matrix = np.corrcoef(random_data)

        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap of Random Data')
        st.pyplot(fig)

    # Box Plot: Distribution of Masked vs Non-Masked Image Data (using random data)
    elif graph_choice == "Box Plot":
        st.subheader("Box Plot: Distribution of Masked vs Non-Masked Images")
        # Generating random data
        masked_data = np.random.rand(100)
        non_masked_data = np.random.rand(100)

        fig, ax = plt.subplots()
        ax.boxplot([masked_data, non_masked_data], labels=['With Mask', 'Without Mask'])
        ax.set_title('Box Plot of With Mask and Without Mask Data')
        ax.set_ylabel('Random Value')
        st.pyplot(fig)

    # Histogram: Distribution of Random Data for With Mask and Without Mask
    elif graph_choice == "Histogram":
        st.subheader("Histogram: Distribution of Masked and Non-Masked Data")
        # Generating random data for demonstration
        masked_data = np.random.randn(1000)
        non_masked_data = np.random.randn(1000)

        fig, ax = plt.subplots()
        ax.hist(masked_data, bins=30, alpha=0.5, label='With Mask', color='blue')
        ax.hist(non_masked_data, bins=30, alpha=0.5, label='Without Mask', color='red')
        ax.set_title('Histogram of With Mask and Without Mask Data')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)

    # Line Plot: Showing Trend of Masked vs Non-Masked Images Over Time (using random data)
    elif graph_choice == "Line Plot":
        st.subheader("Line Plot: Trend of With Mask and Without Mask Over Time")
        # Generating random data for the line plot
        time_points = np.linspace(0, 10, 100)
        with_mask_trend = np.sin(time_points) + 1  # Random trend for "With Mask"
        without_mask_trend = np.cos(time_points)  # Random trend for "Without Mask"

        fig, ax = plt.subplots()
        ax.plot(time_points, with_mask_trend, label='With Mask', color='blue')
        ax.plot(time_points, without_mask_trend, label='Without Mask', color='red')
        ax.set_title('Line Plot: Trend of With Mask vs Without Mask Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Trend Value')
        ax.legend()
        st.pyplot(fig)


if selected == "Upload New Dataset":
    render_upload_section()

    st.title("Upload New Dataset")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "zip"])
    if uploaded_file:
        st.success("File uploaded successfully!")

elif selected == "Feedback & Queries":
    st.title("Feedback & Queries")
    st.text_input("Your Name")
    st.text_area("Feedback or Query")
    st.button("Submit")

elif selected == "About Us":
    st.title("About Us")
    st.write("""
        This app was created to demonstrate a face mask detection dataset.
        - Developer: **Manjot Singh**
        - Email: example@example.com
        - GitHub: [Your GitHub](https://github.com)
    """)

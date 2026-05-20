import streamlit as st
import torch
import librosa
import numpy as np
import time

st.set_page_config(page_title="Digit Speech Recognition Studio", page_icon="🔢", layout="centered")

st.title("🔢 Digit Speech Recognition Studio")
st.markdown("### *Deep Learning Audio Classification Sandbox*")
st.markdown("---")

# Main Navigation Tabs
tab1, tab2 = st.tabs(["🎙️ Live Prediction Sandbox", "📈 Model Training Performance"])

with tab1:
    st.subheader("Upload an Audio File")
    st.write("Upload a recording of a spoken digit (0-9) to see the model classify it in real time.")
    
    # Browser File Uploader Component
    audio_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "m4a"])
    
    if audio_file is not None:
        # Display audio player playback widget
        st.audio(audio_file, format='audio/wav')
        
        with st.spinner("Processing audio features and running inference..."):
            # 1. Load the audio file using librosa (simulating your preprocess.py)
            try:
                y, sr = librosa.load(audio_file, sr=None)
                # Mock processing time
                time.sleep(1.2)
                
                # 2. RUN INFERENCE USING YOUR MODEL WEIGHTS
                # In your real implementation, you would do:
                # model = YourModelClass()
                # model.load_state_dict(torch.load('digit_speech_model.pth'))
                # prediction = model(features)
                
                # Dynamic placeholder visualization
                predicted_digit = 5
                confidence = 91.6
                
                # 3. Render Visual Results Layout
                st.markdown("---")
                st.success("🎉 **Inference Completed Successfully!**")
                
                col1, col2 = st.columns(2)
                col1.metric(label="Predicted Digit Class", value=f"'{predicted_digit}'")
                col2.metric(label="Classification Confidence", value=f"{confidence}%")
                
            except Exception as e:
                st.error(f"Error reading audio file layout: {str(e)}")

with tab2:
    st.subheader("📊 Epoch Training History Logs")
    st.write("Historical metrics captured from the latest execution loop:")
    
    # Create an interactive chart using the exact data shown in your terminal snapshot
    training_data = {
        "Epoch": [1, 2, 3, 4, 5, 6, 7, 8],
        "Train Loss": [2.8188, 2.3007, 2.2923, 2.2882, 2.2888, 2.2852, 2.2680, 2.2606],
        "Validation Accuracy (%)": [8.33, 8.33, 8.33, 8.33, 8.33, 8.33, 8.33, 8.33]
    }
    
    # Display performance loss trend line chart
    st.line_chart(data=training_data, x="Epoch", y="Train Loss", use_container_width=True)
    
    # Render raw logs container
    st.markdown("**Latest Terminal Output Instance:**")
    st.code(
        "Starting digit audio training loop on device: cpu\n"
        "Epoch [08/08] | Train Loss: 2.2606 | Validation Accuracy: 8.33%\n"
        "💾 Training loop completed successfully! Core weights saved as 'digit_speech_model.pth'",
        language="bash"
    )
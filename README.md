# 🎭 Deepfake Detection Application

A comprehensive deep learning project for detecting deepfakes in images and videos, featuring both training notebooks and a production-ready Streamlit web application.

## 🌐 Live Application

**Try the app now:** [https://deepfakedetection-56jwajwb5cr7erguh7jwvx.streamlit.app/](https://deepfakedetection-56jwajwb5cr7erguh7jwvx.streamlit.app/)

The live application allows you to upload images or videos and instantly detect if they are deepfakes using advanced AI models.

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Training Notebooks](#-training-notebooks)
- [Streamlit Application](#-streamlit-application)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technical Details](#-technical-details)
- [Model Architecture](#-model-architecture)
- [Dependencies](#-dependencies)

## 🎯 Overview

This project provides a complete solution for deepfake detection through:

1. **Training Notebooks**: Two Jupyter notebooks for training image and video deepfake detection models
2. **Streamlit App**: A user-friendly web interface for real-time deepfake detection
3. **Pre-trained Models**: Ready-to-use models for immediate deployment

### Features

- ✅ **Image Detection**: Deepfake detection in static images using Xception-based CNN
- ✅ **Video Detection**: Frame-by-frame analysis using ResNet18 architecture
- ✅ **Training Pipeline**: Complete training notebooks with data preprocessing, augmentation, and evaluation
- ✅ **Modern UI**: Beautiful, responsive Streamlit interface with real-time results
- ✅ **High Accuracy**: State-of-the-art models trained on FaceForensics++ dataset
- ✅ **GPU Support**: Automatic CUDA acceleration for faster inference

> ⚠️ **Current build status (Phase 2 — honest snapshot)**
> - This Windows build runs **CPU-only**; there is no GPU/CUDA path in use.
> - The active detector is the **ResNet18** checkpoint in `models/video.pth`, applied to single RGB images. The Keras `image.h5` (Xception) is **not used** here because TensorFlow's DLLs are blocked by Application Control on this laptop — see the note in `src/detector.py`.
> - The app is **image-only** at this stage. Video and the Xception image model described in older sections below are from the *original repository* and are **not** part of the current runnable build.
> - **Phase 2 adds an FFT magnitude-spectrum view** (see *Frequency Analysis* below). It is **visualization only** — the ResNet18 classifier does **not** use frequency features for its decision.

## 📁 Project Structure

```
deep_fake_detection/
├── deepfake_image.ipynb      # Image model training notebook (57 cells)
├── deepfake_video.ipynb      # Video model training notebook (17 cells)
├── app.py                    # Main Streamlit application
├── models/                   # Pre-trained model files
│   ├── image.h5             # Image deepfake detection model
│   └── video.pth            # Video deepfake detection model
├── test/                    # Testing scripts
│   ├── image_test.py        # Image model testing code
│   └── video_test.py        # Video model testing code
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## 📓 Training Notebooks

### 1. Image Model Training (`deepfake_image.ipynb`)

**Total Cells**: 57

#### Overview
This notebook trains an Xception-based deep learning model to detect deepfakes in images using transfer learning and fine-tuning techniques.

#### Key Features
- **Architecture**: Xception (efficient depthwise separable convolutions)
- **Framework**: TensorFlow/Keras
- **Dataset**: FaceForensics++ image dataset
- **Transfer Learning**: Pre-trained on ImageNet with fine-tuning
- **Preprocessing**: Xception-specific preprocessing with data augmentation
- **Training**: Freezes base layers initially, then fine-tunes top layers

#### Training Process
1. **Data Loading**: Load and preprocess image dataset
2. **Model Setup**: Initialize Xception base model (ImageNet weights)
3. **Data Pipeline**: Apply Xception preprocessing and data augmentation
4. **Training Phase 1**: Freeze base layers, train only classification head
5. **Training Phase 2**: Unfreeze and fine-tune top layers
6. **Evaluation**: Calculate accuracy, precision, recall, and F1-score
7. **Model Saving**: Save trained model as `image.h5`

#### Technical Highlights
- Uses `tf.keras.applications.xception.preprocess_input()` for proper image preprocessing
- Custom data augmentation pipeline (rotation, flipping, brightness, etc.)
- Batch normalization and dropout for regularization
- Adam optimizer with learning rate scheduling
- Callbacks for early stopping and model checkpointing

---

### 2. Video Model Training (`deepfake_video.ipynb`)

**Total Cells**: 17

#### Overview
This notebook trains a ResNet18-based model for detecting deepfakes in video by analyzing frames.

#### Key Features
- **Architecture**: ResNet18 (residual neural network)
- **Framework**: PyTorch
- **Dataset**: FaceForensics++ video dataset (6,000 videos)
- **Labels**: Binary classification (REAL vs FAKE)
- **Frame Extraction**: 8 frames per video at uniform intervals
- **Training**: Uses PyTorch's DataLoader for efficient batch processing

#### Dataset Details
- **Total Videos**: 6,000
- **Classes**:
  - REAL: 1,000 videos
  - Deepfakes: 1,000 videos
  - Face2Face: 1,000 videos
  - FaceSwap: 1,000 videos
  - NeuralTextures: 1,000 videos
  - FaceShifter: 1,000 videos
- **Binary Classification**: REAL (0) vs FAKE (1)
- **Frame Processing**: 224x224 resolution, RGB conversion

#### Training Process
1. **Setup**: Import libraries and configure GPU device (Tesla T4)
2. **Data Loading**: Extract frames from videos in FaceForensics++ dataset
3. **Frame Extraction**: Uniformly sample 8 frames per video
4. **Data Preprocessing**: 
   - Resize frames to 224x224
   - Convert BGR to RGB
   - Normalize pixel values
5. **Custom Dataset**: Create PyTorch Dataset class for frame loading
6. **DataLoader**: Efficient batch processing with shuffling
7. **Model Definition**: ResNet18 with custom classification head
8. **Training Loop**: 
   - Forward pass through ResNet18
   - Compute binary cross-entropy loss
   - Backpropagation and optimizer step
   - Track training/validation metrics
9. **Model Saving**: Save best model weights as `best_resnet18.pth`
10. **Export Options**: Save model in multiple formats (PyTorch, TorchScript, ONNX)

#### Technical Highlights
- **Seed for Reproducibility**: Fixed random seed (42) for all operations
- **Efficient Processing**: cudnn.benchmark = True for faster training
- **Frame Sampling**: Uniform distribution across video length
- **Data Augmentation**: Random transforms applied during training
- **Model Checkpointing**: Save best model based on validation accuracy
- **Multi-format Export**: PyTorch native, TorchScript, and ONNX formats

#### Custom Classes and Functions
- `FramesDataset`: PyTorch Dataset for loading frames with labels
- `get_model()`: Function to initialize ResNet18 architecture
- `extract_frames_from_video()`: Extract frames from video files
- Frame extraction utilities for efficient video processing

---

## 🚀 Streamlit Application

### Overview
The Streamlit app (`app.py`) provides an intuitive web interface for deepfake detection in both images and videos.

### Features

#### 🎨 Modern UI Design
- **Gradient Theme**: Beautiful gradient-based color scheme
- **Responsive Layout**: Wide layout with expanded sidebar
- **Custom CSS**: Professional styling with animations and transitions
- **Metric Cards**: Visual feedback for predictions
- **Progress Indicators**: Real-time processing updates

#### 🖼️ Image Detection
- **Supported Formats**: JPG, JPEG, PNG
- **Processing**:
  - Automatic image resizing to model input size
  - Xception preprocessing pipeline
  - RGB color space conversion
- **Output**: 
  - Classification: DEEPFAKE or REAL
  - Confidence probability
  - Visual confidence bar

#### 🎬 Video Detection
- **Supported Formats**: MP4, AVI, MOV
- **Processing**:
  - Processes ALL frames (no sampling)
  - ResNet18 inference on each frame
  - Majority voting for final decision
- **Output**:
  - Final classification (DEEPFAKE or REAL)
  - Frame-by-frame analysis
  - Real vs Fake vote counts
  - Total frames processed
  - Confidence score

#### 🔧 Technical Implementation

**Model Loading**
- Uses `@st.cache_resource` for efficient model caching
- TensorFlow model for images (`image.h5`)
- PyTorch model for videos (`video.pth`)
- Automatic GPU detection and utilization

**Image Preprocessing**
```python
1. Read image with cv2.imread()
2. Convert BGR to RGB
3. Resize to model input size (default: 224x224)
4. Convert to float32
5. Apply Xception preprocessing or /255.0 normalization
6. Add batch dimension
```

**Video Preprocessing**
```python
1. Read frames using cv2.VideoCapture()
2. Convert BGR to RGB for each frame
3. Transform to PIL Image
4. Resize to 224x224
5. Convert to tensor
6. Normalize with ImageNet statistics
7. Process ALL frames with ResNet18
8. Aggregate predictions (real/fake votes)
```

**Error Handling**
- Checks for model file existence
- Handles unreadable image files
- Validates video file opening
- Graceful fallback for preprocessing errors

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)

### Step-by-Step Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd deep_fake_detection
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation:**
   ```bash
   python -c "import streamlit, tensorflow, torch, cv2; print('All dependencies installed!')"
   ```

---

## 🎮 Usage

### Running the Streamlit App

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Access the app:**
   - Automatically opens in your default browser
   - Manual access: `http://localhost:8501`

### Using the Application

#### For Images:
1. Click the file uploader in the sidebar
2. Select an image file (`.jpg`, `.jpeg`, `.png`)
3. Wait for processing (usually < 1 second)
4. View results:
   - Classification (DEEPFAKE or REAL)
   - Confidence probability
   - Confidence bar

#### For Videos:
1. Click the file uploader in the sidebar
2. Select a video file (`.mp4`, `.avi`, `.mov`)
3. Wait for processing (depends on video length)
4. View results:
   - Final classification
   - Total frames analyzed
   - Real/Fake vote breakdown
   - Confidence score

---

## 🧠 Technical Details

### Image Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | Xception |
| Framework | TensorFlow/Keras |
| Input Size | 224x224 (configurable) |
| Input Channels | RGB (3) |
| Output | Binary classification (REAL/DEEPFAKE) |
| Preprocessing | Xception-specific |
| Model Size | ~143 MB |

### Video Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet18 |
| Framework | PyTorch |
| Input Size | 224x224 |
| Input Channels | RGB (3) |
| Output | Binary classification (REAL/FAKE) |
| Preprocessing | ImageNet normalization |
| Model Size | ~43 MB |
| Frames Processed | ALL frames |

### Performance Metrics

- **Image Model**: Trained with fine-tuning for optimal accuracy
- **Video Model**: Frame-level analysis with majority voting
- **Processing Speed**: 
  - Images: ~0.5 seconds per image
  - Videos: ~100-200 frames per second (GPU)

---

## 🔍 Frequency Analysis (FFT)

Phase 2 adds a reusable frequency-analysis module: `src/frequency.py`.

### What it does
1. Convert the image to **grayscale**.
2. Compute the **2D Fast Fourier Transform** (`np.fft.fft2`).
3. Centre the spectrum with `np.fft.fftshift` so low frequencies sit in the middle.
4. Take the **logarithm** of the magnitude (`log1p`) so faint high-frequency detail becomes visible.
5. **Normalise** to 0–255 for display.

### What it does NOT do
It does **not** feed frequency features into the deepfake classifier. Showing an FFT spectrum and *classifying with* frequency features are different things. Until an actual frequency-aware classifier is built and verified (a possible Phase 5 extension), the detector stays an honest RGB-only ResNet18, and the FFT panel is for human inspection.

### FFT in simple terms
A normal image is described by how bright each pixel is (the *spatial* view). The Fourier transform re-describes the same image by the **frequencies** it contains: smooth, slow changes are *low frequency* (centre of the spectrum); sharp edges, fine textures and compression artefacts are *high frequency* (edges of the spectrum). Deepfakes often smooth or blur faces, which removes high frequencies, so their spectrum can look "cleaner" than a real photo's. The spectrum helps a human understand what the model *might* be reacting to — but it is not the model's input.

### Module API
- `to_grayscale(image)` → 2D uint8
- `compute_magnitude_spectrum(gray)` → centred log-magnitude (float32)
- `normalize_for_display(spectrum)` → 0–255 uint8
- `analyze_image(rgb)` / `analyze_image_file(path)` → full pipeline (raises `FrequencyError` on bad input)
- `make_spectrum_figure(rgb, analysis)` → 3-panel matplotlib figure for the app

### Tests
```
.venv\Scripts\python.exe -m pytest test\test_frequency.py -v
```
Covers valid inputs (shapes, dtypes, DC-centering, constant-image behaviour) and invalid inputs (`None`, wrong type, wrong shape, empty array, missing file, corrupt `assets/corrupt.png`).

---

## 🏗️ Model Architecture

### Xception (Image Model)

**Base Architecture:**
- Depthwise separable convolutions
- Inception modules
- Residual connections
- Batch normalization

**Customization:**
- Pre-trained on ImageNet
- Fine-tuned for deepfake detection
- Custom classification head (2 classes)

### ResNet18 (Video Model)

**Architecture:**
```
Input (3, 224, 224)
├── Conv2d(3, 64, 7x7, stride=2)
├── BatchNorm + ReLU
├── MaxPool2d
├── ResBlock1 (2x layers)
├── ResBlock2 (2x layers)
├── ResBlock3 (2x layers)
├── ResBlock4 (2x layers)
├── GlobalAvgPool2d
└── Linear(512, 2) → Output
```

**Key Components:**
- Residual blocks with skip connections
- Batch normalization
- ReLU activation
- Adaptive average pooling
- Fully connected layer for classification

---

## 📦 Dependencies

### Core Dependencies

```
streamlit==1.31.0      # Web framework
tensorflow==2.15.0     # Image model framework
torch==2.6.0           # Video model framework
torchvision==0.21.0    # Video processing
opencv-python==4.8.0   # Image/video processing
numpy==1.26.0          # Numerical operations
Pillow==11.0.0         # Image manipulation
```

### Optional Dependencies

- `cuda` and `cudnn`: GPU acceleration for PyTorch
- `matplotlib`: For visualization in notebooks
- `pandas`: Data manipulation in notebooks
- `scikit-learn`: Metrics calculation in notebooks

---

## 🔧 Troubleshooting

### Common Issues

#### Model Not Found Error
**Problem**: `FileNotFoundError` for model files
**Solution**: 
- Ensure `models/image.h5` and `models/video.pth` exist
- Check file paths in `app.py`
- Verify file permissions

#### Memory Issues
**Problem**: Out of memory during video processing
**Solution**:
- Close other applications
- Use shorter videos for testing
- Process videos in smaller batches

#### Slow Processing
**Problem**: Video analysis takes too long
**Solution**:
- Use GPU if available (automatic detection)
- Shorter video segments
- Lower resolution videos

#### OpenCV Errors
**Problem**: `cv2.resize` assertion errors
**Solution**:
- App handles this automatically
- Ensure valid image dimensions
- Check for corrupted files

### Performance Tips

1. **GPU Acceleration**: Automatic CUDA detection and use
2. **Model Caching**: Models cached using Streamlit's `@st.cache_resource`
3. **Batch Processing**: Efficient tensor operations
4. **Progress Updates**: Real-time progress indicators for long operations

---

## 📝 Notes

- **Training**: Use the notebooks on Kaggle or local GPU for training
- **Inference**: Streamlit app optimized for fast inference
- **Models**: Pre-trained models included for immediate use
- **Extensions**: Add more model architectures in notebooks
- **Customization**: Easy to modify preprocessing and architectures

---

## 🎓 Training Workflow

### For Images:
1. Run `deepfake_image.ipynb`
2. Load FaceForensics++ dataset
3. Train Xception model (2 phases)
4. Evaluate on test set
5. Save model as `image.h5`

### For Videos:
1. Run `deepfake_video.ipynb`
2. Extract frames from videos
3. Train ResNet18 model
4. Evaluate on test set
5. Save model as `video.pth`

---

## 📊 Results and Metrics

### Model Performance

The trained models achieve high accuracy on the FaceForensics++ dataset:
- **Image Model**: Optimized with transfer learning
- **Video Model**: Frame-level accuracy with temporal aggregation
- Both models use state-of-the-art architectures and training techniques

### Detection Capabilities

- ✅ Basic deepfakes (face swaps)
- ✅ Face2Face manipulations
- ✅ FaceShifter techniques
- ✅ Neural texture replacements
- ⚠️ Adversarial robustness may vary

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional model architectures
- Improved data augmentation
- Real-time webcam detection
- Batch processing capabilities
- Model ensemble techniques

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- **FaceForensics++**: Dataset for deepfake detection research
- **PyTorch Team**: Deep learning framework
- **TensorFlow Team**: Deep learning framework
- **Streamlit**: Web application framework
- **OpenCV Community**: Computer vision library

---

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the notebooks for implementation details
3. Test with the provided test scripts
4. Ensure all dependencies are correctly installed

---

**Made with ❤️ for deepfake detection research and education**

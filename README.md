# AudioSpecNet: Environmental Sound Classification with ResNet-CNN

![preview](./AudioSpecNet.png)

## Overview

AudioSpecNet implements a comprehensive deep learning pipeline for environmental sound classification using Convolutional Neural Networks (CNNs). The model classifies 50 different environmental sounds from the ESC-50 dataset, including sounds like dog barking, bird chirping, rain, car engines, and more. Built with PyTorch and deployed on Modal's serverless GPU infrastructure, this project demonstrates modern machine learning practices from data preprocessing to production deployment.

## 🎯 Project Goals

- **Audio Classification**: Train a robust CNN to classify 50 environmental sound categories
- **Modern Architecture**: Implement ResNet-style residual blocks for deep audio understanding
- **Production Ready**: Deploy the model as a scalable FastAPI endpoint on serverless GPU infrastructure
- **Scientific Rigor**: Implement state-of-the-art training techniques and comprehensive evaluation

## 🧠 Machine Learning Model Architecture

### Core Architecture: ResNet-Inspired Audio CNN

The model follows a ResNet-inspired architecture specifically designed for audio spectrograms:

#### **Input Processing**

- **Input Shape**: `[batch_size, 1, 128, time_steps]` (Mel spectrogram)
- **Audio Format**: 22.05 kHz sample rate, mono channel
- **Spectrogram Parameters**:
  - FFT Size: 1024 points
  - Hop Length: 512 samples
  - Mel Filters: 128 bands
  - Frequency Range: 0-11,025 Hz

#### **Network Architecture Breakdown**

1. **Initial Convolutional Block (`conv1`)**

   ```
   Conv2d(1 → 64 channels, kernel=7×7, stride=2, padding=3)
   BatchNorm2d(64)
   ReLU activation
   MaxPool2d(kernel=3×3, stride=2, padding=1)
   ```

   - **Purpose**: Extract low-level audio features from mel spectrograms
   - **Output Shape**: `[batch, 64, H/4, W/4]`

2. **Residual Layer 1 (`layer1`)**

   - **Blocks**: 3 ResidualBlocks
   - **Channels**: 64 → 64 (no downsampling)
   - **Purpose**: Learn basic audio patterns and textures
   - **Output Shape**: `[batch, 64, H/4, W/4]`

3. **Residual Layer 2 (`layer2`)**

   - **Blocks**: 4 ResidualBlocks
   - **Channels**: 64 → 128 (first block downsamples)
   - **Stride**: 2 for first block, 1 for others
   - **Purpose**: Capture mid-level frequency patterns
   - **Output Shape**: `[batch, 128, H/8, W/8]`

4. **Residual Layer 3 (`layer3`)**

   - **Blocks**: 6 ResidualBlocks
   - **Channels**: 128 → 256 (first block downsamples)
   - **Stride**: 2 for first block, 1 for others
   - **Purpose**: Learn complex temporal-spectral relationships
   - **Output Shape**: `[batch, 256, H/16, W/16]`

5. **Residual Layer 4 (`layer4`)**

   - **Blocks**: 3 ResidualBlocks
   - **Channels**: 256 → 512 (first block downsamples)
   - **Stride**: 2 for first block, 1 for others
   - **Purpose**: Extract high-level semantic features
   - **Output Shape**: `[batch, 512, H/32, W/32]`

6. **Classification Head**
   ```
   AdaptiveAvgPool2d(1×1)  # Global average pooling
   Dropout(p=0.5)          # Regularization
   Linear(512 → 50)        # Final classification
   ```

### **Residual Block Details**

Each ResidualBlock implements the identity mapping concept:

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        # Two 3×3 convolutions with batch normalization
        self.conv1 = Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = BatchNorm2d(out_channels)

        # Shortcut connection for dimension matching
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, 1, stride)
```

**Forward Pass**: `output = ReLU(conv_block(x) + shortcut(x))`

### **Total Model Parameters**

- **Estimated Parameters**: ~11.2 Million
- **Model Size**: ~43 MB (FP32)
- **Computational Complexity**: ~2.8 GFLOPs per inference

## 📊 Dataset: ESC-50 Environmental Sound Classification

### **Dataset Specifications**

- **Total Samples**: 2,000 audio recordings
- **Classes**: 50 environmental sound categories
- **Duration**: 5 seconds per clip
- **Sample Rate**: 44.1 kHz (resampled to 22.05 kHz)
- **Format**: WAV files
- **Train/Test Split**: Fold-based (Folds 1-4 for training, Fold 5 for testing)

### **Sound Categories**

The ESC-50 dataset covers 5 major groups:

- **Animals**: Dog bark, cat meow, bird chirping, cow moo, frog croak, etc.
- **Natural**: Rain, wind, thunderstorm, waves, fire crackling, etc.
- **Human Non-speech**: Crying baby, sneezing, clapping, footsteps, etc.
- **Interior/Domestic**: Door knock, clock alarm, vacuum cleaner, etc.
- **Exterior/Urban**: Car horn, siren, helicopter, train, construction, etc.

### **Data Preprocessing Pipeline**

1. **Audio Loading & Normalization**

   ```python
   waveform, sample_rate = torchaudio.load(audio_path)
   # Convert stereo to mono
   if waveform.shape[0] > 1:
       waveform = torch.mean(waveform, dim=0, keepdim=True)
   ```

2. **Mel Spectrogram Conversion**

   ```python
   T.MelSpectrogram(
       sample_rate=22050,    # Nyquist frequency: 11.025 kHz
       n_fft=1024,          # FFT window size: ~46ms
       hop_length=512,      # Step size: ~23ms
       n_mels=128,          # Mel filter banks
       f_min=0,             # Minimum frequency
       f_max=11025          # Maximum frequency (Nyquist)
   )
   ```

3. **Amplitude to Decibel Conversion**
   ```python
   T.AmplitudeToDB()  # Convert to log scale: 20*log10(amplitude)
   ```

## 🔧 Advanced Training Techniques

### **Data Augmentation Strategy**

1. **Frequency Masking**

   - **Parameters**: `freq_mask_param=30`
   - **Effect**: Randomly masks 0-30 mel frequency bins
   - **Purpose**: Improve frequency invariance

2. **Time Masking**

   - **Parameters**: `time_mask_param=80`
   - **Effect**: Randomly masks 0-80 time frames
   - **Purpose**: Improve temporal invariance

3. **Mixup Data Augmentation**
   ```python
   # Linear interpolation between samples
   lambda = Beta(0.2, 0.2)  # Mixup ratio
   mixed_x = λ * x_i + (1-λ) * x_j
   mixed_y = λ * y_i + (1-λ) * y_j
   ```
   - **Application Probability**: 30% of batches
   - **Benefits**: Improved generalization, smoother decision boundaries

### **Optimization Configuration**

#### **Optimizer: AdamW**

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0005,           # Initial learning rate
    weight_decay=0.01    # L2 regularization
)
```

#### **Learning Rate Scheduling: OneCycleLR**

```python
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.002,                    # Peak learning rate
    epochs=100,                      # Total epochs
    steps_per_epoch=len(dataloader), # Steps per epoch
    pct_start=0.1                    # 10% warmup phase
)
```

**Learning Rate Schedule**:

- **Phase 1 (0-10%)**: Warmup from 0.0005 → 0.002
- **Phase 2 (10-90%)**: Cosine annealing from 0.002 → 0.0001
- **Phase 3 (90-100%)**: Final decay to near zero

#### **Loss Function & Regularization**

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

- **Label Smoothing**: ε = 0.1 (reduces overconfidence)
- **Dropout**: p = 0.5 (applied before final linear layer)
- **Batch Normalization**: Applied after each convolution

### **Training Configuration**

- **Batch Size**: 32 samples
- **Epochs**: 100
- **GPU**: NVIDIA A10G (24GB VRAM)
- **Training Time**: ~3 hours on Modal
- **Early Stopping**: Best validation accuracy checkpoint saving

## 📈 Model Performance & Metrics

### **Training Results**

- **Final Validation Accuracy**: 84.50%
- **Training Loss**: Converged to ~0.15
- **Validation Loss**: Stabilized at ~0.45
- **Convergence**: Achieved around epoch 80-90

### **Performance Analysis**

- **Parameter Efficiency**: ~11.2M parameters for 50-class classification
- **Inference Speed**: ~50ms per sample on A10G GPU
- **Memory Usage**: ~2GB GPU memory during inference
- **Model Robustness**: Handles various audio qualities and lengths

## 🚀 Production Deployment

### **Modal Serverless Architecture**

#### **Container Specification**

```python
image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["libsndfile1"])  # Audio processing
         .add_local_python_source("model"))
```

#### **GPU Configuration**

- **GPU Type**: NVIDIA A10G
- **Memory**: 24GB VRAM
- **Scale-down Window**: 15 seconds (cost optimization)
- **Auto-scaling**: Handles traffic spikes automatically

### **API Endpoint Specifications**

#### **FastAPI Integration**

```python
@modal.fastapi_endpoint(method="POST")
def inference(self, request: InferenceRequest):
    # Audio processing and prediction
```

#### **Request Format**

```json
{
  "audio_data": "base64_encoded_audio_bytes"
}
```

#### **Response Format**

```json
{
    "predictions": [
        {"class": "dog", "confidence": 0.95},
        {"class": "cat", "confidence": 0.03},
        {"class": "bird", "confidence": 0.02}
    ],
    "visualization": {
        "layer1": {"shape": [64, 32, 87], "values": [[...]]},
        "layer2": {"shape": [128, 16, 44], "values": [[...]]}
    },
    "input_spectrogram": {
        "shape": [128, 173],
        "values": [[...]]
    },
    "waveform": {
        "values": [...],
        "sample_rate": 44100,
        "duration": 5.0
    }
}
```

### **Audio Processing Pipeline**

1. **Base64 Decoding**: Convert uploaded audio to bytes
2. **Format Handling**: Support WAV, MP3, FLAC formats
3. **Resampling**: Standardize to 44.1 kHz
4. **Mono Conversion**: Average stereo channels
5. **Spectrogram Generation**: Convert to mel spectrogram
6. **Normalization**: Apply same preprocessing as training

## 🔬 Model Interpretability & Visualization

### **Feature Map Extraction**

The model provides internal layer visualizations for understanding what the network learns:

- **Layer 1**: Low-level texture and edge detection
- **Layer 2**: Basic frequency patterns and harmonics
- **Layer 3**: Complex temporal-spectral patterns
- **Layer 4**: High-level semantic features

### **Visualization Processing**

```python
# Aggregate across channels for visualization
aggregated_tensor = torch.mean(tensor, dim=1)  # [batch, height, width]
numpy_array = aggregated_tensor.squeeze(0).cpu().numpy()
```

## 📋 Dependencies & Requirements

### **Core Dependencies**

```
torch>=2.0.0          # Deep learning framework
torchaudio>=2.0.0     # Audio processing
pandas>=1.5.0         # Data manipulation
numpy>=1.24.0         # Numerical computing
librosa>=0.10.0       # Advanced audio analysis
soundfile>=0.12.0     # Audio I/O
```

### **Training Dependencies**

```
tensorboard>=2.12.0   # Training visualization
tqdm>=4.65.0          # Progress bars
```

### **Deployment Dependencies**

```
fastapi>=0.100.0      # API framework
pydantic>=2.0.0       # Data validation
modal>=0.50.0         # Serverless deployment
```

### **System Requirements**

- **Python**: 3.8+ (recommended: 3.10+)
- **CUDA**: 11.8+ for GPU training
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for dataset and model artifacts

## 🛠️ Setup & Installation

### Prerequisites

1. **Python Environment**

   - Python 3.8+ (recommended: Python 3.10+)
   - pip package manager
   - Virtual environment tool

2. **System Dependencies**
   - Git for version control
   - CUDA-compatible GPU (for training)
   - Internet connection for dataset download

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/ItisSubham/AudioSpecNet-CNN-Modal
cd AudioSpecNet-CNN-Modal
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Modal Setup (for Cloud Training/Deployment)

```bash
# Install Modal CLI
pip install modal

# Authenticate with Modal
modal setup
```

### 🏋️ Training the Model

#### Local Training (Not Recommended)

```bash
python train.py
```

_Note: Local training requires significant computational resources and time_

#### Cloud Training with Modal (Recommended)

```bash
# Run training on Modal's GPU infrastructure
modal run train.py

# Monitor training progress through Modal dashboard
```

#### Training Configuration

- **Duration**: ~3 hours on A10G GPU
- **Dataset**: Automatically downloads ESC-50 (2,000 samples)
- **Output**: Best model saved to Modal volume
- **Monitoring**: TensorBoard logs available

### 🚀 Model Deployment

#### Deploy as Serverless API

```bash
# Deploy the inference endpoint
modal deploy main.py
```

#### Test Local Inference

```bash
# Test with sample audio file
modal run main.py
```

### 📊 Monitoring & Evaluation

#### TensorBoard Visualization

```bash
# View training metrics (if training locally)
tensorboard --logdir ./tensorboard_logs
```

#### Modal Dashboard

- Monitor training progress
- View GPU utilization
- Check deployment status
- Monitor API usage

### 🔧 Configuration Options

#### Training Hyperparameters

Edit `train.py` to modify:

- Learning rate schedule
- Batch size
- Number of epochs
- Data augmentation parameters

#### Model Architecture

Edit `model.py` to modify:

- Number of residual blocks
- Channel dimensions
- Dropout rates

#### Deployment Settings

Edit `main.py` to modify:

- GPU type
- Scaling parameters
- API response format

### 📁 Project Structure

```
audiospecnet/
├── model.py           # CNN architecture definition
├── train.py           # Training script with Modal
├── main.py            # Inference API with Modal
├── requirements.txt   # Python dependencies
├── README.md          # This documentation
└── .gitignore        # Git ignore patterns
```

### 🧪 Testing & Validation

#### Unit Tests

```bash
# Test model architecture
python -c "from model import AudioCNN; model = AudioCNN(); print('Model loaded successfully')"

# Test audio processing
python -c "from main import AudioProcessor; proc = AudioProcessor(); print('Audio processor ready')"
```

#### Integration Tests

```bash
# Test full inference pipeline
modal run main.py
```

### 📚 Additional Resources

- **ESC-50 Dataset**: [Original Paper](https://github.com/karolpiczak/ESC-50)
- **Modal Documentation**: [Modal Docs](https://modal.com/docs)
- **PyTorch Audio**: [TorchAudio Docs](https://pytorch.org/audio/)
- **ResNet Architecture**: [Original Paper](https://arxiv.org/abs/1512.03385)

### 🐛 Troubleshooting

#### Common Issues

1. **CUDA Out of Memory**

   - Reduce batch size in `train.py`
   - Use gradient accumulation

2. **Audio Loading Errors**

   - Install system audio libraries: `apt-get install libsndfile1`
   - Check audio file formats (WAV, MP3, FLAC supported)

3. **Modal Authentication**

   - Run `modal setup` and follow prompts
   - Check Modal dashboard for account status

4. **Training Convergence Issues**
   - Adjust learning rate schedule
   - Increase training epochs
   - Check data augmentation parameters

#### Performance Optimization

1. **Faster Training**

   - Use larger batch sizes if GPU memory allows
   - Enable mixed precision training
   - Use more powerful GPU types on Modal

2. **Better Accuracy**
   - Increase model depth (more residual blocks)
   - Experiment with different data augmentation
   - Ensemble multiple models

### 💡 Tips for Success

1. **Start Small**: Begin with fewer epochs to verify pipeline
2. **Monitor Carefully**: Watch TensorBoard metrics during training
3. **Validate Early**: Test inference pipeline before full training
4. **Save Checkpoints**: Enable model checkpointing for long training runs
5. **Document Changes**: Track hyperparameter experiments

This comprehensive setup ensures you can successfully train, deploy, and monitor your audio classification CNN from start to finish.

# Action Detection for Sign Language

This project uses a video-based machine learning model to detect and recognize sign language gestures.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Network Access Options:**
   - **Local Network**: Access the API at `http://localhost:8000` or `http://your-local-ip:8000`
   - **Public Access**: Use network tunneling services (like ngrok, frp, or cloudflare tunnel) to expose your local server to the internet, or deploy to a public server

## Model Information

This project uses the following pre-trained model from Hugging Face:
- **Model:** [VideoMAE_Base_WLASL_100_200_epochs_p20_SR_8](https://huggingface.co/Shawon16/VideoMAE_Base_WLASL_100_200_epochs_p20_SR_8)
- **Vocabulary:** 100 sign language words
- **Framework:** VideoMAE (Video Masked Autoencoders)

The model will be automatically downloaded and cached locally in the `my_model_weights/` directory on first run.

## API Endpoints

- **GET /ping**: Health check endpoint
- **POST /translate**: Upload a video file to get sign language translation

## Hardware Requirements

- GPU support recommended (CUDA-compatible)
- The application will automatically detect and use available GPU or fallback to CPU

## Usage

The application provides a FastAPI server that accepts video files and returns the detected sign language word with confidence score.
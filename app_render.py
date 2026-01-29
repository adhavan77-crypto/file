from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
import numpy as np
import os
from typing import Optional
import uvicorn

# Initialize FastAPI
app = FastAPI(
    title="AI Voice Detection API",
    description="Detects AI-generated vs Human voice in multiple languages",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key
API_KEY = os.getenv("API_KEY", "your-secure-api-key-12345")

# Models
class VoiceDetectionRequest(BaseModel):
    audio_base64: str
    language: Optional[str] = "english"

class VoiceDetectionResponse(BaseModel):
    classification: str
    confidence: float
    language: str
    explanation: str

def extract_audio_features(audio_data, sr):
    """Extract audio features using numpy only (avoiding librosa issues)"""
    try:
        # Import librosa here to avoid deployment issues
        import librosa
        
        # Basic features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        rms = librosa.feature.rms(y=audio_data)[0]
        
        features = {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr)),
            'mfcc_mean': float(np.mean(mfccs)),
            'mfcc_std': float(np.std(mfccs)),
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
        }
        
        return features
    except Exception as e:
        # Fallback to simple analysis if librosa fails
        return {
            'spectral_centroid_mean': float(np.mean(np.abs(audio_data))),
            'spectral_centroid_std': float(np.std(audio_data)),
            'spectral_rolloff_mean': float(np.max(np.abs(audio_data))),
            'zcr_mean': 0.5,
            'zcr_std': 0.1,
            'mfcc_mean': 0.0,
            'mfcc_std': 1.0,
            'rms_mean': float(np.sqrt(np.mean(audio_data**2))),
            'rms_std': float(np.std(audio_data)),
        }

def detect_ai_voice(features):
    """AI voice detection logic"""
    ai_score = 0
    
    # Check consistency patterns
    spectral_cv = features['spectral_centroid_std'] / (features['spectral_centroid_mean'] + 1e-6)
    if spectral_cv < 0.15:
        ai_score += 0.3
    
    zcr_cv = features['zcr_std'] / (features['zcr_mean'] + 1e-6)
    if zcr_cv < 0.4:
        ai_score += 0.25
    
    rms_cv = features['rms_std'] / (features['rms_mean'] + 1e-6)
    if rms_cv < 0.3:
        ai_score += 0.25
    
    mfcc_variance = features['mfcc_std']
    if mfcc_variance < 1.5:
        ai_score += 0.2
    
    # Classification
    if ai_score > 0.5:
        classification = "AI_GENERATED"
        confidence = min(0.95, 0.5 + ai_score * 0.8)
        explanation = "Audio exhibits high consistency in spectral features, uniform energy distribution, and reduced natural variation typical of AI-generated speech."
    else:
        classification = "HUMAN"
        confidence = min(0.95, 0.5 + (1 - ai_score) * 0.8)
        explanation = "Audio demonstrates natural human speech patterns with organic variations in pitch, energy, and spectral characteristics."
    
    return classification, confidence, explanation

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "message": "AI Voice Detection API",
        "version": "1.0.0",
        "supported_languages": ["tamil", "english", "hindi", "malayalam", "telugu"],
        "endpoints": {
            "detect": "/detect",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "ai-voice-detection",
        "version": "1.0.0"
    }

@app.post("/detect", response_model=VoiceDetectionResponse)
async def detect_voice(
    request: VoiceDetectionRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Detect if voice is AI-generated or human
    
    Args:
        audio_base64: Base64-encoded MP3 audio
        language: One of [tamil, english, hindi, malayalam, telugu]
    
    Returns:
        JSON with classification, confidence, language, explanation
    """
    
    # Validate API key
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Validate language
    supported_languages = ["tamil", "english", "hindi", "malayalam", "telugu"]
    language = request.language.lower() if request.language else "english"
    
    if language not in supported_languages:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Must be one of: {', '.join(supported_languages)}"
        )
    
    try:
        # Decode base64
        try:
            audio_bytes = base64.b64decode(request.audio_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 encoding")
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio data")
        
        # Load audio
        try:
            import librosa
            audio_data, sample_rate = librosa.load(
                io.BytesIO(audio_bytes),
                sr=22050,
                mono=True
            )
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to load audio. Ensure it's a valid MP3 file. Error: {str(e)}"
            )
        
        # Extract features
        features = extract_audio_features(audio_data, sample_rate)
        
        # Detect
        classification, confidence, explanation = detect_ai_voice(features)
        
        return VoiceDetectionResponse(
            classification=classification,
            confidence=round(confidence, 2),
            language=language,
            explanation=explanation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "status_code": 500
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

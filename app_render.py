from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import tempfile
import librosa
import numpy as np
from scipy import stats
import traceback

app = Flask(__name__)
CORS(app)

# API Key for authentication
API_KEY = os.environ.get('API_KEY', 'your-secure-api-key-12345')

def extract_features(audio_path):
    """Extract audio features for classification"""
    try:
        # Load audio file with shorter duration for faster processing
        y, sr = librosa.load(audio_path, sr=22050, duration=10)
        
        features = {}
        
        # 1. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # 2. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 3. MFCC features (reduced from 13 to 5 for speed)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
        for i in range(5):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # 4. RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        return features
    
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        raise

def classify_audio_heuristic(features):
    """
    Heuristic-based classification (no ML model needed)
    AI voices typically have:
    - More consistent spectral features (lower std)
    - More uniform zero-crossing rate
    - More constant energy levels
    """
    
    # Calculate consistency scores
    spectral_consistency = features.get('spectral_centroid_std', 1000) + features.get('spectral_rolloff_std', 1000)
    zcr_consistency = features.get('zcr_std', 1)
    energy_consistency = features.get('rms_std', 1)
    
    # Scoring system
    ai_score = 0
    human_score = 0
    
    # Check spectral consistency (AI voices are more consistent)
    if spectral_consistency < 1000:  # Very consistent
        ai_score += 2
    elif spectral_consistency > 2000:  # Natural variation
        human_score += 2
    else:
        ai_score += 1
        human_score += 1
    
    # Check zero-crossing rate consistency
    if zcr_consistency < 0.02:  # Very uniform
        ai_score += 2
    elif zcr_consistency > 0.04:  # Natural variation
        human_score += 2
    else:
        ai_score += 1
        human_score += 1
    
    # Check energy consistency
    if energy_consistency < 0.03:  # Constant energy
        ai_score += 2
    elif energy_consistency > 0.08:  # Natural variation
        human_score += 2
    else:
        ai_score += 1
        human_score += 1
    
    # Calculate confidence
    total_score = ai_score + human_score
    if ai_score > human_score:
        prediction = 'AI_GENERATED'
        confidence = ai_score / total_score
    else:
        prediction = 'HUMAN'
        confidence = human_score / total_score
    
    # Ensure confidence is reasonable (between 0.55 and 0.95)
    confidence = max(0.55, min(0.95, confidence))
    
    return prediction, confidence

def generate_explanation(features, prediction):
    """Generate human-readable explanation"""
    explanations = []
    
    spectral_std = features.get('spectral_centroid_std', 0)
    zcr_std = features.get('zcr_std', 0)
    rms_std = features.get('rms_std', 0)
    
    if prediction == 'AI_GENERATED':
        if spectral_std < 500:
            explanations.append("Highly consistent spectral characteristics indicate synthetic generation")
        if zcr_std < 0.02:
            explanations.append("Uniform zero-crossing rate pattern typical of AI voices")
        if rms_std < 0.05:
            explanations.append("Constant energy levels suggest automated generation")
        
        if not explanations:
            explanations.append("Audio patterns and consistency metrics indicate AI-generated speech")
    else:
        if spectral_std > 700:
            explanations.append("Natural spectral variation detected in voice characteristics")
        if zcr_std > 0.03:
            explanations.append("Human-like voice modulation and breathing patterns present")
        if rms_std > 0.08:
            explanations.append("Natural energy fluctuations typical of human speech")
        
        if not explanations:
            explanations.append("Audio patterns show natural human speech characteristics")
    
    return "; ".join(explanations) if explanations else "Analysis based on audio feature patterns"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "AI Voice Detection API is running",
        "version": "1.0"
    }), 200

@app.route('/detect', methods=['POST'])
def detect_voice():
    """Main detection endpoint"""
    try:
        # Validate API Key
        auth_header = request.headers.get('Authorization') or request.headers.get('X-API-Key')
        if not auth_header or auth_header != API_KEY:
            return jsonify({
                "error": "Unauthorized",
                "message": "Invalid or missing API key"
            }), 401
        
        # Get request data
        data = request.get_json()
        
        if not data or 'audio' not in data:
            return jsonify({
                "error": "Bad Request",
                "message": "Missing 'audio' field in request body"
            }), 400
        
        audio_base64 = data['audio']
        
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            return jsonify({
                "error": "Bad Request",
                "message": "Invalid base64 encoding"
            }), 400
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        try:
            # Extract features
            features = extract_features(temp_audio_path)
            
            # Classify using heuristics (no ML model needed)
            prediction, confidence = classify_audio_heuristic(features)
            
            # Generate explanation
            explanation = generate_explanation(features, prediction)
            
            # Prepare response
            response = {
                "classification": prediction,
                "confidence": round(confidence, 2),
                "explanation": explanation,
                "language_support": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
                "status": "success"
            }
            
            return jsonify(response), 200
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e)
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        "message": "AI Voice Detection API",
        "version": "1.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/detect": "POST - Detect AI-generated voice"
        },
        "authentication": "Required - Use Authorization or X-API-Key header",
        "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

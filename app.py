from flask import Flask, request, jsonify, render_template
import cv2
import base64
import numpy as np
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("DeepFace not available. Facial emotion detection will be disabled.")
import text2emotion as te
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

app = Flask(__name__)

# Spotify API credentials (set these as environment variables)
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

# Initialize Spotify client
if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET))
else:
    sp = None
    print("Spotify API credentials not set. Playlist functionality will be disabled.")

# Emotion to mood mapping
emotion_to_mood = {
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'energetic',
    'fear': 'calm',
    'surprise': 'excited',
    'neutral': 'chill',
    'disgust': 'dark',
    'joy': 'happy',
    'anticipation': 'excited',
    'trust': 'calm',
    'positive': 'happy',
    'negative': 'sad',
    'worry': 'calm',
    'love': 'romantic'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    data = request.json
    emotion = None
    detection_method = None

    # Facial emotion detection
    if 'image' in data and DEEPFACE_AVAILABLE:
        try:
            image_data = base64.b64decode(data['image'].split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                print("Error: Could not decode image")
            else:
                result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list):
                    emotion = result[0]['dominant_emotion']
                else:
                    emotion = result['dominant_emotion']
                detection_method = 'facial'
        except Exception as e:
            print(f"Error in facial emotion detection: {e}")
            return jsonify({'error': f'Facial detection failed. Please ensure your face is clearly visible.'}), 400
    elif 'image' in data and not DEEPFACE_AVAILABLE:
        return jsonify({'error': 'Facial emotion detection is currently unavailable. DeepFace library requires additional setup. Please use Text Emotion Detection instead.'}), 400

    # Text emotion detection
    if 'text' in data and data['text'].strip():
        try:
            text = data['text'].lower().strip()
            
            # Enhanced keyword-based emotion detection
            emotion_keywords = {
                'happy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'love', 'glad', 'cheerful', 'delighted', 'thrilled', 'excellent', 'good', 'awesome', 'perfect'],
                'sad': ['sad', 'unhappy', 'depressed', 'lonely', 'miserable', 'down', 'blue', 'disappointed', 'upset', 'hurt', 'heartbroken', 'crying', 'tears', 'awful', 'terrible', 'bad'],
                'angry': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'rage', 'pissed', 'hate', 'disgusted', 'outraged'],
                'fear': ['scared', 'afraid', 'fear', 'anxious', 'worried', 'nervous', 'terrified', 'frightened', 'panic'],
                'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'wow', 'incredible', 'unbelievable'],
                'neutral': ['okay', 'fine', 'alright', 'normal', 'meh', 'whatever']
            }
            
            # Count keyword matches
            emotion_scores = {}
            for emotion_key, keywords in emotion_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text)
                if score > 0:
                    emotion_scores[emotion_key] = score
            
            # Try text2emotion first
            text_emotions = te.get_emotion(data['text'])
            
            # Combine both methods
            if text_emotions and any(text_emotions.values()):
                # Get emotion from text2emotion
                te_emotion = max(text_emotions, key=text_emotions.get)
                te_score = text_emotions[te_emotion]
                
                # If keyword detection found something, use the one with higher confidence
                if emotion_scores:
                    keyword_emotion = max(emotion_scores, key=emotion_scores.get)
                    keyword_score = emotion_scores[keyword_emotion]
                    
                    # Prefer keyword detection if it has strong matches
                    if keyword_score >= 2 or (keyword_score > 0 and te_score < 0.3):
                        emotion = keyword_emotion
                    else:
                        emotion = te_emotion
                else:
                    emotion = te_emotion
                    
                detection_method = 'text'
            elif emotion_scores:
                # Fallback to keyword detection
                emotion = max(emotion_scores, key=emotion_scores.get)
                detection_method = 'text'
            else:
                return jsonify({'error': 'Could not detect emotion from text. Please provide more descriptive text about your feelings.'}), 400
        except Exception as e:
            print(f"Error in text emotion detection: {e}")
            return jsonify({'error': 'Text analysis failed. Please try again.'}), 400

    if emotion:
        # Normalize emotion to lowercase for mapping
        emotion_lower = emotion.lower()
        mood = emotion_to_mood.get(emotion_lower, 'chill')
        playlists = get_spotify_playlists(mood)
        
        # Debug logging
        print(f"Detected emotion: {emotion} -> Normalized: {emotion_lower} -> Mood: {mood}")
        
        return jsonify({
            'emotion': emotion_lower, 
            'mood': mood, 
            'playlists': playlists,
            'method': detection_method
        })
    else:
        return jsonify({'error': 'Could not detect emotion. Please try again with clearer input.'}), 400

def get_spotify_playlists(mood):
    if sp is None:
        # Return sample playlists when Spotify API is not configured
        sample_playlists = {
            'happy': [
                {'name': 'Happy Hits', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC', 'image': 'https://via.placeholder.com/300x300/FFD700/000000?text=Happy+Hits'},
                {'name': 'Feel Good Indie', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX2sUQwD7tbmL', 'image': 'https://via.placeholder.com/300x300/FF6B9D/000000?text=Feel+Good'},
                {'name': 'Mood Booster', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0', 'image': 'https://via.placeholder.com/300x300/00D4FF/000000?text=Mood+Booster'}
            ],
            'sad': [
                {'name': 'Life Sucks', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1', 'image': 'https://via.placeholder.com/300x300/4169E1/FFFFFF?text=Sad+Songs'},
                {'name': 'Sad Indie', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX59NCqCqJtoH', 'image': 'https://via.placeholder.com/300x300/708090/FFFFFF?text=Sad+Indie'},
                {'name': 'Melancholy', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWX83CujKHHOn', 'image': 'https://via.placeholder.com/300x300/2F4F4F/FFFFFF?text=Melancholy'}
            ],
            'energetic': [
                {'name': 'Beast Mode', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX76Wlfdnj7AP', 'image': 'https://via.placeholder.com/300x300/FF4500/000000?text=Beast+Mode'},
                {'name': 'Power Workout', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX70RN3TfWWJh', 'image': 'https://via.placeholder.com/300x300/DC143C/000000?text=Power+Workout'},
                {'name': 'Adrenaline', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX0pH2SQMRXnC', 'image': 'https://via.placeholder.com/300x300/8B0000/FFFFFF?text=Adrenaline'}
            ],
            'calm': [
                {'name': 'Peaceful Piano', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO', 'image': 'https://via.placeholder.com/300x300/87CEEB/000000?text=Peaceful+Piano'},
                {'name': 'Calm Vibes', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWU0ScTcjJBdj', 'image': 'https://via.placeholder.com/300x300/ADD8E6/000000?text=Calm+Vibes'},
                {'name': 'Relaxing Sounds', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWZd79rJ6a7lp', 'image': 'https://via.placeholder.com/300x300/B0E0E6/000000?text=Relaxing'}
            ],
            'excited': [
                {'name': 'Party Time', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DXaXB8fQg7xif', 'image': 'https://via.placeholder.com/300x300/FF1493/000000?text=Party+Time'},
                {'name': 'Dance Party', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX4dyzvuaRJ0n', 'image': 'https://via.placeholder.com/300x300/FF69B4/000000?text=Dance+Party'},
                {'name': 'Energy Boost', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX3Sp0P28SIer', 'image': 'https://via.placeholder.com/300x300/FFB6C1/000000?text=Energy+Boost'}
            ],
            'chill': [
                {'name': 'Chill Hits', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX4WYpdgoIcn6', 'image': 'https://via.placeholder.com/300x300/9370DB/000000?text=Chill+Hits'},
                {'name': 'Lofi Beats', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWWQRwui0ExPn', 'image': 'https://via.placeholder.com/300x300/BA55D3/000000?text=Lofi+Beats'},
                {'name': 'Chill Vibes', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX889U0CL85jj', 'image': 'https://via.placeholder.com/300x300/DDA0DD/000000?text=Chill+Vibes'}
            ],
            'romantic': [
                {'name': 'Romantic', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX50QitC6Oqtn', 'image': 'https://via.placeholder.com/300x300/FF1493/FFFFFF?text=Romantic'},
                {'name': 'Love Songs', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX0UrRvztWcAU', 'image': 'https://via.placeholder.com/300x300/FF69B4/FFFFFF?text=Love+Songs'},
                {'name': 'Date Night', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX4OzrY981I1W', 'image': 'https://via.placeholder.com/300x300/FFB6C1/000000?text=Date+Night'}
            ],
            'dark': [
                {'name': 'Dark & Stormy', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX0XUfTFmNBRM', 'image': 'https://via.placeholder.com/300x300/2F4F4F/FFFFFF?text=Dark'},
                {'name': 'Metal', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWWOaP4H0w5b0', 'image': 'https://via.placeholder.com/300x300/000000/FFFFFF?text=Metal'},
                {'name': 'Rock Hard', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWXRqgorJj26U', 'image': 'https://via.placeholder.com/300x300/1C1C1C/FFFFFF?text=Rock+Hard'}
            ]
        }
        return sample_playlists.get(mood, sample_playlists['chill'])
    
    try:
        # Enhanced search queries for better results
        search_queries = {
            'happy': ['happy', 'feel good', 'uplifting'],
            'sad': ['sad', 'melancholy', 'emotional'],
            'energetic': ['workout', 'energetic', 'power'],
            'calm': ['calm', 'peaceful', 'relaxing'],
            'excited': ['party', 'dance', 'upbeat'],
            'chill': ['chill', 'lofi', 'relax'],
            'romantic': ['romantic', 'love songs', 'date night'],
            'dark': ['dark', 'intense', 'heavy']
        }
        
        query = search_queries.get(mood, [mood])[0]
        results = sp.search(q=query, type='playlist', limit=5)
        playlists = []
        for item in results['playlists']['items']:
            playlists.append({
                'name': item['name'],
                'url': item['external_urls']['spotify'],
                'image': item['images'][0]['url'] if item['images'] else None
            })
        return playlists
    except Exception as e:
        print(f"Error fetching Spotify playlists: {e}")
        return []

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

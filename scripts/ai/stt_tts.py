import speech_recognition as sr
import requests
import tempfile
import pygame
import os

#----------------- Speech to text -----------------#
# Initialise the recognizer
r = sr.Recognizer()

# Function to record audio and convert it to text
def record_audio():
    # loop indefinitely until the user stops yapping
    while(1):
        # connect user mic and listen to audio
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source)
            
            try:
                # Converting user audio to text using google 
                text = r.recognize_google(audio)
                return text
            except Exception as e:
                print("Error: ", e)
                return None


#----------------- Text to Speech -----------------#
"""
source: https://www.youtube.com/watch?v=3BMy5KPa_kQ 
"""
API_KEY = "sk_2a4636131ac359403267066a77a5142a16aa3f934dc63a87"
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

def play_audio(file_path):
    # Plays audio through pytgame
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.quit()
    except Exception as e:
        print(f"Error playing audio: {str(e)}")

def output_audio(text):
    voice_id = DEFAULT_VOICE_ID

    # API endpoint for text-to-speech conversion
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    # Headers with API key
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY
    }
    
    # Request payload
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_filename = temp_file.name
            temp_file.close()
            
            # Save the audio to the temporary file
            with open(temp_filename, "wb") as f:
                f.write(response.content)
            
            
            # Play the audio
            play_audio(temp_filename)
            
            # Delete the file after playing
            try:
                os.unlink(temp_filename)
            except Exception as e:
                print(f"Error deleting temporary file: {str(e)}")
            
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

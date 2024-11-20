from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
import google.generativeai as genai
import time
import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import WebVTTFormatter
from urllib.parse import urlparse, parse_qs
from werkzeug.utils import secure_filename
import PyPDF2
import io
import sys
import os
import azure.cognitiveservices.speech as speechsdk
import uuid
from flask import send_from_directory
from deep_translator import GoogleTranslator
from typing import List, Optional
from flask_socketio import SocketIO, emit, join_room, leave_room
import requests  # Add this import

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")

# In-memory store for room state
rooms = {}

class Room:
    def __init__(self):
        self.participants = []
        self.resources = []
        self.selected_resource = None

@socketio.on('join')
def on_join(data):
    room_id = data['roomId']
    participant = data['participant']
    
    if room_id not in rooms:
        rooms[room_id] = Room()
    
    join_room(room_id)
    rooms[room_id].participants.append(participant)
    
    # Broadcast updated participant list to all in room
    emit('participants_updated', {
        'participants': rooms[room_id].participants
    }, room=room_id)
    
    # Send current room state to new participant
    emit('room_state', {
        'resources': rooms[room_id].resources,
        'selected_resource': rooms[room_id].selected_resource,
        'participants': rooms[room_id].participants
    }, room=request.sid)

@socketio.on('leave')
def on_leave(data):
    room_id = data['roomId']
    participant = data['participant']
    
    if room_id in rooms:
        rooms[room_id].participants.remove(participant)
        leave_room(room_id)
        
        emit('participants_updated', {
            'participants': rooms[room_id].participants
        }, room=room_id)

@socketio.on('resource_added')
def on_resource_added(data):
    try:
        room_id = data['roomId']
        resource = data['resource']
        sender = data.get('sender', 'Anonymous')
        
        if not room_id or room_id not in rooms:
            return
            
        # Validate resource
        if not all(k in resource for k in ('name', 'url', 'type')):
            return
            
        # Add resource to room
        rooms[room_id].resources.append(resource)
        
        # Broadcast to all participants including sender
        emit('resource_updated', {
            'resources': rooms[room_id].resources,
            'latestResource': resource,
            'sender': sender
        }, room=room_id)
        
        # Send system message about new resource
        emit('new_message', {
            'sender': 'System',
            'content': f'{sender} added a new resource: {resource["name"]}'
        }, room=room_id)
        
    except Exception as e:
        print(f"Error adding resource: {str(e)}")
        emit('error', {'message': 'Failed to add resource'}, room=request.sid)

@socketio.on('resource_selected')
def on_resource_selected(data):
    try:
        room_id = data['roomId']
        resource = data['resource']
        sender = data.get('sender', 'Anonymous')
        
        if not room_id or room_id not in rooms:
            return
            
        # Update selected resource
        rooms[room_id].selected_resource = resource
        
        # Broadcast to all participants
        emit('preview_updated', {
            'selected_resource': resource,
            'sender': sender
        }, room=room_id)
        
        # Notify about preview change
        emit('new_message', {
            'sender': 'System',
            'content': f'{sender} is viewing: {resource["name"]}'
        }, room=room_id)
        
    except Exception as e:
        print(f"Error selecting resource: {str(e)}")
        emit('error', {'message': 'Failed to update preview'}, room=request.sid)

@socketio.on('message')
def handle_message(data):
    room_id = data['roomId']
    message = data['message']
    sender = data['sender']
    
    if room_id in rooms:
        emit('new_message', {
            'sender': sender,
            'content': message
        }, room=room_id)
        
        if message.startswith('@Ai'):
            question = message[len('@Ai'):].strip()
            youtube_urls = [res['url'] for res in rooms[room_id].resources if res['type'] == 'youtube']
            pdf_urls = [res['url'] for res in rooms[room_id].resources if res['type'] == 'pdf']
            
            # Download PDFs from URLs
            pdf_paths = download_pdfs(pdf_urls)
            
            # Get context from resources
            context = get_context(youtube_urls, pdf_paths)
            
            # Prepare context part
            if context:
                context_part = f"Context:\n{context}\n"
            else:
                context_part = ""
            
            # Build the prompt
            prompt = f"""
{context_part}
User question: {question}

Provide a detailed answer.
Format your response using markdown with:
- Headers for main points
- Code blocks where appropriate
- Lists and tables when needed
- Bold and italic for emphasis
"""
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=generation_config
            )
            try:
                response = model.generate_content(prompt)
                if hasattr(response, 'text'):
                    ai_response = response.text
                    emit('new_message', {
                        'sender': 'AI',
                        'content': ai_response
                    }, room=room_id)
            except Exception as e:
                print(f"Error generating AI response: {str(e)}")
                emit('new_message', {
                    'sender': 'AI',
                    'content': 'Sorry, I encountered an error processing your request.'
                }, room=room_id)

# Configure Gemini AI
genai.configure(api_key="AIzaSyDQJSl-a-2lvX_mk1N5dJ6dwg1CkVZnKF0")

# AI model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

def wait_for_files_active(files):
    """Waits for the given files to be active."""
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            time.sleep(2)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    return True

def upload_files(filenames):
    """Upload multiple files and wait for processing"""
    files = []
    try:
        for filename in filenames:
            file = genai.upload_file(
                os.path.join('functionalites', filename), 
                mime_type="application/pdf"
            )
            files.append(file)
        wait_for_files_active(files)
        return files
    except Exception as e:
        raise Exception(f"File upload failed: {str(e)}")
    
topic_files = {
        'c': ['C programming.pdf', 'c imp.pdf'],
        'bee': ['bee.pdf', 'bee imp.pdf'],
        'physics': ['FULL BOOK PHYSICS.pdf', 'Engineering Physics Q Bank.pdf'],
        'math': ['textbook_og_engineering_matematics.pdf', 'maths imp.pdf'],
        'drawing': ['Engineering Drawing.pdf']
    }

def get_video_id(url):
    """
    Extract video ID from various YouTube URL formats
    """
    try:
        parsed_url = urlparse(url)
        if (parsed_url.hostname == 'youtu.be'):
            return parsed_url.path[1:].split('?')[0]
        if 'youtube.com' in parsed_url.hostname:
            if 'v' in parse_qs(parsed_url.query):
                return parse_qs(parsed_url.query)['v'][0]
        return None
    except:
        return None

def process_youtube_transcripts(youtube_urls: List[str]) -> List[str]:
    transcripts = []
    for url in youtube_urls:
        try:
            video_id = get_video_id(url)
            if not video_id:
                continue
                
            # First try to get English transcript
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            except:
                # If English not available, try Hindi and translate
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi'])
                    # Combine text for efficient translation
                    hindi_text = ' '.join([entry['text'] for entry in transcript])
                    # Translate to English
                    translator = GoogleTranslator(source='hi', target='en')
                    english_text = translator.translate(hindi_text)
                    # Create a single transcript entry
                    transcript = [{
                        'text': english_text,
                        'start': 0,
                        'duration': sum(entry['duration'] for entry in transcript)
                    }]
                except Exception as e:
                    print(f"Error getting Hindi transcript: {str(e)}")
                    continue
            
            formatter = WebVTTFormatter()
            webvtt_formatted = formatter.format_transcript(transcript)
            transcripts.append(webvtt_formatted)
        except Exception as e:
            print(f"Error processing YouTube URL {url}: {str(e)}")
            continue
    return transcripts

def get_context(youtube_urls, pdf_paths):
    """Get combined context from YouTube videos and PDFs"""
    contexts = []
    
    # Process YouTube transcripts
    if youtube_urls:
        transcripts = process_youtube_transcripts(youtube_urls)
        contexts.extend(transcripts)
    
    # Process PDFs
    if pdf_paths:
        for path in pdf_paths:
            text = extract_text_from_pdf(path)
            if text:
                contexts.append(text)
    
    return "\n\n".join(contexts)

def generate_conversation_for_audio(context):
    # Initialize model for conversation generation
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )
    
    prompt = f"""
    Create a natural conversational dialogue between two experts discussing this content:
    {context}
    Make it engaging and educational. Return only the conversation text.
    """
    
    response = model.generate_content(prompt)
    return response.text if hasattr(response, 'text') else None

def process_audio_generation(context):
    # Audio generation settings
    SUBSCRIPTION_KEY = "GJiz1uCPNtu7zn4pADr9wDoDyCjuSCA5reSr6RHCt5g0HhxqW6AUJQQJ99AKACGhslBXJ3w3AAAYACOGBbTq"  # Update with your actual Azure key
    REGION = "centralindia"  # Ensure this is the correct region for your Azure resources
    
    # Check if the subscription key is provided
    if not SUBSCRIPTION_KEY or SUBSCRIPTION_KEY == "Your Actual Azure Subscription Key":
        print("Azure subscription key is not set.")
        return None

    # Create output directories
    os.makedirs("summarized/audio", exist_ok=True)  # Corrected exist_ok parameter

    # Generate unique ID for this audio
    unique_id = str(uuid.uuid4())
    output_file = f"summarized/audio/{unique_id}.mp3"

    # Generate conversation
    conversation_text = generate_conversation_for_audio(context)
    if not conversation_text:
        print("Failed to generate conversation text.")
        return None

    try:
        # Initialize TTS and generate audio using Azure SDK
        speech_config = speechsdk.SpeechConfig(subscription=SUBSCRIPTION_KEY, region=REGION)
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        result = synthesizer.speak_text_async(conversation_text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return output_file
        else:
            print(f"Speech synthesis failed: {result.reason}")
            return None
    except Exception as e:
        print(f"Error during speech synthesis: {str(e)}")
        return None

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        youtube_urls = data.get('youtube_urls', [])
        pdf_paths = data.get('pdf_paths', [])
        print(pdf_paths)
        
        if not user_message:
            return jsonify({'error': 'Message is required', 'status': 'error'}), 400

        # Get combined context from all sources
        context = get_context(youtube_urls, pdf_paths)
        
        # Create the prompt
        prompt = f"""
        Context:
        {context}
        
        User question: {user_message}
        
        Format your response using markdown with:
        - Headers for main points
        - Code blocks where appropriate
        - Lists and tables when needed
        - Bold and italic for emphasis
        """

        # Initialize the model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config
        )

        # Generate response with the combined context
        try:
            response = model.generate_content(prompt)
            if hasattr(response, 'text'):
                return jsonify({
                    'response': response.text,
                    'status': 'success'
                })
            else:
                return jsonify({
                    'error': 'Invalid response from model',
                    'status': 'error'
                }), 500
        except Exception as model_error:
            return jsonify({
                'error': f'Model error: {str(model_error)}',
                'status': 'error'
            }), 500

    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/generate-summary', methods=['POST'])
def generate_summary():
    try:
        data = request.json
        youtube_urls = data.get('youtube_urls', [])
        pdf_paths = data.get('pdf_paths', [])
        
        if not youtube_urls and not pdf_paths:
            return jsonify({
                'summary': 'No sources selected. Please select YouTube videos or PDF files.',
                'status': 'success'
            })

        # Get combined context
        context = get_context(youtube_urls, pdf_paths)
        if not context:
            return jsonify({
                'summary': 'Could not extract content from the selected sources.',
                'status': 'success'
            })

        # Create summarization prompt
        summarization_prompt = f"""
        Please provide a comprehensive summary of the following content using markdown formatting.
        Include:
        - ## Main Topics as H2 headers
        - ### Subtopics as H3 headers
        - Bullet points for key concepts
        - Code blocks for technical content
        - Tables for comparing information
        - *Italic* for emphasis and `inline code` for technical terms

        Content to summarize:
        {context}
        """

        # Initialize model and generate summary
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                **generation_config,
                "temperature": 0.7,
                "max_output_tokens": 2048,
            }
        )

        # Generate summary
        response = model.generate_content(summarization_prompt)
        
        if hasattr(response, 'text'):
            return jsonify({
                'summary': response.text,
                'status': 'success'
            })
        else:
            return jsonify({
                'error': 'Invalid response from model',
                'status': 'error'
            }), 500

    except Exception as e:
        print(f"Error in generate_summary: {str(e)}")  # Add debugging
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part', 'status': 'error'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file', 'status': 'error'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({
            'status': 'success',
            'file_path': filepath
        })

    return jsonify({'error': 'Invalid file type', 'status': 'error'}), 400

@app.route('/api/generate-audio', methods=['POST'])
def generate_audio():
    try:
        data = request.json
        youtube_urls = data.get('youtube_urls', [])
        pdf_paths = data.get('pdf_paths', [])
        
        if not youtube_urls and not pdf_paths:
            return jsonify({
                'error': 'No sources selected',
                'status': 'error'
            }), 400

        # Get context from sources
        context = get_context(youtube_urls, pdf_paths)
        if not context:
            return jsonify({
                'error': 'Could not extract content from sources',
                'status': 'error'
            }), 400

        # Generate audio file
        audio_file = process_audio_generation(context)
        
        if audio_file and os.path.exists(audio_file):
            return jsonify({
                'status': 'success',
                'audio_path': audio_file
            })
        else:
            return jsonify({
                'error': 'Failed to generate audio',
                'status': 'error'
            }), 500

    except Exception as e:
        print(f"Error in /api/generate-audio: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    try:
        data = request.json
        youtube_urls = data.get('youtube_urls', [])
        pdf_paths = data.get('pdf_paths', [])
        
        if not youtube_urls and not pdf_paths:
            return jsonify({
                'questions': [],
                'status': 'success'
            })

        # Get combined context
        context = get_context(youtube_urls, pdf_paths)
        if not context:
            return jsonify({
                'questions': [],
                'status': 'success'
            })

        # Create questions generation prompt
        questions_prompt = f"""
        Based on the following content, generate 5 relevant and insightful questions that would help someone better understand the material.
        Make the questions specific to the content and avoid generic questions.
        Return the questions as a simple list, one per line.

        Content:
        {context}
        """

        # Initialize model and generate questions
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                **generation_config,
                "temperature": 0.7,
                "max_output_tokens": 1024,
            }
        )

        # Generate questions
        response = model.generate_content(questions_prompt)
        
        if hasattr(response, 'text'):
            # Parse the response text into a list of questions
            questions = [q.strip() for q in response.text.split('\n') if q.strip()]
            return jsonify({
                'questions': questions,
                'status': 'success'
            })
        else:
            return jsonify({
                'error': 'Invalid response from model',
                'status': 'error'
            }), 500

    except Exception as e:
        print(f"Error in generate_questions: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('summarized/audio', filename)

def download_pdfs(pdf_urls):
    pdf_paths = []
    for url in pdf_urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            filename = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(url))
            with open(filename, 'wb') as f:
                f.write(response.content)
            pdf_paths.append(filename)
        except Exception as e:
            print(f"Error downloading PDF from {url}: {str(e)}")
    return pdf_paths

@socketio.on('resource_removed')
def on_resource_removed(data):
    try:
        room_id = data['roomId']
        resource = data['resource']
        sender = data.get('sender', 'Anonymous')

        if not room_id or room_id not in rooms:
            return

        # Remove resource from the room
        rooms[room_id].resources = [res for res in rooms[room_id].resources if res != resource]

        # Broadcast updated resources to all participants
        emit('resource_updated', {
            'resources': rooms[room_id].resources,
            'sender': sender
        }, room=room_id)

        # Send system message about resource removal
        emit('new_message', {
            'sender': 'System',

            'content': f'{sender} removed a resource: {resource["name"]}'
        }, room=room_id)

    except Exception as e:
        print(f"Error removing resource: {str(e)}")
        emit('error', {'message': 'Failed to remove resource'}, room=request.sid)

if __name__ == '__main__':
    socketio.run(app, debug=True)

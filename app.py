import streamlit as st
import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import logging

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="YouTube Personality Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }

    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        animation: fadeIn 0.3s ease;
    }

    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-left: 20%;
        border-radius: 15px 15px 5px 15px;
    }

    .bot-message {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        margin-right: 20%;
        border-radius: 15px 15px 15px 5px;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }

    .status-success {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }

    .status-warning {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }

    .status-error {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

class PersonalityBot:
    def __init__(self):
        self.client = None
        self.personality_prompt = ""
        self.is_initialized = False

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.client = OpenAI(api_key=api_key)

    def extract_video_id(self, url):
        """Extract video ID from various YouTube URL formats"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        # If it's already just the video ID
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url

        raise ValueError("Invalid YouTube URL or video ID")

    def get_transcript(self, video_url_or_id):
        """Get transcript from a single video"""
        try:
            video_id = self.extract_video_id(video_url_or_id)
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([entry["text"] for entry in transcript])
        except Exception as e:
            st.error(f"Error getting transcript for {video_url_or_id}: {str(e)}")
            return None

    def analyze_personality(self, transcripts_text, person_name):
        """Create personality prompt from transcripts"""
        analysis_prompt = f"""
        Analyze the following transcripts from {person_name}'s YouTube videos and create a personality profile for an AI chatbot.

        Focus on:
        1. Speaking style and tone
        2. Common phrases and expressions
        3. How they explain concepts
        4. Their personality traits and humor
        5. Technical knowledge level
        6. Communication patterns

        Transcripts:
        {transcripts_text[:6000]}

        Create a detailed system prompt that will make an AI assistant respond exactly like {person_name}.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=1000,
                temperature=0.3
            )

            personality_analysis = response.choices[0].message.content

            # Create the final system prompt
            system_prompt = f"""
You are an AI assistant that perfectly mimics {person_name}'s personality and communication style based on their YouTube content.

PERSONALITY ANALYSIS:
{personality_analysis}

COMMUNICATION GUIDELINES:
- Respond exactly as {person_name} would
- Use their typical expressions and speaking style
- Match their energy level and tone
- Explain things the way they do
- Include their characteristic humor and personality quirks
- Stay true to their areas of expertise

SAMPLE SPEECH PATTERNS FROM TRANSCRIPTS:
{transcripts_text[:2000]}

Always maintain {person_name}'s authentic voice while being helpful and engaging.
"""

            return system_prompt

        except Exception as e:
            st.error(f"Error analyzing personality: {str(e)}")
            return None

    def initialize(self, video_urls, person_name):
        """Initialize the bot with personality from videos"""
        if not self.client:
            return False, "OpenAI API key not found. Please check your .env file."

        # Get transcripts from all videos
        all_transcripts = []
        successful_videos = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, url in enumerate(video_urls):
            status_text.text(f"Processing video {i+1}/{len(video_urls)}...")
            progress_bar.progress((i) / len(video_urls))

            transcript = self.get_transcript(url)
            if transcript:
                all_transcripts.append(transcript)
                successful_videos += 1

        progress_bar.progress(1.0)
        status_text.text("Analyzing personality...")

        if not all_transcripts:
            return False, "No transcripts could be extracted from the provided videos."

        # Combine all transcripts
        combined_transcripts = " ".join(all_transcripts)

        # Analyze personality
        self.personality_prompt = self.analyze_personality(combined_transcripts, person_name)

        if self.personality_prompt:
            self.is_initialized = True
            status_text.text("✅ Personality analysis complete!")
            return True, f"Successfully initialized with {successful_videos} videos!"
        else:
            return False, "Failed to analyze personality."

    def chat(self, message, conversation_history):
        """Chat with the personality bot"""
        if not self.is_initialized or not self.client:
            return "Bot not initialized. Please set up the personality first."

        # Prepare messages
        messages = [{"role": "system", "content": self.personality_prompt}]

        # Add conversation history (last 10 messages)
        for msg in conversation_history[-10:]:
            messages.append(msg)

        # Add current message
        messages.append({"role": "user", "content": message})

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error: {str(e)}"

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = PersonalityBot()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'person_name' not in st.session_state:
    st.session_state.person_name = ""

# Header
st.markdown('<h1 class="main-header">🤖 YouTube Personality Chatbot</h1>', unsafe_allow_html=True)

# Check OpenAI API key
if not os.getenv('OPENAI_API_KEY'):
    st.markdown("""
    <div class="status-box status-error">
        <strong>⚠️ OpenAI API Key Missing</strong><br>
        Please create a <code>.env</code> file with your OpenAI API key:<br>
        <code>OPENAI_API_KEY=your_api_key_here</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Sidebar for setup
with st.sidebar:
    st.header("🔧 Setup Personality")

    # Status display
    if st.session_state.bot.is_initialized:
        st.markdown("""
        <div class="status-box status-success">
            <strong>✅ Bot Initialized</strong><br>
            Ready to chat!
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.person_name:
            st.write(f"**Chatting with:** {st.session_state.person_name}")

        if st.button("🔄 Reset Bot"):
            st.session_state.bot = PersonalityBot()
            st.session_state.messages = []
            st.session_state.person_name = ""
            st.rerun()
    else:
        st.markdown("""
        <div class="status-box status-warning">
            <strong>⚠️ Bot Not Initialized</strong><br>
            Please set up the personality below.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Person name
    person_name = st.text_input(
        "Person's Name",
        placeholder="e.g., Tech Reviewer, John Doe",
        help="Name of the YouTuber whose personality you want to mimic"
    )

    # Video URLs
    st.subheader("YouTube Videos")
    video_urls_text = st.text_area(
        "Video URLs (one per line)",
        height=150,
        placeholder="https://youtube.com/watch?v=...\nhttps://youtu.be/...\nvideo_id",
        help="Add YouTube video URLs or video IDs, one per line"
    )

    # Parse video URLs
    video_urls = [url.strip() for url in video_urls_text.split('\n') if url.strip()]

    if video_urls:
        st.write(f"📹 {len(video_urls)} video(s) added")

    # Initialize button
    if st.button("🚀 Initialize Personality", type="primary", disabled=not (person_name and video_urls)):
        if person_name and video_urls:
            with st.spinner("Extracting transcripts and analyzing personality..."):
                success, message = st.session_state.bot.initialize(video_urls, person_name)

                if success:
                    st.session_state.person_name = person_name
                    st.session_state.messages = []  # Clear chat
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        else:
            st.warning("Please provide both person name and video URLs.")

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 Chat")

with col2:
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if st.session_state.bot.is_initialized:
    # Display messages
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>{st.session_state.person_name}:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)

    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your message:",
            placeholder=f"Ask {st.session_state.person_name} anything...",
            height=100,
            key="user_message"
        )

        submitted = st.form_submit_button("📤 Send", type="primary")

        if submitted and user_input.strip():
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Get bot response
            with st.spinner(f"{st.session_state.person_name} is thinking..."):
                bot_response = st.session_state.bot.chat(user_input, st.session_state.messages[:-1])
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

            st.rerun()

    # Show tips
    if not st.session_state.messages:
        st.info(f"💡 Start a conversation with {st.session_state.person_name}! Ask about their expertise, request explanations, or just chat casually.")

else:
    # Setup instructions
    st.markdown("""
    ### 🎯 How to Get Started

    1. **Enter Person's Name**: Add the name of the YouTuber in the sidebar
    2. **Add Videos**: Paste YouTube video URLs (3-10 videos recommended)
    3. **Initialize**: Click "Initialize Personality" and wait for analysis
    4. **Chat**: Start conversing with the AI personality!

    **Tips for Best Results:**
    - Use videos where the person talks extensively
    - Include various types of content (tutorials, vlogs, Q&As)
    - More videos = better personality modeling
    - Choose recent videos that showcase their typical style
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Built with ❤️ using Streamlit and OpenAI GPT-4<br>
</div>
""", unsafe_allow_html=True)

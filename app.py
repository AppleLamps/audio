# app.py
import streamlit as st
import asyncio
import sounddevice as sd
import base64
import websockets
import json
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import numpy as np # Needed for audio data type

# --- Load Environment Variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Configuration ---
# Allow user selection in Streamlit UI
AVAILABLE_MODELS = ["gpt-4o-mini-transcribe", "gpt-4o-transcribe"]
LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean",
    # Add more languages as needed (ensure OpenAI supports them)
}
AUDIO_FORMAT = "pcm16"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024 # Samples per chunk

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide")
st.title("🎙️ Real-time Speech Translator")
st.markdown("Speak into your microphone and see the live translation.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Configuration")
    source_lang_code = st.selectbox("Source Language (Spoken)", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x], index=1) # Default Spanish
    target_lang_name = st.selectbox("Target Language (Translation)", options=list(LANGUAGES.values()), index=0) # Default English
    transcription_model = st.selectbox("Transcription Model", options=AVAILABLE_MODELS, index=0) # Default mini

with col2:
    st.subheader("Controls")
    start_button = st.button("Start Listening", key="start_button")
    stop_button = st.button("Stop Listening", key="stop_button", disabled=True) # Disabled initially

st.divider()

col_transcription, col_translation = st.columns(2)
with col_transcription:
    st.subheader(f"Transcription ({source_lang_code})")
    transcription_placeholder = st.empty()
    transcription_placeholder.text_area("Live Transcription", value="", height=200, key="transcription_area", disabled=True)

with col_translation:
    st.subheader(f"Translation ({target_lang_name})")
    translation_placeholder = st.empty()
    translation_placeholder.text_area("Live Translation", value="", height=200, key="translation_area", disabled=True)

status_placeholder = st.empty()

# --- State Management ---
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'main_task' not in st.session_state:
    st.session_state.main_task = None
if 'audio_stream' not in st.session_state:
    st.session_state.audio_stream = None
if 'transcription_text' not in st.session_state:
    st.session_state.transcription_text = ""
if 'translation_text' not in st.session_state:
    st.session_state.translation_text = ""

# --- Queues (Global within App Context) ---
# Use asyncio queues managed within the async context
audio_input_queue = None
transcription_output_queue = None

# --- Core Logic ---
async def audio_capture_task(loop):
    """Captures audio and puts it into the queue."""
    global audio_input_queue
    if audio_input_queue is None:
        st.error("Audio queue not initialized.")
        return

    audio_event = asyncio.Event()

    def audio_callback(indata, frames, time, status):
        """This runs in a separate thread, so use threadsafe methods."""
        if status:
            print(f"Audio status: {status}") # Log status issues
        try:
            # Use call_soon_threadsafe to put data into the async queue
            loop.call_soon_threadsafe(audio_input_queue.put_nowait, indata.copy())
        except Exception as e:
            print(f"Error in audio callback: {e}") # Log errors

    try:
        st.session_state.audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16', # Matches pcm16
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        )
        st.session_state.audio_stream.start()
        status_placeholder.info("🎙️ Recording audio...")
        # Keep the task running while the stream is active
        while st.session_state.is_running:
            await asyncio.sleep(0.1)
        print("Audio capture task finished.")

    except sd.PortAudioError as pae:
         st.error(f"PortAudio Error: {pae}. Do you have a microphone connected and configured?")
         st.session_state.is_running = False # Stop the process
    except Exception as e:
        st.error(f"Error starting audio stream: {e}")
        st.session_state.is_running = False # Stop the process
    finally:
        if st.session_state.audio_stream:
            try:
                st.session_state.audio_stream.stop()
                st.session_state.audio_stream.close()
                print("Audio stream stopped and closed.")
            except Exception as e_close:
                 print(f"Error stopping audio stream: {e_close}")
            st.session_state.audio_stream = None
        status_placeholder.info("Audio recording stopped.")


async def realtime_transcriber(websocket, loop):
    """Handles sending audio and receiving transcriptions."""
    global audio_input_queue, transcription_output_queue
    if audio_input_queue is None or transcription_output_queue is None:
         st.error("Queues not initialized for transcriber.")
         return

    accumulated_transcript = ""
    current_full_transcript = "" # Store the latest full transcript for display

    async def sender(ws):
        print("Audio sender started.")
        while st.session_state.is_running:
            try:
                # Get audio chunk (already encoded in capture task for simplicity)
                audio_chunk_raw = await asyncio.wait_for(audio_input_queue.get(), timeout=1.0)
                audio_chunk_b64 = base64.b64encode(audio_chunk_raw.tobytes()).decode('utf-8')

                event = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_chunk_b64
                }
                await ws.send(json.dumps(event))
                audio_input_queue.task_done()
                # Optional: Small sleep if needed, but wait_for helps
                # await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                continue # No audio data, keep checking
            except asyncio.CancelledError:
                print("Audio sender cancelled.")
                break
            except Exception as e:
                st.error(f"Error in audio sender: {e}")
                print(f"Error in sender: {e}")
                break
        print("Audio sender finished.")


    async def receiver(ws):
        nonlocal accumulated_transcript, current_full_transcript
        print("Transcription receiver started.")
        while st.session_state.is_running:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                event = json.loads(message)

                if event.get("type") == "response.text.delta":
                    delta = event.get("response", {}).get("text", {}).get("delta", "")
                    accumulated_transcript += delta
                    # Update the UI incrementally
                    st.session_state.transcription_text = current_full_transcript + accumulated_transcript
                    transcription_placeholder.text_area("Live Transcription", value=st.session_state.transcription_text, height=200, key="transcription_area_update", disabled=True)

                elif event.get("type") == "response.done":
                    full_text = event.get("response", {}).get("output", [{}])[0].get("text", "")
                    if full_text and full_text.strip():
                        final_segment = full_text.strip()
                        print(f"Transcribed segment: {final_segment}")
                        # Add finalized segment to the display text
                        current_full_transcript += final_segment + " "
                        st.session_state.transcription_text = current_full_transcript
                        transcription_placeholder.text_area("Live Transcription", value=st.session_state.transcription_text, height=200, key="transcription_area_done", disabled=True)

                        # Put the complete segment into the queue for translation
                        await transcription_output_queue.put(final_segment)

                    accumulated_transcript = "" # Reset for next delta sequence

                elif event.get("type") == "error":
                     error_msg = f"Realtime API Error: {event.get('error', {}).get('message')}"
                     st.error(error_msg)
                     print(error_msg)
                     st.session_state.is_running = False # Stop on error

            except asyncio.TimeoutError:
                 continue # No message received, keep listening
            except asyncio.CancelledError:
                 print("Transcription receiver cancelled.")
                 break
            except websockets.exceptions.ConnectionClosedOK:
                 print("WebSocket connection closed normally.")
                 break
            except Exception as e:
                 st.error(f"Error processing WebSocket message: {e}")
                 print(f"Error processing message: {e}\nMessage: {message if 'message' in locals() else 'N/A'}")
                 st.session_state.is_running = False # Stop on error
                 break
        print("Transcription receiver finished.")

    # Create and run sender/receiver tasks concurrently
    sender_task = loop.create_task(sender(websocket))
    receiver_task = loop.create_task(receiver(websocket))

    # Wait for tasks to complete or be cancelled
    done, pending = await asyncio.wait(
        [sender_task, receiver_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel any pending tasks if one completes/errors out
    for task in pending:
        task.cancel()
    # Wait for cancellations to process
    if pending:
        await asyncio.wait(pending)

    print("Realtime transcriber tasks finished.")


async def text_translator(loop):
    """Translates text segments using Chat Completions API."""
    global transcription_output_queue
    if transcription_output_queue is None:
        st.error("Transcription output queue not initialized for translator.")
        return

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    print("Translator started, waiting for text...")
    current_full_translation = "" # Store the latest full translation

    while st.session_state.is_running:
        try:
            text_to_translate = await asyncio.wait_for(transcription_output_queue.get(), timeout=1.0)
            if text_to_translate:
                print(f"Translating: {text_to_translate}")
                accumulated_translation_segment = ""
                try:
                    stream = await client.chat.completions.create(
                        model="gpt-4o", # Or gpt-4o-mini if sufficient
                        messages=[
                            {"role": "system", "content": f"You are a concise translator. Translate the following text from {LANGUAGES[source_lang_code]} into {target_lang_name}. Output only the translation."},
                            {"role": "user", "content": text_to_translate}
                        ],
                        stream=True,
                    )
                    async for chunk in stream:
                        content = chunk.choices[0].delta.content or ""
                        if content:
                            accumulated_translation_segment += content
                            # Update UI incrementally within the segment
                            st.session_state.translation_text = current_full_translation + accumulated_translation_segment
                            translation_placeholder.text_area("Live Translation", value=st.session_state.translation_text, height=200, key="translation_area_update", disabled=True)

                    # Finalize the segment translation
                    current_full_translation += accumulated_translation_segment + " " # Add space between segments
                    st.session_state.translation_text = current_full_translation
                    translation_placeholder.text_area("Live Translation", value=st.session_state.translation_text, height=200, key="translation_area_done", disabled=True)
                    print(f"Translated segment: {accumulated_translation_segment}")

                except Exception as e_openai:
                    st.error(f"OpenAI Translation Error: {e_openai}")
                    print(f"OpenAI Translation Error: {e_openai}")
                    # Don't stop the whole process for a single translation error? Or maybe do?
                    # Consider adding the untranslated text to the output with an error marker

            transcription_output_queue.task_done()

        except asyncio.TimeoutError:
            continue # No transcription segment received, keep waiting
        except asyncio.CancelledError:
            print("Translator task cancelled.")
            break
        except Exception as e:
            st.error(f"Error in translator task: {e}")
            print(f"Error in translator: {e}")
            break
    print("Text translator task finished.")

async def main_task_wrapper(loop):
    """Manages the overall async workflow."""
    global audio_input_queue, transcription_output_queue
    audio_input_queue = asyncio.Queue()
    transcription_output_queue = asyncio.Queue()

    REALTIME_URL = f"wss://api.openai.com/v1/realtime?intent=transcription&language={source_lang_code}&model={transcription_model}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    websocket_task = None
    translator_task = None
    audio_task = None

    try:
        async with websockets.connect(REALTIME_URL, extra_headers=headers) as ws:
            print("WebSocket connected. Configuring session...")
            status_placeholder.info("WebSocket connected. Configuring session...")

            # Configure session (optional if parameters are in URL, but good practice)
            # config_event = { ... } # Add if specific config needed beyond URL params
            # await ws.send(json.dumps(config_event))
            # print("Session configured.")
            # status_placeholder.info("Session configured.")

            # Start the interdependent tasks
            audio_task = loop.create_task(audio_capture_task(loop))
            websocket_task = loop.create_task(realtime_transcriber(ws, loop))
            translator_task = loop.create_task(text_translator(loop))

            # Wait for tasks to complete (or be cancelled by stop button)
            done, pending = await asyncio.wait(
                 [audio_task, websocket_task, translator_task],
                 return_when=asyncio.FIRST_COMPLETED # Stop if any task finishes/errors
            )

            # If one task finishes/errors, signal others to stop
            st.session_state.is_running = False
            print("One of the main tasks completed or failed. Signaling stop.")

    except websockets.exceptions.InvalidURI:
        st.error(f"Invalid WebSocket URI: {REALTIME_URL}")
        st.session_state.is_running = False
    except websockets.exceptions.WebSocketException as wse:
        st.error(f"WebSocket connection failed: {wse}")
        st.session_state.is_running = False
    except asyncio.CancelledError:
        print("Main task wrapper cancelled.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        print(f"Main task wrapper error: {e}")
        st.session_state.is_running = False
    finally:
        print("Cleaning up main tasks...")
        # Ensure all tasks are cancelled on exit
        tasks_to_cancel = [t for t in [audio_task, websocket_task, translator_task] if t and not t.done()]
        if tasks_to_cancel:
            for task in tasks_to_cancel:
                task.cancel()
            await asyncio.wait(tasks_to_cancel) # Wait for cancellations
        print("Main task cleanup complete.")
        status_placeholder.info("Process stopped.")


# --- Button Actions ---
if start_button and not st.session_state.is_running:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")
    else:
        st.session_state.is_running = True
        st.session_state.transcription_text = "" # Clear previous text
        st.session_state.translation_text = ""   # Clear previous text
        transcription_placeholder.text_area("Live Transcription", value="", height=200, key="transcription_area_start", disabled=True)
        translation_placeholder.text_area("Live Translation", value="", height=200, key="translation_area_start", disabled=True)
        status_placeholder.info("Starting process...")

        try:
            # Get the current event loop or create a new one if needed
            loop = asyncio.get_event_loop()
        except RuntimeError:
             loop = asyncio.new_event_loop()
             asyncio.set_event_loop(loop)

        # Start the main async task
        st.session_state.main_task = loop.create_task(main_task_wrapper(loop))

        # Disable start, enable stop
        st.experimental_rerun()


if stop_button and st.session_state.is_running:
    st.session_state.is_running = False
    status_placeholder.warning("Stopping process...")

    # Cancel the main task gracefully
    if st.session_state.main_task and not st.session_state.main_task.done():
        st.session_state.main_task.cancel()

    # The finally block in main_task_wrapper and audio_capture_task handles cleanup
    # Reset state variables
    st.session_state.main_task = None

    # Give a moment for tasks to potentially clean up before rerun
    # This might need adjustment or a more robust signaling mechanism
    # time.sleep(0.5) # Avoid using time.sleep in async context if possible

    # Update button states
    st.experimental_rerun()


# --- UI Updates based on State ---
# Update button states based on is_running
if 'start_button' in st.session_state: # Check if widgets rendered
    st.button("Start Listening", key="start_button_update", disabled=st.session_state.is_running)
    st.button("Stop Listening", key="stop_button_update", disabled=not st.session_state.is_running)

# Display current text (handled within async tasks updating placeholders)


# --- Keep Streamlit running ---
# The Streamlit script re-runs, but the async tasks run in the background
# managed by the asyncio event loop started/managed via button clicks.

# Add a note about microphone permissions
st.sidebar.info(
    "ℹ️ Your browser might ask for microphone permissions when you click 'Start Listening'."
    " Please ensure you grant permission."
)
st.sidebar.info(
     "Audio processing happens locally for capture, then streams to OpenAI."
)
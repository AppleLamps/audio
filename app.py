# app.py
import streamlit as st
import asyncio
import sounddevice as sd
import base64
import websockets
import json
import os
import threading # Import threading
from openai import AsyncOpenAI
from dotenv import load_dotenv
import numpy as np
import queue # Use standard queue for thread-safe communication

# --- Load Environment Variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Configuration ---
AVAILABLE_MODELS = ["gpt-4o-mini-transcribe", "gpt-4o-transcribe"]
LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean",
}
AUDIO_FORMAT = "pcm16"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

# --- Streamlit UI Setup ---
# (UI setup remains the same as before)
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
    # Check state before rendering buttons
    start_disabled = st.session_state.get('is_running', False)
    stop_disabled = not st.session_state.get('is_running', False)
    start_button = st.button("Start Listening", key="start_button", disabled=start_disabled)
    stop_button = st.button("Stop Listening", key="stop_button", disabled=stop_disabled)

st.divider()

col_transcription, col_translation = st.columns(2)
with col_transcription:
    st.subheader(f"Transcription ({source_lang_code})")
    transcription_placeholder = st.empty()

with col_translation:
    st.subheader(f"Translation ({target_lang_name})")
    translation_placeholder = st.empty()

status_placeholder = st.empty()

# --- State Management ---
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'audio_stream' not in st.session_state:
    st.session_state.audio_stream = None
if 'transcription_text' not in st.session_state:
    st.session_state.transcription_text = ""
if 'translation_text' not in st.session_state:
    st.session_state.translation_text = ""
if 'async_thread' not in st.session_state:
    st.session_state.async_thread = None # Store the background thread

# --- Queues (Thread-safe Queues for Cross-Thread Communication) ---
# These queues will be accessed by the audio callback (separate thread),
# the asyncio loop (background thread), and Streamlit (main thread).
# Using standard queue.Queue for simplicity here. Asyncio queues are loop-bound.
ui_update_queue = queue.Queue()

# --- Core Async Logic (will run in background thread) ---
# Note: These async functions CANNOT directly interact with st.* elements
# They should put results into the ui_update_queue

async def audio_capture_task(loop, audio_q):
    """Captures audio and puts it into the asyncio queue."""
    audio_event = asyncio.Event()

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        try:
            # Use threadsafe put_nowait for asyncio queue from sync thread
            loop.call_soon_threadsafe(audio_q.put_nowait, indata.copy())
        except Exception as e:
            print(f"Error in audio callback: {e}")

    try:
        audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        )
        st.session_state.audio_stream = audio_stream # Store ref for stopping
        audio_stream.start()
        print("🎙️ Audio recording started...")
        ui_update_queue.put(("status", "info", "🎙️ Recording audio..."))
        while st.session_state.get('is_running', False):
            await asyncio.sleep(0.1)
        print("Audio capture task finishing.")

    except sd.PortAudioError as pae:
         print(f"PortAudio Error: {pae}")
         ui_update_queue.put(("status", "error", f"PortAudio Error: {pae}. Check microphone."))
         st.session_state.is_running = False
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        ui_update_queue.put(("status", "error", f"Error starting audio: {e}"))
        st.session_state.is_running = False
    finally:
        if st.session_state.audio_stream:
            try:
                st.session_state.audio_stream.stop()
                st.session_state.audio_stream.close()
                print("Audio stream stopped and closed.")
            except Exception as e_close:
                 print(f"Error stopping audio stream: {e_close}")
            st.session_state.audio_stream = None
        print("Audio recording stopped.")
        ui_update_queue.put(("status", "info", "Audio recording stopped."))


async def realtime_transcriber(websocket, audio_q, transcription_q):
    """Handles sending audio and receiving transcriptions."""
    accumulated_transcript = ""
    current_full_transcript = st.session_state.get("transcription_text", "") # Use state

    async def sender(ws):
        print("Audio sender started.")
        while st.session_state.get('is_running', False):
            try:
                audio_chunk_raw = await asyncio.wait_for(audio_q.get(), timeout=1.0)
                audio_chunk_b64 = base64.b64encode(audio_chunk_raw.tobytes()).decode('utf-8')
                event = {"type": "input_audio_buffer.append", "audio": audio_chunk_b64}
                await ws.send(json.dumps(event))
                audio_q.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                print("Audio sender cancelled.")
                break
            except Exception as e:
                print(f"Error in sender: {e}")
                ui_update_queue.put(("status", "error", f"Audio sender error: {e}"))
                break
        print("Audio sender finished.")

    async def receiver(ws):
        nonlocal accumulated_transcript, current_full_transcript
        print("Transcription receiver started.")
        while st.session_state.get('is_running', False):
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                event = json.loads(message)

                if event.get("type") == "response.text.delta":
                    delta = event.get("response", {}).get("text", {}).get("delta", "")
                    accumulated_transcript += delta
                    # Put incremental update into UI queue
                    ui_update_queue.put(("transcription", current_full_transcript + accumulated_transcript))

                elif event.get("type") == "response.done":
                    full_text = event.get("response", {}).get("output", [{}])[0].get("text", "")
                    if full_text and full_text.strip():
                        final_segment = full_text.strip()
                        print(f"Transcribed segment: {final_segment}")
                        current_full_transcript += final_segment + " "
                        # Put final segment update into UI queue
                        ui_update_queue.put(("transcription", current_full_transcript))
                        # Put segment into queue for translator
                        await transcription_q.put(final_segment)
                    accumulated_transcript = ""

                elif event.get("type") == "error":
                     error_msg = f"Realtime API Error: {event.get('error', {}).get('message')}"
                     print(error_msg)
                     ui_update_queue.put(("status", "error", error_msg))
                     st.session_state.is_running = False # Stop on error

            except asyncio.TimeoutError:
                 continue
            except asyncio.CancelledError:
                 print("Transcription receiver cancelled.")
                 break
            except websockets.exceptions.ConnectionClosed:
                 print("WebSocket connection closed.")
                 if st.session_state.get('is_running', False): # If running, it's unexpected
                      ui_update_queue.put(("status", "warning", "WebSocket closed unexpectedly."))
                 break # Exit loop if connection closed
            except Exception as e:
                 error_msg = f"WebSocket processing error: {e}"
                 print(error_msg)
                 ui_update_queue.put(("status", "error", error_msg))
                 st.session_state.is_running = False
                 break
        print("Transcription receiver finished.")

    # Run sender/receiver concurrently
    tasks = [loop.create_task(sender(websocket)), loop.create_task(receiver(websocket))]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.wait(pending)
    print("Realtime transcriber tasks finished.")


async def text_translator(transcription_q):
    """Translates text segments using Chat Completions API."""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    print("Translator started, waiting for text...")
    current_full_translation = st.session_state.get("translation_text", "") # Use state

    while st.session_state.get('is_running', False):
        try:
            text_to_translate = await asyncio.wait_for(transcription_q.get(), timeout=1.0)
            if text_to_translate:
                print(f"Translating: {text_to_translate}")
                accumulated_translation_segment = ""
                try:
                    stream = await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": f"Translate from {LANGUAGES[source_lang_code]} to {target_lang_name}. Output only the translation."},
                            {"role": "user", "content": text_to_translate}
                        ],
                        stream=True,
                    )
                    async for chunk in stream:
                        content = chunk.choices[0].delta.content or ""
                        if content:
                            accumulated_translation_segment += content
                            # Put incremental translation update into UI queue
                            ui_update_queue.put(("translation", current_full_translation + accumulated_translation_segment))

                    # Finalize the segment translation
                    current_full_translation += accumulated_translation_segment + " "
                    ui_update_queue.put(("translation", current_full_translation)) # Final update for segment
                    print(f"Translated segment: {accumulated_translation_segment}")

                except Exception as e_openai:
                    error_msg = f"Translation API Error: {e_openai}"
                    print(error_msg)
                    ui_update_queue.put(("status", "error", error_msg))
                    # Optionally add marker to translation text:
                    # current_full_translation += f"[Translation Error] "
                    # ui_update_queue.put(("translation", current_full_translation))

            transcription_q.task_done()

        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            print("Translator task cancelled.")
            break
        except Exception as e:
            error_msg = f"Translator task error: {e}"
            print(error_msg)
            ui_update_queue.put(("status", "error", error_msg))
            break
    print("Text translator task finished.")


async def main_async_tasks(loop):
    """Manages the overall async workflow within the background thread."""
    print("main_async_tasks started")
    audio_q = asyncio.Queue()
    transcription_q = asyncio.Queue()

    # Use selected values from Streamlit UI (captured when thread started)
    selected_source_lang = st.session_state.selected_source_lang
    selected_target_lang = st.session_state.selected_target_lang
    selected_model = st.session_state.selected_model

    REALTIME_URL = f"wss://api.openai.com/v1/realtime?intent=transcription&language={selected_source_lang}&model={selected_model}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    websocket_task = None
    translator_task = None
    audio_task = None

    try:
        async with websockets.connect(REALTIME_URL, extra_headers=headers, logger=None) as ws: # Disable default logger noise
            print("WebSocket connected.")
            ui_update_queue.put(("status", "info", "WebSocket connected. Starting services..."))

            audio_task = loop.create_task(audio_capture_task(loop, audio_q))
            websocket_task = loop.create_task(realtime_transcriber(ws, audio_q, transcription_q))
            translator_task = loop.create_task(text_translator(transcription_q))

            # Wait for tasks to complete or be cancelled by stop flag
            done, pending = await asyncio.wait(
                 [audio_task, websocket_task, translator_task],
                 return_when=asyncio.FIRST_COMPLETED
            )

            print("One of the main async tasks completed or failed.")
            # If one task finishes, it likely means is_running became False or an error occurred
            # Ensure is_running is False so other tasks also stop cleanly
            st.session_state.is_running = False

    except websockets.exceptions.InvalidURI:
        ui_update_queue.put(("status", "error", f"Invalid WebSocket URI"))
    except websockets.exceptions.WebSocketException as wse:
        ui_update_queue.put(("status", "error", f"WebSocket connection failed: {wse}"))
    except asyncio.CancelledError:
        print("Main async tasks cancelled.")
    except Exception as e:
        error_msg = f"Async tasks error: {e}"
        print(error_msg)
        ui_update_queue.put(("status", "error", error_msg))
    finally:
        print("Cleaning up async tasks...")
        # is_running should be False here, tasks check this flag to exit loops
        # Explicit cancellation might still be needed if tasks are stuck
        tasks_to_cancel = [t for t in [audio_task, websocket_task, translator_task] if t and not t.done()]
        if tasks_to_cancel:
            for task in tasks_to_cancel:
                task.cancel()
            await asyncio.wait(tasks_to_cancel, timeout=2.0) # Give time for cancellation
        print("Async tasks cleanup complete.")
        ui_update_queue.put(("status", "info", "Process stopped."))
        ui_update_queue.put(("finished", None)) # Signal the main thread we are done


def run_async_loop(async_coro):
    """Runs the given async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(async_coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)
        print("Async loop closed.")

# --- Button Actions (Main Thread) ---
if start_button:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")
    elif not st.session_state.is_running:
        st.session_state.is_running = True
        st.session_state.transcription_text = "" # Clear previous text
        st.session_state.translation_text = ""   # Clear previous text
        # Store selected config at start time for the background thread
        st.session_state.selected_source_lang = source_lang_code
        st.session_state.selected_target_lang = target_lang_name
        st.session_state.selected_model = transcription_model

        status_placeholder.info("Starting background process...")
        print("Starting background thread...")

        # Create and start the background thread
        # Pass the main async tasks coroutine to the runner function
        thread = threading.Thread(target=run_async_loop, args=(main_async_tasks(asyncio.new_event_loop()),), daemon=True)
        st.session_state.async_thread = thread
        thread.start()

        st.rerun() # Update button states

if stop_button:
    if st.session_state.is_running:
        status_placeholder.warning("Stopping process...")
        print("Stop button pressed. Setting is_running to False.")
        st.session_state.is_running = False # Signal async tasks to stop

        # Wait briefly for thread to potentially finish based on the flag
        if st.session_state.async_thread and st.session_state.async_thread.is_alive():
             st.session_state.async_thread.join(timeout=3.0) # Wait max 3 seconds

        if st.session_state.async_thread and st.session_state.async_thread.is_alive():
             print("Thread still alive after timeout, may need forceful closure (not implemented).")
             # Forcibly stopping threads is generally discouraged.
             # Relying on the tasks checking 'is_running' is preferred.

        st.session_state.async_thread = None # Clear thread reference
        print("Stop action completed.")
        st.rerun() # Update button states


# --- UI Updates from Queue (Main Thread) ---
# This part runs on every Streamlit rerun when the app is potentially active
# It processes messages put into the queue by the background thread
new_transcription = None
new_translation = None
status_updates = []

while not ui_update_queue.empty():
    message_type, payload, *extra = ui_update_queue.get()
    if message_type == "transcription":
        new_transcription = payload
        st.session_state.transcription_text = payload # Update state
    elif message_type == "translation":
        new_translation = payload
        st.session_state.translation_text = payload # Update state
    elif message_type == "status":
        status_type, message = payload, extra[0] if extra else "" # Unpack status tuple
        status_updates.append((status_type, message))
    elif message_type == "finished":
         print("Received finished signal in main thread.")
         # Ensure state reflects stopped status if thread finished unexpectedly
         if st.session_state.is_running:
              st.session_state.is_running = False
              st.rerun() # Rerun to update buttons if stopped unexpectedly

# Update UI elements if new data arrived
# Use the latest text from session_state which was updated from queue
transcription_placeholder.text_area(
    "Live Transcription",
    value=st.session_state.transcription_text,
    height=200,
    key="transcription_area_display",
    disabled=True
)
translation_placeholder.text_area(
    "Live Translation",
    value=st.session_state.translation_text,
    height=200,
    key="translation_area_display",
    disabled=True
)

# Display the last status message received in this cycle
if status_updates:
    status_type, message = status_updates[-1]
    if status_type == "info":
        status_placeholder.info(message)
    elif status_type == "warning":
        status_placeholder.warning(message)
    elif status_type == "error":
        status_placeholder.error(message)

# --- Sidebar Info ---
st.sidebar.info(
    "ℹ️ Your browser might ask for microphone permissions when you click 'Start Listening'."
    " Please ensure you grant permission."
)
st.sidebar.info(
     "Audio processing happens locally for capture, then streams to OpenAI."
)

# Add a periodic rerun to keep checking the queue if the thread is running
if st.session_state.get('is_running', False):
     asyncio.sleep(0.5) # Short sleep to prevent tight loop if nothing in queue
     st.rerun()

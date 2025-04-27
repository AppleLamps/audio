# app.py
import streamlit as st
import asyncio
import sounddevice as sd
import base64
import websockets
import json
import os
import threading
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
st.set_page_config(layout="wide")
st.title("🎙️ Real-time Speech Translator")
st.markdown("Speak into your microphone and see the live translation.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Configuration")
    # Get values directly, will be passed to thread
    selected_source_lang_code = st.selectbox("Source Language (Spoken)", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x], index=1)
    selected_target_lang_name = st.selectbox("Target Language (Translation)", options=list(LANGUAGES.values()), index=0)
    selected_transcription_model = st.selectbox("Transcription Model", options=AVAILABLE_MODELS, index=0)

with col2:
    st.subheader("Controls")
    start_disabled = st.session_state.get('is_running', False)
    stop_disabled = not st.session_state.get('is_running', False)
    start_button = st.button("Start Listening", key="start_button", disabled=start_disabled)
    stop_button = st.button("Stop Listening", key="stop_button", disabled=stop_disabled)

st.divider()

col_transcription, col_translation = st.columns(2)
with col_transcription:
    st.subheader(f"Transcription ({selected_source_lang_code})")
    transcription_placeholder = st.empty()

with col_translation:
    st.subheader(f"Translation ({selected_target_lang_name})")
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
    st.session_state.async_thread = None

# --- Queue for UI Updates (Thread-safe) ---
ui_update_queue = queue.Queue()

# --- Core Async Logic (runs in background thread) ---
# Modified to accept config and use thread-safe queue for UI updates

async def audio_capture_task(loop, audio_q):
    """Captures audio and puts it into the asyncio queue."""
    # This function doesn't need external config, only the loop and queue
    audio_event = asyncio.Event()

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"[Audio Thread] Audio status: {status}")
        try:
            loop.call_soon_threadsafe(audio_q.put_nowait, indata.copy())
        except Exception as e:
            print(f"[Audio Thread] Error in audio callback: {e}")

    audio_stream = None # Define locally
    try:
        audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        )
        st.session_state.audio_stream = audio_stream # Store ref for stopping only
        audio_stream.start()
        print("[Async Thread] 🎙️ Audio recording started...")
        ui_update_queue.put(("status", "info", "🎙️ Recording audio..."))
        # Check is_running flag from session_state (reading is generally safer than writing)
        while st.session_state.get('is_running', False):
            await asyncio.sleep(0.1)
        print("[Async Thread] Audio capture task finishing (is_running is False).")

    except sd.PortAudioError as pae:
         print(f"[Async Thread] PortAudio Error: {pae}")
         ui_update_queue.put(("status", "error", f"PortAudio Error: {pae}. Check microphone."))
         # Signal main process to stop if audio fails critically
         ui_update_queue.put(("stop_signal", None))
    except Exception as e:
        print(f"[Async Thread] Error starting audio stream: {e}")
        ui_update_queue.put(("status", "error", f"Error starting audio: {e}"))
        ui_update_queue.put(("stop_signal", None))
    finally:
        if audio_stream: # Use local variable
            try:
                # Check if stream was actually started before stopping
                if audio_stream.active:
                     audio_stream.stop()
                     audio_stream.close()
                     print("[Async Thread] Audio stream stopped and closed.")
            except Exception as e_close:
                 print(f"[Async Thread] Error stopping audio stream: {e_close}")
        # Don't clear st.session_state.audio_stream here, let main thread handle it
        print("[Async Thread] Audio recording stopped.")
        ui_update_queue.put(("status", "info", "Audio recording stopped."))


async def realtime_transcriber(websocket, audio_q, transcription_q):
    """Handles sending audio and receiving transcriptions."""
    # Reads session_state only for is_running flag
    accumulated_transcript = ""
    # Initialize local state for text accumulation
    current_full_transcript = st.session_state.get("transcription_text", "") # Read initial state

    async def sender(ws):
        print("[Async Thread] Audio sender started.")
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
                print("[Async Thread] Audio sender cancelled.")
                break
            except Exception as e:
                print(f"[Async Thread] Error in sender: {e}")
                ui_update_queue.put(("status", "error", f"Audio sender error: {e}"))
                ui_update_queue.put(("stop_signal", None)) # Signal stop on error
                break
        print("[Async Thread] Audio sender finished.")

    async def receiver(ws):
        nonlocal accumulated_transcript, current_full_transcript
        print("[Async Thread] Transcription receiver started.")
        while st.session_state.get('is_running', False):
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                event = json.loads(message)

                if event.get("type") == "response.text.delta":
                    delta = event.get("response", {}).get("text", {}).get("delta", "")
                    accumulated_transcript += delta
                    ui_update_queue.put(("transcription", current_full_transcript + accumulated_transcript))

                elif event.get("type") == "response.done":
                    full_text = event.get("response", {}).get("output", [{}])[0].get("text", "")
                    if full_text and full_text.strip():
                        final_segment = full_text.strip()
                        print(f"[Async Thread] Transcribed segment: {final_segment}")
                        current_full_transcript += final_segment + " "
                        ui_update_queue.put(("transcription", current_full_transcript))
                        await transcription_q.put(final_segment)
                    accumulated_transcript = ""

                elif event.get("type") == "error":
                     error_msg = f"Realtime API Error: {event.get('error', {}).get('message')}"
                     print(f"[Async Thread] {error_msg}")
                     ui_update_queue.put(("status", "error", error_msg))
                     ui_update_queue.put(("stop_signal", None))

            except asyncio.TimeoutError:
                 continue
            except asyncio.CancelledError:
                 print("[Async Thread] Transcription receiver cancelled.")
                 break
            except websockets.exceptions.ConnectionClosed:
                 print("[Async Thread] WebSocket connection closed.")
                 if st.session_state.get('is_running', False):
                      ui_update_queue.put(("status", "warning", "WebSocket closed unexpectedly."))
                      ui_update_queue.put(("stop_signal", None))
                 break
            except Exception as e:
                 error_msg = f"WebSocket processing error: {e}"
                 print(f"[Async Thread] {error_msg}")
                 ui_update_queue.put(("status", "error", error_msg))
                 ui_update_queue.put(("stop_signal", None))
                 break
        print("[Async Thread] Transcription receiver finished.")

    # Use the current loop obtained in run_async_loop
    loop = asyncio.get_running_loop()
    sender_task = loop.create_task(sender(websocket))
    receiver_task = loop.create_task(receiver(websocket))
    tasks = [sender_task, receiver_task]

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.wait(pending, timeout=1.0) # Wait briefly for cancellations
    print("[Async Thread] Realtime transcriber tasks finished.")


async def text_translator(transcription_q, source_lang_code, target_lang_name, api_key):
    """Translates text segments using Chat Completions API."""
    # Accepts config now, reads session_state only for is_running
    client = AsyncOpenAI(api_key=api_key) # Use passed API key
    print("[Async Thread] Translator started, waiting for text...")
    current_full_translation = st.session_state.get("translation_text", "") # Read initial state

    while st.session_state.get('is_running', False):
        try:
            text_to_translate = await asyncio.wait_for(transcription_q.get(), timeout=1.0)
            if text_to_translate:
                print(f"[Async Thread] Translating: {text_to_translate}")
                accumulated_translation_segment = ""
                try:
                    # Use passed language config
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
                            ui_update_queue.put(("translation", current_full_translation + accumulated_translation_segment))

                    current_full_translation += accumulated_translation_segment + " "
                    ui_update_queue.put(("translation", current_full_translation))
                    print(f"[Async Thread] Translated segment: {accumulated_translation_segment}")

                except Exception as e_openai:
                    error_msg = f"Translation API Error: {e_openai}"
                    print(f"[Async Thread] {error_msg}")
                    ui_update_queue.put(("status", "error", error_msg))
                    # No stop signal here? Let transcription continue maybe.

            transcription_q.task_done()

        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            print("[Async Thread] Translator task cancelled.")
            break
        except Exception as e:
            error_msg = f"Translator task error: {e}"
            print(f"[Async Thread] {error_msg}")
            ui_update_queue.put(("status", "error", error_msg))
            ui_update_queue.put(("stop_signal", None)) # Signal stop on error
            break
    print("[Async Thread] Text translator task finished.")


async def main_async_tasks(loop, config):
    """Manages the overall async workflow within the background thread."""
    # Accepts config dictionary
    print("[Async Thread] main_async_tasks started")
    audio_q = asyncio.Queue()
    transcription_q = asyncio.Queue()

    # Use config passed as argument
    source_lang = config['source_lang']
    target_lang = config['target_lang']
    model = config['model']
    api_key = config['api_key']

    REALTIME_URL = f"wss://api.openai.com/v1/realtime?intent=transcription&language={source_lang}&model={model}"
    headers = {
        "Authorization": f"Bearer {api_key}", # Use passed API key
        "OpenAI-Beta": "realtime=v1"
    }

    websocket_task = None
    translator_task = None
    audio_task = None

    try:
        # Note: Increased connection timeout
        async with websockets.connect(REALTIME_URL, extra_headers=headers, logger=None, open_timeout=10) as ws:
            print("[Async Thread] WebSocket connected.")
            ui_update_queue.put(("status", "info", "WebSocket connected. Starting services..."))

            # Pass necessary config down
            audio_task = loop.create_task(audio_capture_task(loop, audio_q))
            websocket_task = loop.create_task(realtime_transcriber(ws, audio_q, transcription_q))
            translator_task = loop.create_task(text_translator(transcription_q, source_lang, target_lang, api_key))

            tasks = [audio_task, websocket_task, translator_task]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            print("[Async Thread] One of the main async tasks completed or failed.")
            # If one task finishes, signal stop (is_running should already be False or will be set by stop_signal)
            # Ensure is_running is False so other tasks also stop cleanly
            if st.session_state.get('is_running', False):
                 ui_update_queue.put(("stop_signal", None)) # Ensure main thread knows to stop

    except websockets.exceptions.InvalidURI:
        ui_update_queue.put(("status", "error", f"Invalid WebSocket URI"))
        ui_update_queue.put(("stop_signal", None))
    except websockets.exceptions.WebSocketException as wse:
        ui_update_queue.put(("status", "error", f"WebSocket connection failed: {wse}"))
        ui_update_queue.put(("stop_signal", None))
    except asyncio.CancelledError:
        print("[Async Thread] Main async tasks cancelled.")
    except Exception as e:
        error_msg = f"Async tasks error: {e}"
        print(f"[Async Thread] {error_msg}")
        ui_update_queue.put(("status", "error", error_msg))
        ui_update_queue.put(("stop_signal", None))
    finally:
        print("[Async Thread] Cleaning up async tasks...")
        # Ensure is_running is false so tasks exit their loops
        st.session_state.is_running = False
        # Cancel any tasks that might still be pending (e.g., stuck waiting)
        tasks_to_cancel = [t for t in [audio_task, websocket_task, translator_task] if t and not t.done()]
        if tasks_to_cancel:
            for task in tasks_to_cancel:
                task.cancel()
            await asyncio.wait(tasks_to_cancel, timeout=2.0)
        print("[Async Thread] Async tasks cleanup complete.")
        ui_update_queue.put(("status", "info", "Process stopped."))
        ui_update_queue.put(("finished", None)) # Signal the main thread


def run_async_loop(async_coro):
    """Runs the given async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        print(f"[Thread Runner {threading.current_thread().name}] Starting event loop...")
        loop.run_until_complete(async_coro)
        print(f"[Thread Runner {threading.current_thread().name}] Event loop finished.")
    except Exception as e:
         print(f"[Thread Runner {threading.current_thread().name}] Error in event loop: {e}")
         # Send error to main thread if possible
         try:
              ui_update_queue.put(("status", "error", f"Background thread error: {e}"))
              ui_update_queue.put(("stop_signal", None))
         except Exception: # Ignore if queue putting fails during shutdown
              pass
    finally:
        try:
            # Graceful shutdown of loop tasks
            tasks = asyncio.all_tasks(loop=loop)
            if tasks:
                print(f"[Thread Runner {threading.current_thread().name}] Cancelling {len(tasks)} remaining tasks...")
                for task in tasks:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                print(f"[Thread Runner {threading.current_thread().name}] Remaining tasks cancelled.")

            loop.run_until_complete(loop.shutdown_asyncgens())
            print(f"[Thread Runner {threading.current_thread().name}] Async generators shutdown.")
        except Exception as e_shutdown:
             print(f"[Thread Runner {threading.current_thread().name}] Error during loop shutdown: {e_shutdown}")
        finally:
             loop.close()
             asyncio.set_event_loop(None)
             print(f"[Thread Runner {threading.current_thread().name}] Async loop closed.")

# --- Button Actions (Main Thread) ---
if start_button:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")
    elif not st.session_state.is_running:
        st.session_state.is_running = True
        st.session_state.transcription_text = ""
        st.session_state.translation_text = ""
        # Clear placeholders immediately
        transcription_placeholder.text_area("Live Transcription", value="", height=200, key="transcription_area_cleared", disabled=True)
        translation_placeholder.text_area("Live Translation", value="", height=200, key="translation_area_cleared", disabled=True)

        # Prepare config to pass to the thread
        config = {
            'source_lang': selected_source_lang_code,
            'target_lang': selected_target_lang_name,
            'model': selected_transcription_model,
            'api_key': OPENAI_API_KEY
        }

        status_placeholder.info("Starting background process...")
        print("[Main Thread] Starting background thread...")

        # Create and start the background thread
        thread = threading.Thread(
            target=run_async_loop,
            # Pass the coroutine object with its arguments
            args=(main_async_tasks(asyncio.new_event_loop(), config),),
            daemon=True # Ensure thread exits if main process exits
        )
        st.session_state.async_thread = thread
        thread.start()

        st.rerun()

if stop_button:
    if st.session_state.is_running:
        status_placeholder.warning("Stopping process...")
        print("[Main Thread] Stop button pressed. Setting is_running to False.")
        st.session_state.is_running = False # Signal async tasks in background thread

        # Wait briefly for thread to finish based on the flag
        thread = st.session_state.get('async_thread')
        if thread and thread.is_alive():
             print("[Main Thread] Waiting for background thread to join...")
             thread.join(timeout=5.0) # Increased timeout
             if thread.is_alive():
                  print("[Main Thread] Background thread still alive after join timeout.")
             else:
                  print("[Main Thread] Background thread joined successfully.")

        # Clean up audio stream explicitly if it wasn't closed by the thread
        if st.session_state.get('audio_stream') and st.session_state.audio_stream.active:
             print("[Main Thread] Forcing audio stream closure.")
             try:
                  st.session_state.audio_stream.stop()
                  st.session_state.audio_stream.close()
             except Exception as e:
                  print(f"[Main Thread] Error closing audio stream on stop: {e}")
        st.session_state.audio_stream = None
        st.session_state.async_thread = None
        print("[Main Thread] Stop action completed.")
        # Ensure UI reflects stopped state
        status_placeholder.info("Process stopped.")
        st.rerun()

# --- UI Updates from Queue (Main Thread) ---
while not ui_update_queue.empty():
    try:
        message_type, payload, *extra = ui_update_queue.get_nowait() # Use get_nowait
        if message_type == "transcription":
            st.session_state.transcription_text = payload
        elif message_type == "translation":
            st.session_state.translation_text = payload
        elif message_type == "status":
            status_type, message = payload, extra[0] if extra else ""
            if status_type == "info": status_placeholder.info(message)
            elif status_type == "warning": status_placeholder.warning(message)
            elif status_type == "error": status_placeholder.error(message)
        elif message_type == "stop_signal":
             print("[Main Thread] Received stop signal from background thread.")
             if st.session_state.is_running:
                  st.session_state.is_running = False
                  # Trigger the stop logic cleanly if not already stopping
                  # This might cause a double stop attempt, but should be safe
                  stop_button = True # Simulate stop button press for cleanup logic
                  st.rerun() # Rerun to process stop
        elif message_type == "finished":
             print("[Main Thread] Received finished signal.")
             if st.session_state.is_running: # If thread finished but we thought it was running
                  st.session_state.is_running = False
                  st.rerun() # Rerun to update button states

        ui_update_queue.task_done() # Mark task as done

    except queue.Empty:
        break # No more messages for now
    except Exception as e:
        print(f"[Main Thread] Error processing UI queue: {e}")
        # Avoid breaking the main loop, just log the error

# Update UI elements from session state
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

# --- Sidebar Info ---
st.sidebar.info("ℹ️ Grant microphone permissions if prompted.")
st.sidebar.info("Audio streams to OpenAI.")

# --- Optional: Periodic Rerun (Use time.sleep in main thread if needed) ---
# import time
# if st.session_state.get('is_running', False):
#      time.sleep(0.3) # Use regular sleep in main thread
#      st.rerun()
# Removing the sleep/rerun here - UI updates should happen when queue has data.

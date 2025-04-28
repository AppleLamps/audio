document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const targetLangSelect = document.getElementById('targetLang');
    const apiKeyInput = document.getElementById('apiKeyInput'); // Added
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const statusDiv = document.getElementById('status');
    // const apiStatusDiv = document.getElementById('apiStatus'); // Removed
    const transcriptionOutput = document.getElementById('transcriptionOutput');
    const translationOutput = document.getElementById('translationOutput');
    const targetLangLabel = document.getElementById('targetLangLabel');

    // --- State Variables ---
    let isRunning = false;
    // let userApiKey = null; // Removed - will read from input directly
    let selectedTargetLang = 'English'; // Default

    // --- Fetch API Key on Load --- (Removed)
    // fetchApiKey();

    // Audio Processing
    let audioContext = null;
    let audioProcessor = null;
    let micStream = null;
    const SAMPLE_RATE = 16000; // Target sample rate for OpenAI
    const CHUNK_DURATION_MS = 100; // Send audio chunks duration

    // WebSocket
    let webSocket = null;
    const REALTIME_URL = "wss://api.openai.com/v1/realtime"; // Base URL

    // Translation
    let currentTranscriptionSegment = "";
    let fullTranscription = "";
    let fullTranslation = "";

    // --- API Key Management --- (Removed fetchApiKey function)

    // --- Event Listeners ---
    apiKeyInput.addEventListener('input', () => {
        // Enable start button only if API key input has value
        startButton.disabled = !apiKeyInput.value.trim();
    });
    startButton.addEventListener('click', startProcess);
    stopButton.addEventListener('click', stopProcess);
    targetLangSelect.addEventListener('change', (e) => {
        selectedTargetLang = e.target.value;
        targetLangLabel.textContent = selectedTargetLang; // Update label in heading
    });

    // --- Core Functions ---

    async function startProcess() {
        selectedTargetLang = targetLangSelect.value; // Ensure latest value
        targetLangLabel.textContent = selectedTargetLang;
        const userApiKey = apiKeyInput.value.trim(); // Read key from input

        if (!userApiKey) {
            updateStatus("Please enter your OpenAI API key.", "error");
            return;
        }
        // Basic format check (optional but helpful)
        if (!userApiKey.startsWith('sk-')) {
             updateStatus("Invalid API key format. It should start with 'sk-'.", "error");
             return;
        }

        if (isRunning) return;

        isRunning = true;
        updateUIState(true);
        updateStatus("Initializing...", "processing");
        fullTranscription = "";
        fullTranslation = "";
        transcriptionOutput.value = "";
        translationOutput.value = "";

        try {
            await setupAudio(); // Request mic and set up processing
            connectWebSocket(); // Connect after audio is ready
        } catch (error) {
            console.error("Initialization failed:", error);
            updateStatus(`Error starting: ${error.message}`, "error");
            stopProcess(); // Cleanup on error
        }
    }

    function stopProcess() {
        if (!isRunning && !webSocket && !audioContext) return; // Avoid multiple stops
        console.log("Stopping process...");
        isRunning = false;

        // 1. Close WebSocket with a graceful shutdown
        if (webSocket) {
            try {
                // Send graceful close if connection is open
                if (webSocket.readyState === WebSocket.OPEN) {
                    // Send a session.close message to properly end the session
                    webSocket.send(JSON.stringify({ type: "session.close" }));
                    console.log("Sent session.close message");

                    // Give a small delay to allow the message to be sent before closing
                    setTimeout(() => {
                        if (webSocket) {
                            webSocket.close(1000, "User initiated close");
                            console.log("WebSocket closed gracefully.");
                        }
                    }, 200);
                } else {
                    // If not open, just close it
                    webSocket.close();
                    console.log("WebSocket closed (was not in OPEN state).");
                }
            } catch (error) {
                console.error("Error closing WebSocket:", error);
            } finally {
                webSocket = null; // Ensure it's cleared
            }
        }

        // 2. Stop Audio Processing
        if (micStream) {
            try {
                micStream.getTracks().forEach(track => {
                    track.stop();
                    console.log(`Track ${track.id} stopped.`);
                });
            } catch (error) {
                console.error("Error stopping microphone tracks:", error);
            } finally {
                micStream = null;
                console.log("Microphone stream reference cleared.");
            }
        }

        if (audioProcessor) {
            try {
                audioProcessor.disconnect(); // Disconnect the node
                console.log("Audio processor disconnected.");

                // For AudioWorkletNode, we should also post a message to clean up
                if (audioProcessor.port && audioProcessor.port.postMessage) {
                    audioProcessor.port.postMessage({ type: 'cleanup' });
                    console.log("Sent cleanup message to AudioWorklet.");
                }
            } catch (error) {
                console.error("Error disconnecting audio processor:", error);
            } finally {
                audioProcessor = null;
                console.log("Audio processor reference cleared.");
            }
        }

        if (audioContext) {
            // Close context after a short delay to allow disconnects to settle
            setTimeout(() => {
                if (audioContext && audioContext.state !== 'closed') {
                    audioContext.close()
                        .then(() => console.log("AudioContext closed successfully."))
                        .catch(err => console.error("Error closing AudioContext:", err))
                        .finally(() => {
                            audioContext = null;
                            console.log("AudioContext reference cleared.");
                        });
                } else {
                    audioContext = null;
                    console.log("AudioContext was already closed or null.");
                }
            }, 300); // Slightly longer delay for cleanup
        }

        // 3. Update UI
        updateUIState(false);
        updateStatus("Process stopped.", "stopped");

        // 4. Reset state variables if needed
        currentTranscriptionSegment = "";
    }

    async function setupAudio() {
        console.log("Setting up audio...");
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Browser does not support audio capture (getUserMedia).");
        }

        try {
            micStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: SAMPLE_RATE, // Request desired rate
                    channelCount: 1,
                    // Other constraints like noiseSuppression, echoCancellation can be added
                }
            });

            // Use standard AudioContext
            try {
                audioContext = new AudioContext({
                    sampleRate: SAMPLE_RATE // Ensure context matches desired rate
                });
                console.log("Using standard AudioContext");
            } catch (e) {
                console.error("Failed to create standard AudioContext:", e);

                // Try with webkit prefix as fallback (for older browsers)
                try {
                    // @ts-ignore - Ignore TypeScript warning about webkitAudioContext
                    const WebkitAudioContext = window.webkitAudioContext;
                    if (WebkitAudioContext) {
                        audioContext = new WebkitAudioContext({
                            sampleRate: SAMPLE_RATE
                        });
                        console.log("Using webkitAudioContext as fallback");
                    } else {
                        throw new Error("webkitAudioContext not available");
                    }
                } catch (fallbackError) {
                    console.error("Failed to create fallback AudioContext:", fallbackError);
                    throw new Error("AudioContext not supported in this browser");
                }
            }

            // Calculate desired buffer size based on chunk duration
            const bufferSize = Math.round(SAMPLE_RATE * CHUNK_DURATION_MS / 1000);
            // Ensure bufferSize is a power of 2 (256, 512, 1024, etc.)
            const validBufferSizes = [256, 512, 1024, 2048, 4096, 8192, 16384];
            const nodeBufferSize = validBufferSizes.reduce((prev, curr) =>
                (Math.abs(curr - bufferSize) < Math.abs(prev - bufferSize) ? curr : prev)
            );
            console.log(`Using AudioWorklet buffer size: ${nodeBufferSize}`);

            // Load and register the audio worklet processor
            try {
                await audioContext.audioWorklet.addModule('audio-processor.js');
                console.log("AudioWorklet module loaded successfully");

                // Create the audio worklet node
                audioProcessor = new AudioWorkletNode(audioContext, 'audio-sampler-processor');

                // Configure the processor with the desired buffer size
                audioProcessor.port.postMessage({
                    type: 'configure',
                    bufferSize: nodeBufferSize
                });

                // Handle messages from the processor
                audioProcessor.port.onmessage = (event) => {
                    if (!isRunning || !webSocket || webSocket.readyState !== WebSocket.OPEN) return;

                    if (event.data.type === 'audio-buffer') {
                        const inputData = event.data.audioData;
                        // Convert Float32Array to PCM16 Int16Array
                        const pcm16Buffer = float32ToPCM16(inputData);
                        // Encode PCM16 data to Base64
                        const base64Audio = pcm16ToBase64(pcm16Buffer);

                        // Send audio data via WebSocket
                        webSocket.send(JSON.stringify({
                            type: "input_audio_buffer.append",
                            audio: base64Audio
                        }));
                    }
                };

                // Connect the audio processing chain
                const source = audioContext.createMediaStreamSource(micStream);
                source.connect(audioProcessor);
                audioProcessor.connect(audioContext.destination); // Connect to destination to keep processing active

                console.log("Audio setup complete with AudioWorklet.");
                updateStatus("Audio ready. Connecting to OpenAI...", "processing");

            } catch (workletError) {
                console.error("Failed to load AudioWorklet:", workletError);
                throw new Error(`AudioWorklet failed to load: ${workletError.message}. Your browser may not support this feature.`);
            }

        } catch (err) {
            console.error("Error accessing microphone:", err);
            if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                throw new Error("Microphone permission denied.");
            } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
                throw new Error("No microphone found.");
            }
            else {
                throw new Error(`Error setting up audio: ${err.message}`);
            }
        }
    }

    function connectWebSocket() {
        if (webSocket && webSocket.readyState === WebSocket.OPEN) {
            console.log("WebSocket already open.");
            return;
        }

        // Validate API key format
        if (!userApiKey || !userApiKey.startsWith('sk-')) {
            updateStatus("Invalid API key format. Should start with 'sk-'", "error");
            stopProcess();
            return;
        }

        const userApiKey = apiKeyInput.value.trim(); // Read key again just in case

        // Construct URL and Subprotocols
        // Note: Model can be passed in URL or session update. Using URL here.
        const model = "gpt-4o-mini-transcribe"; // Or select dynamically
        // Include both model and intent parameters for better compatibility
        const wsUrl = `${REALTIME_URL}?model=${model}&intent=transcription`;

        // Prepare subprotocols with proper format
        const subProtocols = [
            "realtime",
            `openai-insecure-api-key.${userApiKey}`, // Use key from input
            "openai-beta.realtime-v1"
        ];

        console.log(`Connecting to WebSocket: ${wsUrl} with protocols...`);
        updateStatus("Connecting WebSocket...", "processing");

        try {
            webSocket = new WebSocket(wsUrl, subProtocols);

            // Set a connection timeout
            const connectionTimeout = setTimeout(() => {
                if (webSocket && webSocket.readyState !== WebSocket.OPEN) {
                    console.error("WebSocket connection timeout");
                    updateStatus("Connection timeout. Please check your API key and internet connection.", "error");
                    webSocket.close();
                }
            }, 10000); // 10 second timeout

            webSocket.onopen = () => {
                console.log("WebSocket Connected!");
                clearTimeout(connectionTimeout); // Clear the timeout
                updateStatus("Listening... Speak now!", "listening");

                // Send initial configuration message
                const configMsg = {
                    type: "session.update",
                    data: {
                        input_mode: "speech",
                        output_mode: "text",
                        speech_language: "auto", // Auto-detect language
                        text_language: "auto",
                        temperature: 0.7,
                        model: "gpt-4o-mini-transcribe", // Ensure model is specified in session update
                        intent: "transcription", // Explicitly set intent
                        // Add any other configuration parameters needed
                    }
                };
                webSocket.send(JSON.stringify(configMsg));
                console.log("Sent configuration message:", configMsg);
            };

            webSocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log("WS Message:", data); // Enable debugging to see all messages

                    // Handle both response.text.delta (newer API) and transcript.text.delta (as mentioned in docs)
                    if (data.type === "response.text.delta" || data.type === "transcript.text.delta") {
                        // Extract delta text from either format
                        const delta = data.response?.text?.delta || data.delta || "";
                        currentTranscriptionSegment += delta;
                        transcriptionOutput.value = fullTranscription + currentTranscriptionSegment;
                        transcriptionOutput.scrollTop = transcriptionOutput.scrollHeight; // Auto-scroll
                    }
                    // Handle both response.done (newer API) and transcript.text.done (as mentioned in docs)
                    else if (data.type === "response.done" || data.type === "transcript.text.done") {
                        // Extract text from either format
                        const segmentText = data.response?.output?.[0]?.text?.trim() || data.text?.trim();
                        if (segmentText) {
                            console.log("Transcription segment done:", segmentText);
                            fullTranscription += segmentText + " ";
                            transcriptionOutput.value = fullTranscription;
                            transcriptionOutput.scrollTop = transcriptionOutput.scrollHeight;
                            // Trigger translation for the completed segment
                            translateText(segmentText);
                        }
                        currentTranscriptionSegment = ""; // Reset for next segment
                    } else if (data.type === "error") {
                        const errorMsg = data.error?.message || "Unknown WebSocket error";
                        const errorCode = data.error?.code || "unknown";
                        console.error(`WebSocket Error: Code=${errorCode}, Message=${errorMsg}`);
                        updateStatus(`WebSocket Error: ${errorMsg} (Code: ${errorCode})`, "error");
                        stopProcess(); // Stop on error
                    } else if (data.type === "session.created") {
                        console.log("Session created:", data.session?.id);
                    } else if (data.type === "session.status") {
                        console.log("Session status:", data.status);
                    } else {
                        console.log("Unhandled message type:", data.type);
                    }
                } catch (error) {
                    console.error("Error parsing WebSocket message:", error, event.data);
                }
            };

            webSocket.onerror = (error) => {
                console.error("WebSocket Error Event:", error);
                clearTimeout(connectionTimeout); // Clear the timeout

                // Log more detailed error information if available
                const errorDetails = error.message || "Unknown error";
                console.error("WebSocket error details:", errorDetails);

                updateStatus(`WebSocket connection error: ${errorDetails}. Check console for details.`, "error");
                // Don't call stopProcess here directly, wait for onclose
            };

            webSocket.onclose = (event) => {
                clearTimeout(connectionTimeout); // Clear the timeout

                // Log detailed close information
                console.log(`WebSocket Closed: Code=${event.code}, Reason='${event.reason}'`);

                // Interpret close codes
                let closeReason = "Unknown reason";
                switch (event.code) {
                    case 1000: closeReason = "Normal closure"; break;
                    case 1001: closeReason = "Going away"; break;
                    case 1002: closeReason = "Protocol error"; break;
                    case 1003: closeReason = "Unsupported data"; break;
                    case 1005: closeReason = "No status received"; break;
                    case 1006: closeReason = "Abnormal closure"; break;
                    case 1007: closeReason = "Invalid frame payload data"; break;
                    case 1008: closeReason = "Policy violation"; break;
                    case 1009: closeReason = "Message too big"; break;
                    case 1010: closeReason = "Mandatory extension"; break;
                    case 1011: closeReason = "Internal server error"; break;
                    case 1015: closeReason = "TLS handshake"; break;
                }

                console.log(`Close reason: ${closeReason}`);

                if (isRunning) { // If we were running, the closure was unexpected
                    updateStatus(`WebSocket closed unexpectedly (${closeReason}). Please try starting again.`, "error");
                    stopProcess(); // Ensure cleanup if closed unexpectedly
                } else {
                    updateStatus("WebSocket connection closed.", "stopped");
                }
                webSocket = null; // Clear reference after close
            };

        } catch (error) {
            console.error("Error creating WebSocket:", error);
            updateStatus(`Failed to create WebSocket: ${error.message}`, "error");
            stopProcess();
        }
    }

    async function translateText(textToTranslate) {
        const userApiKey = apiKeyInput.value.trim(); // Read key from input
        if (!textToTranslate || !userApiKey) return;

        console.log(`Translating segment: "${textToTranslate}" to ${selectedTargetLang}`);
        updateStatus("Translating...", "processing"); // Indicate translation activity

        const TRANSLATION_URL = "https://api.openai.com/v1/chat/completions";
        let accumulatedTranslationSegment = "";

        try {
            const response = await fetch(TRANSLATION_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${userApiKey}` // Use the input API key
                },
                body: JSON.stringify({
                    model: "gpt-4o-mini", // Using gpt-4o-mini for translation
                    messages: [
                        { role: "system", content: `Translate the following text into ${selectedTargetLang}. Output only the translation.` },
                        { role: "user", content: textToTranslate }
                    ],
                    stream: true // Enable streaming response
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ message: response.statusText }));
                throw new Error(`Translation API Error (${response.status}): ${errorData.message || 'Unknown error'}`);
            }

            // Process the stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // Process lines separated by newline (SSE format)
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep the last partial line

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const jsonData = line.substring(6).trim();
                        if (jsonData === '[DONE]') {
                            break; // Stream finished signal
                        }
                        try {
                            const parsed = JSON.parse(jsonData);
                            const deltaContent = parsed.choices?.[0]?.delta?.content;
                            if (deltaContent) {
                                accumulatedTranslationSegment += deltaContent;
                                // Update UI incrementally
                                translationOutput.value = fullTranslation + accumulatedTranslationSegment;
                                translationOutput.scrollTop = translationOutput.scrollHeight;
                            }
                        } catch (e) {
                            console.warn("Error parsing stream JSON:", e, jsonData);
                        }
                    }
                }
            }

            fullTranslation += accumulatedTranslationSegment + " "; // Add finalized segment
            translationOutput.value = fullTranslation; // Final update for the segment
            translationOutput.scrollTop = translationOutput.scrollHeight;
            console.log("Translation segment finished:", accumulatedTranslationSegment);

        } catch (error) {
            console.error("Translation Fetch Error:", error);
            updateStatus(`Translation Error: ${error.message}`, "error");
            // Optionally add error marker to translation text
            translationOutput.value += ` [Translation Failed]`;
        } finally {
            // Reset status if still "Translating..." and not stopped or errored
            if (isRunning && statusDiv.textContent === "Translating...") {
                updateStatus("Listening... Speak now!", "listening");
            }
        }
    }


    // --- Helper Functions ---

    function updateUIState(running) {
        // Start button is disabled if not running AND api key is missing
        startButton.disabled = running || !apiKeyInput.value.trim();
        stopButton.disabled = !running;
        targetLangSelect.disabled = running;
        apiKeyInput.disabled = running; // Disable API key input while running
    }

    function updateStatus(message, type = "info") {
        statusDiv.textContent = message;
        statusDiv.className = `status-${type}`; // Update class for styling
    }

    // Function to convert Float32 Array to PCM16 Int16Array
    function float32ToPCM16(float32Arr) {
        const pcm16Arr = new Int16Array(float32Arr.length);
        for (let i = 0; i < float32Arr.length; i++) {
            let s = Math.max(-1, Math.min(1, float32Arr[i]));
            // Scale to Int16 range, clamp, and convert
            pcm16Arr[i] = Math.max(-32768, Math.min(32767, s * 32767));
        }
        return pcm16Arr;
    }

    // Function to encode PCM16 Int16Array to Base64 string
    function pcm16ToBase64(pcm16Arr) {
        // Get the raw bytes of the Int16Array
        const bytes = new Uint8Array(pcm16Arr.buffer);
        // Convert byte array to binary string
        let binary = '';
        const len = bytes.byteLength;
        for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        // Encode binary string to Base64
        return btoa(binary);
    }

    // --- Initial UI State ---
    updateUIState(false); // Set initial button states

}); // End DOMContentLoaded

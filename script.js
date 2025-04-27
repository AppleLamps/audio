document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const apiKeyInput = document.getElementById('apiKey');
    const targetLangSelect = document.getElementById('targetLang');
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const statusDiv = document.getElementById('status');
    const transcriptionOutput = document.getElementById('transcriptionOutput');
    const translationOutput = document.getElementById('translationOutput');
    const targetLangLabel = document.getElementById('targetLangLabel');

    // --- State Variables ---
    let isRunning = false;
    let userApiKey = null;
    let selectedTargetLang = 'English'; // Default

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

    // --- Event Listeners ---
    startButton.addEventListener('click', startProcess);
    stopButton.addEventListener('click', stopProcess);
    targetLangSelect.addEventListener('change', (e) => {
        selectedTargetLang = e.target.value;
        targetLangLabel.textContent = selectedTargetLang; // Update label in heading
    });

    // --- Core Functions ---

    async function startProcess() {
        userApiKey = apiKeyInput.value.trim();
        selectedTargetLang = targetLangSelect.value; // Ensure latest value
        targetLangLabel.textContent = selectedTargetLang;

        if (!userApiKey) {
            updateStatus("Please enter your OpenAI API Key.", "error");
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

        // 1. Close WebSocket
        if (webSocket) {
            // Send graceful close if needed, or just close
            if (webSocket.readyState === WebSocket.OPEN) {
                // Optional: Send a specific close message if API supports it
                // webSocket.send(JSON.stringify({ type: "session.close" }));
            }
            webSocket.close();
            webSocket = null; // Ensure it's cleared
            console.log("WebSocket closed.");
        }

        // 2. Stop Audio Processing
        if (micStream) {
            micStream.getTracks().forEach(track => track.stop());
            micStream = null;
            console.log("Microphone stream stopped.");
        }
        if (audioProcessor) {
            audioProcessor.disconnect(); // Disconnect the node
            audioProcessor = null;
            console.log("Audio processor disconnected.");
        }
        if (audioContext) {
            // Close context after a short delay to allow disconnects to settle
            setTimeout(() => {
                if (audioContext && audioContext.state !== 'closed') {
                    audioContext.close().then(() => console.log("AudioContext closed."));
                }
                audioContext = null;
            }, 100); // Delay closing context slightly
        }

        updateUIState(false);
        updateStatus("Process stopped.", "stopped");
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

            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: SAMPLE_RATE // Ensure context matches desired rate
            });

            const source = audioContext.createMediaStreamSource(micStream);

            // --- Using ScriptProcessorNode (simpler, but deprecated) ---
            // Adjust bufferSize for desired chunk duration
            const bufferSize = Math.round(SAMPLE_RATE * CHUNK_DURATION_MS / 1000);
            // Ensure bufferSize is a power of 2 for ScriptProcessorNode (256, 512, 1024, etc.)
            const validBufferSizes = [256, 512, 1024, 2048, 4096, 8192, 16384];
            const nodeBufferSize = validBufferSizes.reduce((prev, curr) =>
                (Math.abs(curr - bufferSize) < Math.abs(prev - bufferSize) ? curr : prev)
            );
            console.log(`Using ScriptProcessorNode buffer size: ${nodeBufferSize}`);

            audioProcessor = audioContext.createScriptProcessor(nodeBufferSize, 1, 1);

            audioProcessor.onaudioprocess = (event) => {
                if (!isRunning || !webSocket || webSocket.readyState !== WebSocket.OPEN) return;

                const inputData = event.inputBuffer.getChannelData(0);
                // Convert Float32Array to PCM16 Int16Array
                const pcm16Buffer = float32ToPCM16(inputData);
                // Encode PCM16 data to Base64
                const base64Audio = pcm16ToBase64(pcm16Buffer);

                // Send audio data via WebSocket
                webSocket.send(JSON.stringify({
                    type: "input_audio_buffer.append",
                    audio: base64Audio
                }));
            };

            source.connect(audioProcessor);
            audioProcessor.connect(audioContext.destination); // Connect to destination to keep processing active

            console.log("Audio setup complete.");
            updateStatus("Audio ready. Connecting to OpenAI...", "processing");

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

        // Construct URL and Subprotocols
        // Note: Model can be passed in URL or session update. Using URL here.
        const model = "gpt-4o-mini-transcribe"; // Or select dynamically
        const wsUrl = `${REALTIME_URL}?model=${model}`; // Add language later if needed/supported in URL
        const subProtocols = [
            "realtime",
            `openai-insecure-api-key.${userApiKey}`, // SECURITY RISK!
            "openai-beta.realtime-v1"
            // Add org/project ID protocols if needed
        ];

        console.log(`Connecting to WebSocket: ${wsUrl} with protocols...`);
        updateStatus("Connecting WebSocket...", "processing");

        webSocket = new WebSocket(wsUrl, subProtocols);

        webSocket.onopen = () => {
            console.log("WebSocket Connected!");
            updateStatus("Listening... Speak now!", "listening");
            // Optional: Send configuration update message if needed
            // const configMsg = { ... };
            // webSocket.send(JSON.stringify(configMsg));
        };

        webSocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                // console.log("WS Message:", data); // Debugging

                if (data.type === "response.text.delta") {
                    const delta = data.response?.text?.delta || "";
                    currentTranscriptionSegment += delta;
                    transcriptionOutput.value = fullTranscription + currentTranscriptionSegment;
                    transcriptionOutput.scrollTop = transcriptionOutput.scrollHeight; // Auto-scroll
                } else if (data.type === "response.done") {
                    const segmentText = data.response?.output?.[0]?.text?.trim();
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
                    console.error("WebSocket Error:", errorMsg);
                    updateStatus(`WebSocket Error: ${errorMsg}`, "error");
                    stopProcess(); // Stop on error
                } else if (data.type === "session.created") {
                    console.log("Session created:", data.session?.id);
                }
                // Handle other message types if necessary
            } catch (error) {
                console.error("Error parsing WebSocket message:", error, event.data);
            }
        };

        webSocket.onerror = (error) => {
            console.error("WebSocket Error Event:", error);
            // The 'error' event often precedes 'close'. Let onclose handle state.
            updateStatus(`WebSocket connection error. Check console/network tab.`, "error");
            // Don't call stopProcess here directly, wait for onclose
        };

        webSocket.onclose = (event) => {
            console.log(`WebSocket Closed: Code=${event.code}, Reason='${event.reason}'`);
            if (isRunning) { // If we were running, the closure was unexpected
                updateStatus(`WebSocket closed unexpectedly (Code: ${event.code}). Please try starting again.`, "error");
                stopProcess(); // Ensure cleanup if closed unexpectedly
            } else {
                updateStatus("WebSocket connection closed.", "stopped");
            }
            webSocket = null; // Clear reference after close
        };
    }

    async function translateText(textToTranslate) {
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
                    model: "gpt-4o", // Or gpt-4o-mini
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
        startButton.disabled = running;
        stopButton.disabled = !running;
        apiKeyInput.disabled = running;
        targetLangSelect.disabled = running;
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
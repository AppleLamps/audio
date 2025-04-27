// audio-processor.js
// This file defines the AudioWorkletProcessor for real-time audio processing

class AudioSamplerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 2048; // Default buffer size
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
    this.isActive = true; // Flag to control processing

    console.log("[AudioWorklet] AudioSamplerProcessor initialized");

    // Handle messages from the main thread
    this.port.onmessage = (event) => {
      if (event.data.type === 'configure') {
        // Allow configuration of buffer size
        if (event.data.bufferSize) {
          this.bufferSize = event.data.bufferSize;
          this.buffer = new Float32Array(this.bufferSize);
          this.bufferIndex = 0;
          console.log(`[AudioWorklet] Buffer size configured to ${this.bufferSize}`);
        }
      } else if (event.data.type === 'cleanup') {
        // Handle cleanup request
        this.isActive = false;
        this.buffer = null; // Free memory
        console.log("[AudioWorklet] Cleanup requested, stopping processor");

        // Acknowledge cleanup
        this.port.postMessage({
          type: 'cleanup-complete'
        });
      }
    };
  }

  process(inputs, outputs, parameters) {
    // If we're not active, don't process anything
    if (!this.isActive) return false;

    // Get the first input channel data
    const input = inputs[0][0];

    if (!input) return true;

    // Fill the buffer with incoming audio data
    for (let i = 0; i < input.length; i++) {
      if (!this.isActive) break; // Check if we've been deactivated during processing

      this.buffer[this.bufferIndex++] = input[i];

      // When buffer is full, send it to the main thread and reset
      if (this.bufferIndex >= this.bufferSize) {
        try {
          // Clone the buffer to avoid issues with shared memory
          const audioData = this.buffer.slice(0);

          // Send the buffer to the main thread
          this.port.postMessage({
            type: 'audio-buffer',
            audioData: audioData
          });
        } catch (error) {
          console.error("[AudioWorklet] Error sending buffer:", error);
        }

        // Reset buffer index
        this.bufferIndex = 0;
      }
    }

    // Return true to keep the processor running, or false to terminate
    return this.isActive;
  }
}

// Register the processor
registerProcessor('audio-sampler-processor', AudioSamplerProcessor);

// Log registration
console.log("[AudioWorklet] AudioSamplerProcessor registered");

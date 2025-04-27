# Real-time Speech Translator

A browser-based application that provides real-time transcription and translation of spoken language using OpenAI's APIs.

## Features

- Real-time speech transcription using OpenAI's Realtime API
- Automatic translation to multiple languages
- Simple, intuitive user interface
- Secure API key management through environment variables

## How It Works

1. The application captures audio from your microphone
2. Audio is sent to OpenAI's Realtime API for transcription
3. Transcribed text is then sent to OpenAI's Chat API for translation
4. Both the original transcription and translation are displayed in real-time

## Deployment to Vercel

### Prerequisites

- A Vercel account
- An OpenAI API key

### Steps

1. Fork or clone this repository
2. Create a new project on Vercel and connect it to your repository
3. Add the following environment variable in the Vercel project settings:
   - Name: `OPENAI_API_KEY`
   - Value: Your OpenAI API key
4. Deploy the project
5. Your application will be available at the Vercel-provided URL

### Local Development

1. Clone the repository
2. Create a `.env` file in the root directory with the following content:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```
3. Install the Vercel CLI: `npm install -g vercel`
4. Run the development server: `vercel dev`
5. Open your browser to the provided URL (usually http://localhost:3000)

## Security Considerations

- The API key is stored securely in Vercel's environment variables
- The key is never exposed to the client-side code
- API requests are made through a serverless function to protect your API key

## Technologies Used

- HTML, CSS, and JavaScript for the frontend
- Web Audio API for microphone access
- WebSockets for real-time communication with OpenAI
- Vercel Serverless Functions for secure API access

## License

MIT

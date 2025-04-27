// /api/getApiKey.js
export default function handler(req, res) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return res.status(404).json({ error: 'API key not found' });
  }
  res.status(200).json({ apiKey });
}

export interface Language {
  id: string
  name: string
}

export interface Voice {
  id: string
  name: string
  filename: string
  refText: string | null
  language: string | null
}

/** OpenAI-compatible speech synthesis request. */
export interface SpeechRequest {
  model?: string
  input: string
  voice?: string
  response_format?: 'mp3' | 'opus' | 'aac' | 'flac' | 'wav' | 'pcm'
  speed?: number
  instructions?: string
  // Extensions (not part of OpenAI schema):
  language?: string
  sanitize?: boolean
  stream_format?: 'audio' | 'sse'
}

/** Legacy synthesis params kept for the LLM-sim WebSocket flow. */
export interface SynthesisParams {
  text: string
  language: string
  speed?: number
  voiceId?: string
  refAudio?: File
  refText?: string
  refVoiceName?: string
  instruct?: string
  sanitize?: boolean
}

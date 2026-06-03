export interface Language {
  id: string
  name: string
}

export interface Voice {
  id: string
  name: string
  filename: string
  refText: string | null
}

export interface SynthesisParams {
  text: string
  language: string
  speed?: number
  voiceId?: string
  refAudio?: File
  refText?: string
  refVoiceName?: string
  instruct?: string
}

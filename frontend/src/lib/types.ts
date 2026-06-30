export interface Language {
  id: string
  name: string
}

export interface Voice {
  id: string
  name: string
  kind: 'builtin' | 'custom'
}

export interface SynthesisParams {
  text: string
  language: string
  speed?: number
  totalSteps?: number
  voiceId: string
}

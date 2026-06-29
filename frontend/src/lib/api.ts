import type { Language, SpeechRequest, SynthesisParams, Voice } from './types.js'

const configured = import.meta.env.VITE_API_BASE
const BASE =
    configured && configured.length > 0 ? configured : window.location.origin

function camelize(s: string): string {
  return s.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase())
}

function camelizeKeys<T>(obj: Record<string, unknown>): T {
  return Object.fromEntries(
    Object.entries(obj).map(([k, v]) => [camelize(k), v]),
  ) as T
}

// ── Health / metadata ────────────────────────────────────────────────────

export async function checkHealth(): Promise<{ status: string; modelLoaded: boolean }> {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error('Server unreachable')
  return camelizeKeys(await res.json())
}

export async function fetchLanguages(): Promise<Language[]> {
  const res = await fetch(`${BASE}/v1/languages`)
  if (!res.ok) throw new Error('Failed to fetch languages')
  return res.json()
}

// ── Voice management ─────────────────────────────────────────────────────

export async function fetchVoices(): Promise<Voice[]> {
  const res = await fetch(`${BASE}/v1/voices`)
  if (!res.ok) throw new Error('Failed to fetch voices')
  const data: Record<string, unknown>[] = await res.json()
  return data.map((v) => camelizeKeys<Voice>(v))
}

export function voicePreviewUrl(voiceId: string): string {
  return `${BASE}/v1/voices/${encodeURIComponent(voiceId)}/preview`
}

export async function deleteVoice(voiceId: string): Promise<void> {
  const res = await fetch(`${BASE}/v1/voices/${encodeURIComponent(voiceId)}`, {
    method: 'DELETE',
  })
  if (!res.ok) {
    const detail = await res.json().then((d) => d.detail).catch(() => res.statusText)
    throw new Error(detail)
  }
}

export async function createVoice(params: {
  name: string
  refText: string
  refAudio: File
  language?: string
}): Promise<{ id: string; language: string }> {
  const form = new FormData()
  form.append('name', params.name)
  form.append('ref_text', params.refText)
  form.append('ref_audio', params.refAudio)
  if (params.language) form.append('language', params.language)

  const res = await fetch(`${BASE}/v1/voices`, { method: 'POST', body: form })
  if (!res.ok) {
    const detail = await res.json().then((d) => d.detail).catch(() => res.statusText)
    throw new Error(detail)
  }
  return res.json()
}

// ── Speech synthesis (OpenAI-compatible) ──────────────────────────────────

/** Build a SpeechRequest from the UI's SynthesisParams. */
function buildSpeechRequest(params: SynthesisParams, opts?: { streamFormat?: 'audio' | 'sse' }): SpeechRequest {
  const req: SpeechRequest = {
    input: params.text.trim(),
    language: params.language,
    response_format: 'wav', // Web Audio API decodes WAV natively
    stream_format: opts?.streamFormat ?? 'audio',
  }

  if (params.speed != null && params.speed !== 1.0) {
    req.speed = params.speed
  }

  // Voice selection: stored voice (cloning) or instructions (design mode)
  if (params.voiceId) {
    req.voice = params.voiceId
  } else if (params.instruct) {
    req.instructions = params.instruct
  }

  if (params.sanitize != null) {
    req.sanitize = params.sanitize
  }

  return req
}

/** Non-streaming synthesis — returns a single audio Blob. */
export async function synthesize(params: SynthesisParams): Promise<Blob> {
  const req = buildSpeechRequest(params, { streamFormat: 'audio' })
  const res = await fetch(`${BASE}/v1/audio/speech`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    const detail = await res.json().then((d: { detail: string }) => d.detail).catch(() => res.statusText)
    throw new Error(detail)
  }
  return res.blob()
}

/**
 * Streaming synthesis via SSE — yields WAV audio chunks as they arrive.
 *
 * Each SSE delta is base64-encoded WAV audio. We decode and yield the raw
 * ArrayBuffer so the StreamingPlayer can schedule it immediately.
 */
export async function* synthesizeStream(
  params: SynthesisParams,
  signal?: AbortSignal,
): AsyncGenerator<ArrayBuffer> {
  const req = buildSpeechRequest(params, { streamFormat: 'sse' })
  const res = await fetch(`${BASE}/v1/audio/speech`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  })
  if (!res.ok) {
    const detail = await res.json().then((d: { detail: string }) => d.detail).catch(() => res.statusText)
    throw new Error(detail)
  }

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      // Parse SSE events from the buffer
      const lines = buffer.split('\n')
      // Keep the last (potentially incomplete) line in the buffer
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const data = line.slice(6).trim()
        if (!data) continue

        try {
          const event = JSON.parse(data)
          if (event.type === 'speech.audio.delta' && event.audio) {
            // Decode base64 WAV chunk
            const binaryStr = atob(event.audio)
            const bytes = new Uint8Array(binaryStr.length)
            for (let i = 0; i < binaryStr.length; i++) {
              bytes[i] = binaryStr.charCodeAt(i)
            }
            yield bytes.buffer
          }
          // speech.audio.done signals completion
        } catch {
          // Skip malformed JSON lines
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

// ── WebSocket (LLM simulation mode) ──────────────────────────────────────

const WS_BASE = BASE.replace(/^http/, 'ws')

export function openSynthSocket(params: {
  language: string
  voiceId?: string
  speed?: number
  instruct?: string
  sanitize?: boolean
}): WebSocket {
  const url = new URL(`${WS_BASE}/v1/ws/synthesize`)
  url.searchParams.set('language', params.language)
  if (params.voiceId) url.searchParams.set('voice_id', params.voiceId)
  if (params.speed != null && params.speed !== 1.0) url.searchParams.set('speed', String(params.speed))
  if (params.instruct) url.searchParams.set('instruct', params.instruct)
  if (params.sanitize != null) url.searchParams.set('sanitize', String(params.sanitize))
  return new WebSocket(url.toString())
}

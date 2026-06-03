import type { Language, SynthesisParams, Voice } from './types.js'

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

const WS_BASE = BASE.replace(/^http/, 'ws')

function buildSynthForm(params: SynthesisParams, stream = false): FormData {
  const form = new FormData()
  form.append('text', params.text)
  form.append('language', params.language)
  if (params.speed != null && params.speed !== 1.0) form.append('speed', String(params.speed))
  if (params.refAudio) {
    form.append('ref_audio', params.refAudio)
    form.append('ref_text', params.refText ?? '')
    if (params.refVoiceName) form.append('ref_voice_name', params.refVoiceName)
  } else if (params.voiceId) {
    form.append('voice_id', params.voiceId)
  }
  if (params.instruct) form.append('instruct', params.instruct)
  if (stream) form.append('stream', 'true')
  return form
}

export async function synthesize(params: SynthesisParams): Promise<Blob> {
  const res = await fetch(`${BASE}/v1/synthesize`, { method: 'POST', body: buildSynthForm(params, false) })
  if (!res.ok) {
    const detail = await res.json().then((d) => d.detail).catch(() => res.statusText)
    throw new Error(detail)
  }
  return res.blob()
}

export async function* synthesizeStream(params: SynthesisParams, signal?: AbortSignal): AsyncGenerator<ArrayBuffer> {
  const res = await fetch(`${BASE}/v1/synthesize`, { method: 'POST', body: buildSynthForm(params, true), signal })
  if (!res.ok) {
    const detail = await res.json().then((d: { detail: string }) => d.detail).catch(() => res.statusText)
    throw new Error(detail)
  }

  const reader = res.body!.getReader()
  let pending = new Uint8Array(0)

  try {
    while (true) {
      const { done, value } = await reader.read()

      if (value) {
        const merged = new Uint8Array(pending.length + value.length)
        merged.set(pending)
        merged.set(value, pending.length)
        pending = merged
      }

      while (pending.length >= 4) {
        const size = new DataView(pending.buffer, pending.byteOffset).getUint32(0, false)
        if (pending.length < 4 + size) break
        const chunk = new Uint8Array(size)
        chunk.set(pending.subarray(4, 4 + size))
        yield chunk.buffer
        pending = pending.slice(4 + size)
      }

      if (done) break
    }
  } finally {
    reader.releaseLock()
  }
}

export function openSynthSocket(params: {
  language: string
  voiceId?: string
  speed?: number
  instruct?: string
}): WebSocket {
  const url = new URL(`${WS_BASE}/v1/ws/synthesize`)
  url.searchParams.set('language', params.language)
  if (params.voiceId) url.searchParams.set('voice_id', params.voiceId)
  if (params.speed != null && params.speed !== 1.0) url.searchParams.set('speed', String(params.speed))
  if (params.instruct) url.searchParams.set('instruct', params.instruct)
  return new WebSocket(url.toString())
}


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

export async function synthesize(params: SynthesisParams): Promise<Blob> {
  const form = new FormData()
  form.append('text', params.text)
  form.append('language', params.language)
  if (params.speed != null && params.speed !== 1.0) {
    form.append('speed', String(params.speed))
  }
  if (params.voiceId) {
    form.append('voice_id', params.voiceId)
  } else if (params.refAudio) {
    form.append('ref_audio', params.refAudio)
    if (params.refText) form.append('ref_text', params.refText)
  }

  const res = await fetch(`${BASE}/v1/synthesize`, { method: 'POST', body: form })
  if (!res.ok) {
    const detail = await res.json().then((d) => d.detail).catch(() => res.statusText)
    throw new Error(detail)
  }
  return res.blob()
}

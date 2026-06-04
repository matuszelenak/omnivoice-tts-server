import { combineWavBuffers } from './wav.js'

/** Gapless Web Audio playback of incrementally arriving WAV chunks. */
export class StreamingPlayer {
  /** True while scheduled audio is (or will be) audible. Reactive. */
  playing = $state(false)

  private ctx: AudioContext | null = null
  private nextPlayAt = 0
  private chunks: ArrayBuffer[] = []
  private sources: AudioBufferSourceNode[] = []

  /** Reset state and open a fresh AudioContext. Call before each session. */
  async init(): Promise<void> {
    this.ctx?.close()
    this.ctx = new AudioContext()
    if (this.ctx.state === 'suspended') await this.ctx.resume()
    this.nextPlayAt = this.ctx.currentTime + 0.1
    this.chunks = []
    this.sources = []
    this.playing = false
  }

  /** Decode a WAV chunk and schedule it right after the previously queued one. */
  async schedule(wavBuf: ArrayBuffer): Promise<void> {
    this.playing = true
    this.chunks.push(wavBuf.slice(0))
    const decoded = await this.ctx!.decodeAudioData(wavBuf)
    const src = this.ctx!.createBufferSource()
    src.buffer = decoded
    src.connect(this.ctx!.destination)
    const playAt = Math.max(this.nextPlayAt, this.ctx!.currentTime + 0.05)
    src.start(playAt)
    this.nextPlayAt = playAt + decoded.duration
    this.sources.push(src)
  }

  /** After the last chunk is scheduled, clear `playing` once it finishes. */
  trackEnd(): void {
    const last = this.sources.at(-1)
    if (!last) {
      this.playing = false
      return
    }
    last.addEventListener('ended', () => { this.playing = false }, { once: true })
  }

  /** Silence everything scheduled so far. */
  stop(): void {
    this.sources.forEach((s) => { try { s.stop() } catch { /* already stopped */ } })
    this.sources = []
    this.playing = false
  }

  get hasAudio(): boolean {
    return this.chunks.length > 0
  }

  /** Combined WAV of everything received this session. */
  toBlob(): Blob {
    return combineWavBuffers(this.chunks)
  }

  destroy(): void {
    this.ctx?.close()
  }
}

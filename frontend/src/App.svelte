<script lang="ts">
  import { onDestroy, onMount } from 'svelte'
  import type { Language, Voice } from './lib/types.js'
  import * as api from './lib/api.js'
  import LanguageSelect from './lib/LanguageSelect.svelte'
  import VoiceSelect from './lib/VoiceSelect.svelte'

  type ServerStatus = 'loading' | 'ready' | 'unavailable'

  let serverStatus = $state<ServerStatus>('loading')
  let languages = $state<Language[]>([])
  let voices = $state<Voice[]>([])
  let loadError = $state<string | null>(null)

  let language = $state('en')
  let selectedVoiceId = $state<string | null>(null)
  let useCustomVoice = $state(false)
  let customFile = $state<File | null>(null)
  let refText = $state('')
  let text = $state('')
  let speed = $state(1.0)

  let synthesizing = $state(false)
  let synthError = $state<string | null>(null)
  let resultUrl = $state<string | null>(null)
  let resultAudioEl: HTMLAudioElement | undefined = $state()

  let fileInput: HTMLInputElement | undefined = $state()

  let audioCtx: AudioContext | null = null
  let nextPlayAt = 0
  let rawChunks: ArrayBuffer[] = []
  let firstChunkDelay = $state<number | null>(null)
  let abortCtrl: AbortController | null = null
  let scheduledSources: AudioBufferSourceNode[] = []

  function combineWavBuffers(chunks: ArrayBuffer[]): Blob {
    if (chunks.length === 0) return new Blob([], { type: 'audio/wav' })
    if (chunks.length === 1) return new Blob([chunks[0]], { type: 'audio/wav' })

    const HEADER = 44
    const pcmParts = chunks.map((buf) => buf.slice(HEADER))
    const totalPcm = pcmParts.reduce((s, p) => s + p.byteLength, 0)

    const header = chunks[0].slice(0, HEADER)
    const view = new DataView(header)
    view.setUint32(4, 36 + totalPcm, true)
    view.setUint32(40, totalPcm, true)

    return new Blob([header, ...pcmParts], { type: 'audio/wav' })
  }

  onMount(async () => {
    try {
      const health = await api.checkHealth()
      serverStatus = health.modelLoaded ? 'ready' : 'unavailable'
    } catch {
      serverStatus = 'unavailable'
    }

    try {
      const [langs, vcs] = await Promise.all([api.fetchLanguages(), api.fetchVoices()])
      languages = langs
      voices = vcs
    } catch (e) {
      loadError = e instanceof Error ? e.message : 'Failed to load data'
    }
  })

  onDestroy(() => {
    abortCtrl?.abort()
    audioCtx?.close()
    if (resultUrl) URL.revokeObjectURL(resultUrl)
  })

  function handleStop() {
    abortCtrl?.abort()
    scheduledSources.forEach((s) => { try { s.stop() } catch { /* already stopped */ } })
    scheduledSources = []
  }

  async function handleSynthesize() {
    if (!text.trim() || synthesizing) return
    synthesizing = true
    synthError = null
    rawChunks = []
    scheduledSources = []
    firstChunkDelay = null

    if (resultUrl) {
      URL.revokeObjectURL(resultUrl)
      resultUrl = null
    }

    audioCtx?.close()
    audioCtx = new AudioContext()
    if (audioCtx.state === 'suspended') await audioCtx.resume()
    nextPlayAt = audioCtx.currentTime + 0.1

    abortCtrl = new AbortController()
    const startTime = Date.now()
    let firstChunk = true

    try {
      for await (const wavBuf of api.synthesizeStream({
        text: text.trim(),
        language,
        speed: speed !== 1.0 ? speed : undefined,
        voiceId: !useCustomVoice && selectedVoiceId ? selectedVoiceId : undefined,
        refAudio: useCustomVoice && customFile ? customFile : undefined,
        refText: useCustomVoice && refText.trim() ? refText.trim() : undefined,
      }, abortCtrl.signal)) {
        if (firstChunk) {
          firstChunkDelay = Date.now() - startTime
          firstChunk = false
        }
        rawChunks.push(wavBuf.slice(0))

        const decoded = await audioCtx.decodeAudioData(wavBuf)
        const src = audioCtx.createBufferSource()
        src.buffer = decoded
        src.connect(audioCtx.destination)
        const playAt = Math.max(nextPlayAt, audioCtx.currentTime + 0.05)
        src.start(playAt)
        nextPlayAt = playAt + decoded.duration
        scheduledSources.push(src)
      }
    } catch (e) {
      if (e instanceof Error && e.name !== 'AbortError') {
        synthError = e instanceof Error ? e.message : 'Synthesis failed'
      }
    } finally {
      synthesizing = false
    }

    if (rawChunks.length > 0) {
      resultUrl = URL.createObjectURL(combineWavBuffers(rawChunks))
    }
  }

  function handleFileChange(e: Event) {
    const input = e.target as HTMLInputElement
    customFile = input.files?.[0] ?? null
  }

  function handleDownload() {
    if (!resultUrl) return
    const a = document.createElement('a')
    a.href = resultUrl
    a.download = 'output.wav'
    a.click()
  }

  let speedLabel = $derived(speed === 1.0 ? '1.0× (normal)' : `${speed.toFixed(1)}×`)

  let canSynthesize = $derived(
    text.trim().length > 0 &&
      !synthesizing &&
      (!useCustomVoice || customFile !== null),
  )
</script>

<div class="app">
  <header class="header">
    <div class="header-inner">
      <h1 class="logo">OmniVoice <span class="logo-sub">TTS</span></h1>
      <div
        class="status-badge"
        class:ready={serverStatus === 'ready'}
        class:unavailable={serverStatus === 'unavailable'}
        class:loading={serverStatus === 'loading'}
      >
        <span class="status-dot"></span>
        {#if serverStatus === 'loading'}
          Connecting…
        {:else if serverStatus === 'ready'}
          Model ready
        {:else}
          Server unavailable
        {/if}
      </div>
    </div>
  </header>

  <main class="main">
    {#if loadError}
      <div class="banner banner-warning">⚠ {loadError} — running with limited functionality</div>
    {/if}

    {#if serverStatus === 'unavailable'}
      <div class="banner banner-error">
        ⚠ Cannot reach the TTS server at localhost:9001. Synthesis will fail until it is running.
      </div>
    {/if}

    <div class="card">
      <label class="field-label" for="lang-select">Language</label>
      <LanguageSelect {languages} bind:value={language} />
    </div>

    <div class="card">
      <div class="section-header">
        <span class="field-label">Reference Voice</span>
        <label class="toggle-label">
          <input type="checkbox" bind:checked={useCustomVoice} />
          <span>Use custom audio</span>
        </label>
      </div>

      {#if !useCustomVoice}
        <VoiceSelect
          {voices}
          bind:value={selectedVoiceId}
          ondelete={(id) => {
            voices = voices.filter((v) => v.id !== id)
            if (selectedVoiceId === id) selectedVoiceId = null
          }}
        />
        {#if voices.length === 0}
          <p class="hint">
            No voice samples found. Place <code>.wav</code> + <code>.txt</code> pairs in
            <code>server/voices/</code> or set <code>VOICE_SAMPLES_DIR</code>.
          </p>
        {/if}
      {:else}
        <div class="custom-voice">
          <div class="file-row">
            <button
              class="file-btn"
              type="button"
              onclick={() => fileInput?.click()}
            >
              {customFile ? '✓ ' + customFile.name : 'Choose audio file…'}
            </button>
            <input
              bind:this={fileInput}
              type="file"
              accept="audio/*,.wav,.mp3,.flac,.ogg"
              style="display:none"
              onchange={handleFileChange}
            />
            {#if customFile}
              <button
                class="clear-btn"
                type="button"
                onclick={() => {
                  customFile = null
                  if (fileInput) fileInput.value = ''
                }}
                title="Remove file"
              >✕</button>
            {/if}
          </div>
          <label class="field-label" for="ref-text" style="margin-top:0.75rem">
            Reference transcript
            <span class="hint-inline">(auto-transcribed by Whisper if left blank)</span>
          </label>
          <textarea
            id="ref-text"
            class="textarea"
            rows="2"
            placeholder="Optionally type what the audio says…"
            bind:value={refText}
          ></textarea>
        </div>
      {/if}
    </div>

    <div class="card">
      <label class="field-label" for="text-input">Text to synthesize</label>
      <textarea
        id="text-input"
        class="textarea text-input"
        rows="5"
        placeholder="Type or paste the text you want to convert to speech…"
        bind:value={text}
      ></textarea>
    </div>

    <div class="card speed-card">
      <div class="speed-row">
        <label class="field-label" for="speed-range">Speed</label>
        <span class="speed-value">{speedLabel}</span>
      </div>
      <input
        id="speed-range"
        type="range"
        min="0.5"
        max="2.0"
        step="0.1"
        bind:value={speed}
        class="range-input"
      />
      <div class="range-marks">
        <span>0.5×</span>
        <span>1.0×</span>
        <span>2.0×</span>
      </div>
    </div>

    <button
      class="synth-btn"
      class:stop={synthesizing}
      type="button"
      disabled={!synthesizing && !canSynthesize}
      onclick={synthesizing ? handleStop : handleSynthesize}
    >
      {#if synthesizing}
        <span class="spinner"></span> Synthesizing… <span class="stop-label">■ Stop</span>
      {:else}
        ▶&nbsp; Synthesize
      {/if}
    </button>
    {#if firstChunkDelay !== null}
      <p class="stream-hint">First audio chunk received in {(firstChunkDelay / 1000).toFixed(1)} s</p>
    {/if}

    {#if synthError}
      <div class="banner banner-error">✕ {synthError}</div>
    {/if}

    {#if resultUrl}
      <div class="card result-card">
        <div class="result-header">
          <span class="field-label">Replay / Download</span>
          <button class="download-btn" type="button" onclick={handleDownload} title="Download WAV">
            ↓ Download
          </button>
        </div>
        <!-- svelte-ignore a11y_media_has_caption -->
        <audio
          bind:this={resultAudioEl}
          src={resultUrl}
          controls
          class="audio-player"
        ></audio>
      </div>
    {/if}
  </main>
</div>

<style>
  .app {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    background: var(--bg);
  }

  .header {
    background: var(--surface-1);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 10;
  }

  .header-inner {
    max-width: 680px;
    margin: 0 auto;
    padding: 0.875rem 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .logo {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
  }

  .logo-sub {
    color: var(--primary);
  }

  .status-badge {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--text-muted);
  }

  .status-badge.ready {
    border-color: var(--success);
    color: var(--success);
  }

  .status-badge.unavailable {
    border-color: var(--error);
    color: var(--error);
  }

  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: currentColor;
  }

  .status-badge.ready .status-dot {
    animation: blink 2s ease-in-out infinite;
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .main {
    max-width: 680px;
    width: 100%;
    margin: 0 auto;
    padding: 1.5rem 1.25rem 3rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .banner {
    padding: 0.75rem 1rem;
    border-radius: var(--radius);
    font-size: 0.85rem;
    border: 1px solid;
  }

  .banner-warning {
    background: color-mix(in srgb, var(--warning) 10%, transparent);
    border-color: var(--warning);
    color: var(--warning);
  }

  .banner-error {
    background: color-mix(in srgb, var(--error) 10%, transparent);
    border-color: var(--error);
    color: var(--error);
  }

  .card {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }

  .field-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
  }

  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .toggle-label {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.82rem;
    color: var(--text-muted);
    cursor: pointer;
    user-select: none;
  }

  .toggle-label input[type='checkbox'] {
    accent-color: var(--primary);
    cursor: pointer;
  }

  .hint {
    font-size: 0.8rem;
    color: var(--text-muted);
    line-height: 1.5;
  }

  .hint-inline {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
    margin-left: 0.35rem;
  }

  code {
    font-family: monospace;
    font-size: 0.85em;
    background: var(--surface-3);
    padding: 0.1em 0.35em;
    border-radius: 4px;
  }

  .custom-voice {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .file-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .file-btn {
    flex: 1;
    padding: 0.6rem 0.875rem;
    background: var(--surface-2);
    border: 1px dashed var(--border);
    border-radius: 8px;
    color: var(--text-muted);
    font-size: 0.875rem;
    text-align: left;
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }

  .file-btn:hover {
    border-color: var(--primary);
    color: var(--text);
  }

  .clear-btn {
    padding: 0.5rem 0.6rem;
    background: var(--surface-3);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-muted);
    font-size: 0.75rem;
    cursor: pointer;
    flex-shrink: 0;
    transition: color 0.15s;
  }

  .clear-btn:hover {
    color: var(--error);
  }

  .textarea {
    width: 100%;
    padding: 0.6rem 0.875rem;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-size: 0.9rem;
    resize: vertical;
    transition: border-color 0.15s;
    line-height: 1.55;
  }

  .textarea:focus {
    outline: none;
    border-color: var(--primary);
  }

  .text-input {
    min-height: 120px;
  }

  .speed-card {
    gap: 0.4rem;
  }

  .speed-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .speed-value {
    font-size: 0.875rem;
    color: var(--primary);
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  .range-input {
    width: 100%;
    accent-color: var(--primary);
    cursor: pointer;
    height: 4px;
  }

  .range-marks {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: -0.1rem;
  }

  .synth-btn {
    padding: 0.85rem 2rem;
    background: var(--primary);
    border: none;
    border-radius: var(--radius);
    color: #fff;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s, opacity 0.15s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    align-self: stretch;
  }

  .synth-btn:hover:not(:disabled) {
    background: var(--primary-hover);
  }

  .synth-btn.stop:hover:not(:disabled) {
    background: var(--error);
  }

  .synth-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .stop-label {
    margin-left: 0.5rem;
    font-size: 0.8rem;
    opacity: 0.75;
  }

  .stream-hint {
    text-align: center;
    font-size: 0.75rem;
    color: var(--text-muted);
    margin: -0.25rem 0 0;
  }

  .spinner {
    width: 18px;
    height: 18px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    flex-shrink: 0;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .result-card {
    gap: 0.75rem;
  }

  .result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .download-btn {
    padding: 0.35rem 0.75rem;
    background: var(--surface-3);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-muted);
    font-size: 0.8rem;
    cursor: pointer;
    transition: color 0.15s, border-color 0.15s;
  }

  .download-btn:hover {
    color: var(--text);
    border-color: var(--primary);
  }

  .audio-player {
    width: 100%;
    border-radius: 6px;
    outline: none;
    filter: invert(1) hue-rotate(220deg) brightness(0.85);
  }
</style>

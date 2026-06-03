<script lang="ts">
  import { onDestroy, onMount } from 'svelte'
  import { get_encoding } from 'tiktoken'
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
  let refVoiceName = $state('')
  let text = $state('')
  let speed = $state(1.0)

  type VoiceTab = 'cloning' | 'design'
  let activeTab = $state<VoiceTab>('cloning')

  let designGender = $state('')
  let designAge = $state('')
  let designPitch = $state('')
  let designStyle = $state('')
  let designAccent = $state('')
  let instruct = $derived(
    [designGender, designAge, designPitch, designStyle, designAccent].filter(Boolean).join(', ')
  )

  let streamMode = $state(true)
  let llmSimMode = $state(false)
  let tokensPerSecond = $state(30)

  let synthesizing = $state(false)
  let synthError = $state<string | null>(null)
  let resultUrl = $state<string | null>(null)
  let resultAudioEl: HTMLAudioElement | undefined = $state()

  let fileInput: HTMLInputElement | undefined = $state()

  let audioCtx: AudioContext | null = null
  let nextPlayAt = 0
  let rawChunks: ArrayBuffer[] = []
  let firstChunkDelay = $state<number | null>(null)
  let totalDuration = $state<number | null>(null)
  let playingBack = $state(false)
  let abortCtrl: AbortController | null = null
  let scheduledSources: AudioBufferSourceNode[] = []
  let activeWs: WebSocket | null = null
  let simSentChars = $state(0)

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
      if (vcs.length > 0 && selectedVoiceId === null) selectedVoiceId = vcs[0].id
    } catch (e) {
      loadError = e instanceof Error ? e.message : 'Failed to load data'
    }
  })

  onDestroy(() => {
    abortCtrl?.abort()
    activeWs?.close()
    audioCtx?.close()
    if (resultUrl) URL.revokeObjectURL(resultUrl)
  })

  function handleStop() {
    abortCtrl?.abort()
    activeWs?.close()
    activeWs = null
    scheduledSources.forEach((s) => { try { s.stop() } catch { /* already stopped */ } })
    scheduledSources = []
    playingBack = false
  }

  async function initAudio() {
    audioCtx?.close()
    audioCtx = new AudioContext()
    if (audioCtx.state === 'suspended') await audioCtx.resume()
    nextPlayAt = audioCtx.currentTime + 0.1
    rawChunks = []
    scheduledSources = []
    playingBack = false
  }

  async function scheduleChunk(wavBuf: ArrayBuffer) {
    playingBack = true
    rawChunks.push(wavBuf.slice(0))
    const decoded = await audioCtx!.decodeAudioData(wavBuf)
    const src = audioCtx!.createBufferSource()
    src.buffer = decoded
    src.connect(audioCtx!.destination)
    const playAt = Math.max(nextPlayAt, audioCtx!.currentTime + 0.05)
    src.start(playAt)
    nextPlayAt = playAt + decoded.duration
    scheduledSources.push(src)
  }

  function trackPlaybackEnd() {
    const last = scheduledSources.at(-1)
    if (!last) { playingBack = false; return }
    last.addEventListener('ended', () => { playingBack = false }, { once: true })
  }

  async function handleLLMSim() {
    simSentChars = 0
    await initAudio()

    const ws = api.openSynthSocket({
      language,
      voiceId: activeTab === 'cloning' && !useCustomVoice && selectedVoiceId ? selectedVoiceId : undefined,
      speed: speed !== 1.0 ? speed : undefined,
      instruct: activeTab === 'design' && instruct ? instruct : undefined,
    })
    ws.binaryType = 'arraybuffer'
    activeWs = ws

    const startTime = Date.now()
    let firstChunk = true

    const allAudioReceived = new Promise<void>((resolve) => {
      ws.onmessage = async (event: MessageEvent<ArrayBuffer>) => {
        if (firstChunk) { firstChunkDelay = Date.now() - startTime; firstChunk = false }
        await scheduleChunk(event.data)
      }
      ws.onclose = () => resolve()
    })

    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => resolve()
      ws.onerror = () => reject(new Error('WebSocket connection failed'))
    })

    const enc = get_encoding('cl100k_base')
    const tokenIds = enc.encode(text.trim())
    const utf8 = new TextDecoder('utf-8', { fatal: false })
    const msPerToken = 1000 / tokensPerSecond

    for (const id of tokenIds) {
      if (ws.readyState !== WebSocket.OPEN) break
      const tokenText = utf8.decode(enc.decode_single_token_bytes(id))
      ws.send(tokenText)
      simSentChars += tokenText.length
      await new Promise((r) => setTimeout(r, msPerToken))
    }

    enc.free()

    if (ws.readyState === WebSocket.OPEN) {
      ws.send('')
      await allAudioReceived
      totalDuration = Date.now() - startTime
      trackPlaybackEnd()
    }

    activeWs = null

    if (rawChunks.length > 0) {
      resultUrl = URL.createObjectURL(combineWavBuffers(rawChunks))
    }
  }

  function synthParams() {
    const cloning = activeTab === 'cloning'
    return {
      text: text.trim(),
      language,
      speed: speed !== 1.0 ? speed : undefined,
      voiceId: cloning && !useCustomVoice && selectedVoiceId ? selectedVoiceId : undefined,
      refAudio: cloning && useCustomVoice && customFile ? customFile : undefined,
      refText: cloning && useCustomVoice ? refText.trim() : undefined,
      refVoiceName: cloning && useCustomVoice && refVoiceName.trim() ? refVoiceName.trim() : undefined,
      instruct: activeTab === 'design' && instruct ? instruct : undefined,
    }
  }

  async function handleSynthesize() {
    if (!text.trim() || synthesizing) return
    synthesizing = true
    synthError = null
    firstChunkDelay = null
    totalDuration = null

    if (resultUrl) {
      URL.revokeObjectURL(resultUrl)
      resultUrl = null
    }

    try {
      if (llmSimMode) {
        await handleLLMSim()
      } else if (streamMode) {
        await initAudio()
        abortCtrl = new AbortController()
        const startTime = Date.now()
        let firstChunk = true

        try {
          for await (const wavBuf of api.synthesizeStream(synthParams(), abortCtrl.signal)) {
            if (firstChunk) { firstChunkDelay = Date.now() - startTime; firstChunk = false }
            await scheduleChunk(wavBuf)
          }
          totalDuration = Date.now() - startTime
          trackPlaybackEnd()
        } catch (e) {
          if (e instanceof Error && e.name !== 'AbortError') {
            synthError = e instanceof Error ? e.message : 'Synthesis failed'
          }
        }

        if (rawChunks.length > 0) {
          resultUrl = URL.createObjectURL(combineWavBuffers(rawChunks))
        }
      } else {
        const startTime = Date.now()
        const blob = await api.synthesize(synthParams())
        totalDuration = Date.now() - startTime
        resultUrl = URL.createObjectURL(blob)
      }
    } catch (e) {
      synthError = e instanceof Error ? e.message : 'Synthesis failed'
    } finally {
      synthesizing = false
    }

    if (activeTab === 'cloning' && useCustomVoice && refVoiceName.trim()) {
      try {
        voices = await api.fetchVoices()
      } catch { /* non-critical */ }
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
      !(llmSimMode && activeTab === 'cloning' && useCustomVoice) &&
      (activeTab === 'design' ||
        selectedVoiceId !== null ||
        (useCustomVoice && customFile !== null && refText.trim().length > 0)),
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

    <div class="card voice-card">
      <div class="tab-bar">
        <button class="tab-btn" class:active={activeTab === 'cloning'} onclick={() => activeTab = 'cloning'}>
          Voice Cloning
        </button>
        <button class="tab-btn" class:active={activeTab === 'design'} onclick={() => activeTab = 'design'}>
          Voice Design
        </button>
      </div>

      <div class="tab-panels">
        <div class="tab-panel" class:active={activeTab === 'cloning'}>
          {#if !useCustomVoice}
            <div class="voice-scroll">
              <VoiceSelect
                {voices}
                bind:value={selectedVoiceId}
                ondelete={(id) => {
                  const remaining = voices.filter((v) => v.id !== id)
                  voices = remaining
                  if (selectedVoiceId === id) selectedVoiceId = remaining[0]?.id ?? null
                }}
              />
              {#if voices.length === 0}
                <p class="hint">
                  No voice samples found. Place <code>.wav</code> + <code>.txt</code> pairs in
                  <code>server/voices/</code> or set <code>VOICE_SAMPLES_DIR</code>.
                </p>
              {/if}
            </div>
            <button
              class="use-custom-btn"
              type="button"
              disabled={llmSimMode}
              onclick={() => useCustomVoice = true}
            >+ Use custom audio</button>
          {:else}
            <div class="custom-voice">
              <div class="custom-voice-header">
                <span class="field-label">Custom audio</span>
                <button class="back-btn" type="button" onclick={() => useCustomVoice = false}>
                  ← Built-in voices
                </button>
              </div>
              <label class="field-label">
                Reference audio <span class="required">*</span>
              </label>
              <div class="file-row">
                <button class="file-btn" type="button" onclick={() => fileInput?.click()}>
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
                    onclick={() => { customFile = null; if (fileInput) fileInput.value = '' }}
                    title="Remove file"
                  >✕</button>
                {/if}
              </div>
              <label class="field-label" for="ref-text" style="margin-top:0.75rem">
                Reference transcript <span class="required">*</span>
              </label>
              <textarea
                id="ref-text"
                class="textarea"
                rows="2"
                placeholder="Type exactly what the audio says…"
                bind:value={refText}
              ></textarea>
              <label class="field-label" for="ref-voice-name" style="margin-top:0.75rem">
                Save as voice
                <span class="hint-inline">(optional — letters, digits, hyphens, underscores)</span>
              </label>
              <input
                id="ref-voice-name"
                class="text-input-sm"
                type="text"
                placeholder="e.g. my_voice"
                bind:value={refVoiceName}
              />
            </div>
          {/if}
        </div>

        <div class="tab-panel" class:active={activeTab === 'design'}>
          <div class="design-grid">
            <div class="design-field">
              <label class="field-label" for="d-gender">Gender</label>
              <select id="d-gender" class="design-select" bind:value={designGender}>
                <option value="">— any —</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>
            <div class="design-field">
              <label class="field-label" for="d-age">Age</label>
              <select id="d-age" class="design-select" bind:value={designAge}>
                <option value="">— any —</option>
                <option value="child">Child</option>
                <option value="teenager">Teenager</option>
                <option value="young adult">Young Adult</option>
                <option value="middle-aged">Middle-aged</option>
                <option value="elderly">Elderly</option>
              </select>
            </div>
            <div class="design-field">
              <label class="field-label" for="d-pitch">Pitch</label>
              <select id="d-pitch" class="design-select" bind:value={designPitch}>
                <option value="">— any —</option>
                <option value="very low pitch">Very Low</option>
                <option value="low pitch">Low</option>
                <option value="moderate pitch">Moderate</option>
                <option value="high pitch">High</option>
                <option value="very high pitch">Very High</option>
              </select>
            </div>
            <div class="design-field">
              <label class="field-label" for="d-style">Style</label>
              <select id="d-style" class="design-select" bind:value={designStyle}>
                <option value="">— none —</option>
                <option value="whisper">Whisper</option>
              </select>
            </div>
            <div class="design-field design-field-full">
              <label class="field-label" for="d-accent">English Accent</label>
              <select id="d-accent" class="design-select" bind:value={designAccent}>
                <option value="">— any —</option>
                <option value="american accent">American</option>
                <option value="british accent">British</option>
                <option value="australian accent">Australian</option>
                <option value="canadian accent">Canadian</option>
                <option value="indian accent">Indian</option>
                <option value="korean accent">Korean</option>
                <option value="japanese accent">Japanese</option>
                <option value="portuguese accent">Portuguese</option>
                <option value="russian accent">Russian</option>
              </select>
            </div>
          </div>
          {#if instruct}
            <p class="instruct-preview">"{instruct}"</p>
          {:else}
            <p class="hint">All fields optional — leave empty to let the model decide.</p>
          {/if}
        </div>
      </div>
    </div>

    <div class="card">
      <label class="field-label" for="text-input">Text to synthesize</label>
      {#if llmSimMode && synthesizing}
        <div class="textarea text-input sim-display" aria-live="polite">
          <span class="sim-sent">{text.slice(0, simSentChars)}</span><span class="sim-pending">{text.slice(simSentChars)}</span>
        </div>
      {:else}
        <textarea
          id="text-input"
          class="textarea text-input"
          rows="9"
          placeholder="Type or paste the text you want to convert to speech…"
          bind:value={text}
        ></textarea>
      {/if}
      <p class="expression-hint">
        <span class="expression-hint-label">Expression tags</span>
        {#each ['laughter','sigh','confirmation-en','question-en','question-ah','question-oh','question-ei','question-yi','surprise-ah','surprise-oh','surprise-wa','surprise-yo','dissatisfaction-hnn'] as tag}
          <code class="tag-chip">[{tag}]</code>
        {/each}
      </p>
    </div>

    <div class="card settings-card">
      <div class="settings-row">
        <div class="settings-col">
          <label class="field-label" for="lang-select">Language</label>
          <LanguageSelect {languages} bind:value={language} />
        </div>
        <div class="settings-col">
          <div class="settings-label-row">
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
            <span>2.0×</span>
          </div>
        </div>
      </div>

      <div class="settings-col">
        <span class="field-label">Output mode</span>
        <div class="output-btn-group" role="group" aria-label="Output mode">
          <button
            class="output-btn"
            class:active={!streamMode && !llmSimMode}
            type="button"
            onclick={() => { streamMode = false; llmSimMode = false }}
          >⬇ Full audio</button>
          <button
            class="output-btn"
            class:active={streamMode || llmSimMode}
            type="button"
            onclick={() => { streamMode = true }}
          >▶ Per-sentence streaming</button>
        </div>
      </div>

      {#if streamMode || llmSimMode}
        <label class="llm-toggle" class:disabled={activeTab === 'cloning' && useCustomVoice}>
          <input
            type="checkbox"
            bind:checked={llmSimMode}
            disabled={activeTab === 'cloning' && useCustomVoice}
            style="display:none"
          />
          <div class="toggle-track" class:on={llmSimMode}>
            <div class="toggle-knob"></div>
          </div>
          <div class="toggle-text">
            <span>Simulate LLM</span>
            <span class="hint-inline">send text token-by-token over WebSocket</span>
          </div>
        </label>

        {#if llmSimMode}
          <div class="settings-col">
            <div class="settings-label-row">
              <label class="field-label" for="tps-range">Token rate</label>
              <span class="speed-value">{tokensPerSecond} tok/s</span>
            </div>
            <input
              id="tps-range"
              type="range"
              min="1"
              max="150"
              step="1"
              bind:value={tokensPerSecond}
              class="range-input"
            />
            <div class="range-marks">
              <span>1</span>
              <span>75</span>
              <span>150 tok/s</span>
            </div>
          </div>
        {/if}
      {/if}
    </div>

    <button
      class="synth-btn"
      class:stop={synthesizing && (streamMode || llmSimMode) || playingBack}
      type="button"
      disabled={!synthesizing && !playingBack && !canSynthesize}
      onclick={synthesizing && (streamMode || llmSimMode) || playingBack ? handleStop : handleSynthesize}
    >
      {#if synthesizing}
        <span class="spinner"></span> Synthesizing…
        {#if streamMode || llmSimMode}<span class="stop-label">■ Stop</span>{/if}
      {:else if playingBack}
        ■&nbsp; Stop playback
      {:else}
        ▶&nbsp; Synthesize
      {/if}
    </button>
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

    {#if totalDuration !== null || firstChunkDelay !== null}
      <p class="stream-hint">
        {#if firstChunkDelay !== null}
          First chunk: {(firstChunkDelay / 1000).toFixed(2)} s
          {#if totalDuration !== null}
            &nbsp;·&nbsp; Total: {(totalDuration / 1000).toFixed(2)} s
          {/if}
        {:else if totalDuration !== null}
          Received in {(totalDuration / 1000).toFixed(2)} s
        {/if}
      </p>
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
    max-width: 800px;
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
    max-width: 800px;
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

  .voice-scroll {
    max-height: 260px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }

  .use-custom-btn {
    align-self: center;
    padding: 0.45rem 1.1rem;
    background: none;
    border: 1px dashed var(--border);
    border-radius: 6px;
    color: var(--text-muted);
    font-size: 0.8rem;
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
  }

  .use-custom-btn:hover:not(:disabled) {
    border-color: var(--primary);
    color: var(--primary);
  }

  .use-custom-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .custom-voice-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.25rem;
  }

  .back-btn {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 0.78rem;
    cursor: pointer;
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    transition: color 0.15s, background 0.15s;
  }

  .back-btn:hover {
    color: var(--text);
    background: var(--surface-3);
  }

  .hint {
    font-size: 0.8rem;
    color: var(--text-muted);
    line-height: 1.5;
  }

  .expression-hint {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.3rem;
    margin: 0;
  }

  .expression-hint-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-right: 0.15rem;
    white-space: nowrap;
  }

  .tag-chip {
    font-family: monospace;
    font-size: 0.72rem;
    background: var(--surface-3);
    color: var(--text-muted);
    padding: 0.15em 0.45em;
    border-radius: 4px;
    border: 1px solid var(--border);
    white-space: nowrap;
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

  .voice-card {
    gap: 0;
    padding: 0;
    overflow: hidden;
  }

  .tab-bar {
    display: flex;
    border-bottom: 1px solid var(--border);
  }

  .tab-btn {
    flex: 1;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    border: none;
    background: none;
    color: var(--text-muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
  }

  .tab-btn.active {
    color: var(--primary);
    border-bottom-color: var(--primary);
    background: color-mix(in srgb, var(--primary) 5%, transparent);
  }

  .tab-btn:hover:not(.active) {
    color: var(--text);
    background: var(--surface-2);
  }

  .tab-panels {
    display: grid;
    padding: 1.1rem 1.25rem;
  }

  .tab-panel {
    grid-area: 1 / 1;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }

  .tab-panel:not(.active) {
    visibility: hidden;
    pointer-events: none;
  }

  .design-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
  }

  .design-field {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }

  .design-field-full {
    grid-column: 1 / -1;
  }

  .design-select {
    width: 100%;
    padding: 0.5rem 0.75rem;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-size: 0.875rem;
    cursor: pointer;
    transition: border-color 0.15s;
  }

  .design-select:focus {
    outline: none;
    border-color: var(--primary);
  }

  .instruct-preview {
    font-family: monospace;
    font-size: 0.8rem;
    color: var(--primary);
    background: color-mix(in srgb, var(--primary) 8%, transparent);
    border: 1px solid color-mix(in srgb, var(--primary) 25%, transparent);
    border-radius: 6px;
    padding: 0.4rem 0.75rem;
    margin-top: 0.25rem;
    word-break: break-all;
  }

  .custom-voice {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .required {
    color: var(--error);
  }

  .text-input-sm {
    width: 100%;
    padding: 0.5rem 0.875rem;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-size: 0.9rem;
    transition: border-color 0.15s;
  }

  .text-input-sm:focus {
    outline: none;
    border-color: var(--primary);
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
    min-height: 200px;
  }

  .settings-card {
    gap: 0.75rem;
  }

  .settings-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    align-items: start;
  }

  .settings-col {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .settings-label-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .output-btn-group {
    display: flex;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }

  .output-btn {
    flex: 1;
    padding: 0.5rem 0.75rem;
    background: var(--surface-2);
    border: none;
    border-right: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.82rem;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
  }

  .output-btn:last-child {
    border-right: none;
  }

  .output-btn:hover:not(.active) {
    background: var(--surface-3);
    color: var(--text);
  }

  .output-btn.active {
    background: color-mix(in srgb, var(--primary) 15%, var(--surface-2));
    color: var(--primary);
    font-weight: 600;
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

  .llm-toggle {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    cursor: pointer;
    user-select: none;
    padding: 0.15rem 0;
  }

  .llm-toggle.disabled {
    opacity: 0.4;
    cursor: not-allowed;
    pointer-events: none;
  }

  .toggle-track {
    position: relative;
    width: 38px;
    height: 22px;
    background: var(--border);
    border-radius: 11px;
    transition: background 0.2s;
    flex-shrink: 0;
  }

  .toggle-track.on {
    background: var(--primary);
  }

  .toggle-knob {
    position: absolute;
    top: 3px;
    left: 3px;
    width: 16px;
    height: 16px;
    background: #fff;
    border-radius: 50%;
    transition: transform 0.2s;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
  }

  .toggle-track.on .toggle-knob {
    transform: translateX(16px);
  }

  .toggle-text {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.3;
  }

  .toggle-text span:first-child {
    font-weight: 600;
    color: var(--text);
  }

  .sim-display {
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    cursor: default;
    min-height: 120px;
  }

  .sim-sent {
    color: var(--text-muted);
  }

  .sim-pending {
    color: var(--text);
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

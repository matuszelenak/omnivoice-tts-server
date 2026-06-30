<script lang="ts">
  import { onDestroy, onMount } from 'svelte'
  import type { Language, Voice } from './lib/types.js'
  import * as api from './lib/api.js'
  import { StreamingPlayer } from './lib/player.svelte.js'
  import { streamTokens } from './lib/llm.js'
  import LanguageSelect from './lib/LanguageSelect.svelte'
  import VoiceSelect from './lib/VoiceSelect.svelte'

  // ── Server / data state ──────────────────────────────────────────────

  type ServerStatus = 'loading' | 'ready' | 'unavailable'

  let serverStatus = $state<ServerStatus>('loading')
  let languages = $state<Language[]>([])
  let voices = $state<Voice[]>([])
  let loadError = $state<string | null>(null)

  // ── Synthesis inputs ─────────────────────────────────────────────────

  let language = $state('en')
  let text = $state('')
  let speed = $state(1.0)
  let totalSteps = $state(16)
  let selectedVoiceId = $state<string | null>(null)

  // Import panel
  let showImport = $state(false)
  let importName = $state('')
  let importFile = $state<File | null>(null)
  let importFileInput: HTMLInputElement | undefined = $state()
  let importing = $state(false)
  let importError = $state<string | null>(null)

  // Output modes
  let streamMode = $state(true)
  let llmSimMode = $state(false)
  let tokensPerSecond = $state(30)

  // ── Synthesis session state ──────────────────────────────────────────

  const player = new StreamingPlayer()
  let synthesizing = $state(false)
  let synthError = $state<string | null>(null)
  let resultUrl = $state<string | null>(null)
  let firstChunkDelay = $state<number | null>(null)
  let totalDuration = $state<number | null>(null)
  let simSentChars = $state(0)
  let abortCtrl: AbortController | null = null
  let activeWs: WebSocket | null = null

  // ── Derived view state ───────────────────────────────────────────────

  let streaming = $derived(streamMode || llmSimMode)
  let showStop = $derived((synthesizing && streaming) || player.playing)
  let speedLabel = $derived(speed === 1.0 ? '1.0× (normal)' : `${speed.toFixed(1)}×`)
  let stepsLabel = $derived(`${totalSteps} step${totalSteps === 1 ? '' : 's'}`)

  let canSynthesize = $derived(
    text.trim().length > 0 &&
      selectedVoiceId !== null &&
      !synthesizing,
  )

  // ── Lifecycle ────────────────────────────────────────────────────────

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
      if (vcs.length > 0 && selectedVoiceId === null) {
        selectedVoiceId = vcs[0].id
      }
    } catch (e) {
      loadError = e instanceof Error ? e.message : 'Failed to load data'
    }
  })

  onDestroy(() => {
    abortCtrl?.abort()
    activeWs?.close()
    player.destroy()
    if (resultUrl) URL.revokeObjectURL(resultUrl)
  })

  // ── Helpers ──────────────────────────────────────────────────────────

  function handleVoiceDelete(id: string) {
    voices = voices.filter((v) => v.id !== id)
    if (selectedVoiceId === id) {
      selectedVoiceId = voices[0]?.id ?? null
    }
  }

  function handleImportFileChange(e: Event) {
    const input = e.target as HTMLInputElement
    const f = input.files?.[0] ?? null
    importFile = f
    if (f && !importName) {
      importName = f.name.replace(/\.json$/i, '')
    }
  }

  async function handleImport() {
    if (!importFile || !importName.trim()) return
    importing = true
    importError = null
    try {
      const voice = await api.importVoice(importName.trim(), importFile)
      voices = [...voices, voice]
      selectedVoiceId = voice.id
      showImport = false
      importName = ''
      importFile = null
      if (importFileInput) importFileInput.value = ''
    } catch (e) {
      importError = e instanceof Error ? e.message : 'Import failed'
    } finally {
      importing = false
    }
  }

  function synthParams() {
    return {
      text: text.trim(),
      language,
      speed: speed === 1.0 ? undefined : speed,
      totalSteps,
      voiceId: selectedVoiceId!,
    }
  }

  /** Track first-chunk latency and total duration of a synthesis session. */
  function startTimer() {
    const start = Date.now()
    let first = true
    return {
      chunk() { if (first) { firstChunkDelay = Date.now() - start; first = false } },
      total() { totalDuration = Date.now() - start },
    }
  }

  // ── Synthesis flows ──────────────────────────────────────────────────

  async function runLLMSim() {
    simSentChars = 0
    await player.init()

    const params = synthParams()
    const ws = api.openSynthSocket({
      language: params.language,
      voiceId: params.voiceId,
      speed: params.speed,
      totalSteps: params.totalSteps,
    })
    ws.binaryType = 'arraybuffer'
    activeWs = ws

    const timer = startTimer()
    const allAudioReceived = new Promise<void>((resolve) => {
      ws.onmessage = async (event: MessageEvent<ArrayBuffer>) => {
        timer.chunk()
        await player.schedule(event.data)
      }
      ws.onclose = () => resolve()
    })

    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => resolve()
      ws.onerror = () => reject(new Error('WebSocket connection failed'))
    })

    await streamTokens(ws, params.text, tokensPerSecond, (sent) => { simSentChars = sent })

    if (ws.readyState === WebSocket.OPEN) {
      ws.send('')  // end-of-stream sentinel
      await allAudioReceived
      timer.total()
      player.trackEnd()
    }

    activeWs = null
  }

  async function runStream() {
    await player.init()
    abortCtrl = new AbortController()
    const timer = startTimer()

    for await (const wavBuf of api.synthesizeStream(synthParams(), abortCtrl.signal)) {
      timer.chunk()
      await player.schedule(wavBuf)
    }
    timer.total()
    player.trackEnd()
  }

  async function runFullAudio() {
    const timer = startTimer()
    const blob = await api.synthesize(synthParams())
    timer.total()
    resultUrl = URL.createObjectURL(blob)
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
      if (llmSimMode) await runLLMSim()
      else if (streamMode) await runStream()
      else await runFullAudio()
    } catch (e) {
      if (!(e instanceof Error && e.name === 'AbortError')) {
        synthError = e instanceof Error ? e.message : 'Synthesis failed'
      }
    } finally {
      synthesizing = false
    }

    if (streaming && player.hasAudio && !resultUrl) {
      resultUrl = URL.createObjectURL(player.toBlob())
    }
  }

  function handleStop() {
    abortCtrl?.abort()
    activeWs?.close()
    activeWs = null
    player.stop()
  }

  function handleDownload() {
    if (!resultUrl) return
    const a = document.createElement('a')
    a.href = resultUrl
    a.download = 'output.wav'
    a.click()
  }
</script>

<div class="app">
  <header class="header">
    <div class="header-inner">
      <h1 class="logo">Supertonic <span class="logo-sub">TTS</span></h1>
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
      <div class="voice-card-header">
        <span class="field-label">Voice</span>
        <button
          class="import-toggle-btn"
          type="button"
          onclick={() => { showImport = !showImport; importError = null }}
        >
          {showImport ? '✕ Cancel' : '+ Import style'}
        </button>
      </div>

      {#if showImport}
        <div class="import-panel">
          <span class="field-label">
            Style JSON
            <span class="hint-inline">(exported from Supertonic Cloud)</span>
          </span>
          <div class="file-row">
            <button class="file-btn" type="button" onclick={() => importFileInput?.click()}>
              {importFile ? '✓ ' + importFile.name : 'Choose .json file…'}
            </button>
            <input
              bind:this={importFileInput}
              type="file"
              accept=".json,application/json"
              style="display:none"
              onchange={handleImportFileChange}
            />
          </div>
          <label class="field-label spaced" for="import-name">
            Save as <span class="required">*</span>
            <span class="hint-inline">(letters, digits, hyphens, underscores)</span>
          </label>
          <div class="import-name-row">
            <input
              id="import-name"
              class="text-input-sm"
              type="text"
              placeholder="e.g. my_voice"
              bind:value={importName}
            />
            <button
              class="import-btn"
              type="button"
              disabled={!importFile || !importName.trim() || importing}
              onclick={handleImport}
            >
              {#if importing}<span class="btn-spinner"></span> Importing…{:else}Import{/if}
            </button>
          </div>
          {#if importError}
            <p class="import-error">✕ {importError}</p>
          {/if}
        </div>
      {:else}
        <div class="voice-scroll">
          <VoiceSelect
            {voices}
            bind:value={selectedVoiceId}
            ondelete={handleVoiceDelete}
          />
          {#if voices.length === 0}
            <p class="hint">No voices loaded yet. Import a style JSON from Supertonic Cloud.</p>
          {/if}
        </div>
      {/if}
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
    </div>

    <div class="card settings-card">
      <div class="settings-row">
        <div class="settings-col">
          <span class="field-label">Language</span>
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
            min="0.7"
            max="2.0"
            step="0.1"
            bind:value={speed}
            class="range-input"
          />
          <div class="range-marks">
            <span>0.7×</span>
            <span>2.0×</span>
          </div>
        </div>
      </div>

      <div class="settings-col">
        <div class="settings-label-row">
          <label class="field-label" for="steps-range">Quality steps</label>
          <span class="speed-value">{stepsLabel}</span>
        </div>
        <input
          id="steps-range"
          type="range"
          min="1"
          max="50"
          step="1"
          bind:value={totalSteps}
          class="range-input"
        />
        <div class="range-marks">
          <span>1 (fastest)</span>
          <span>50 (best)</span>
        </div>
      </div>

      <div class="settings-col">
        <span class="field-label">Output mode</span>
        <div class="output-btn-group" role="group" aria-label="Output mode">
          <button
            class="output-btn"
            class:active={!streaming}
            type="button"
            onclick={() => { streamMode = false; llmSimMode = false }}
          >⬇ Full audio</button>
          <button
            class="output-btn"
            class:active={streaming}
            type="button"
            onclick={() => { streamMode = true }}
          >▶ Per-sentence streaming</button>
        </div>
      </div>

      {#if streaming}
        <label class="llm-toggle">
          <input
            type="checkbox"
            bind:checked={llmSimMode}
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
      class:stop={showStop}
      type="button"
      disabled={!showStop && !canSynthesize}
      onclick={showStop ? handleStop : handleSynthesize}
    >
      {#if synthesizing}
        <span class="spinner"></span> Synthesizing…
        {#if streaming}<span class="stop-label">■ Stop</span>{/if}
      {:else if player.playing}
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
        <audio src={resultUrl} controls class="audio-player"></audio>
      </div>
    {/if}

    {#if firstChunkDelay !== null || totalDuration !== null}
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

  <footer class="footer">
    <a
      class="footer-link"
      href="https://github.com/matuszelenak/omnivoice-tts-server"
      target="_blank"
      rel="noopener noreferrer"
    >
      <svg class="footer-icon" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>
      </svg>
      <span>Source code</span>
    </a>
    <span class="footer-sep">·</span>
    <a
      class="footer-link"
      href="https://supertone.ai/supertonic"
      target="_blank"
      rel="noopener noreferrer"
    >
      <span class="footer-icon footer-emoji" aria-hidden="true">🔊</span>
      <span>Supertonic model</span>
    </a>
  </footer>
</div>

<style>
  /* ── Layout ── */
  .app {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    background: var(--bg);
    color: var(--text);
  }

  .main {
    flex: 1;
    max-width: 680px;
    width: 100%;
    margin: 0 auto;
    padding: 1.5rem 1rem 3rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  /* ── Header ── */
  .header {
    border-bottom: 1px solid var(--border);
    background: var(--surface-1);
  }

  .header-inner {
    max-width: 680px;
    margin: 0 auto;
    padding: 0.75rem 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .logo {
    font-size: 1.125rem;
    font-weight: 700;
    margin: 0;
    color: var(--text);
  }

  .logo-sub {
    color: var(--primary);
    font-weight: 500;
  }

  /* ── Status badge ── */
  .status-badge {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-muted);
    padding: 0.25rem 0.625rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--surface-2);
  }

  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--text-muted);
    flex-shrink: 0;
  }

  .status-badge.ready { color: var(--success); border-color: color-mix(in srgb, var(--success) 30%, var(--border)); }
  .status-badge.ready .status-dot { background: var(--success); box-shadow: 0 0 0 2px color-mix(in srgb, var(--success) 20%, transparent); }
  .status-badge.unavailable { color: var(--error); border-color: color-mix(in srgb, var(--error) 30%, var(--border)); }
  .status-badge.unavailable .status-dot { background: var(--error); }
  .status-badge.loading .status-dot { animation: pulse 1.2s ease-in-out infinite; }

  /* ── Cards ── */
  .card {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }

  /* ── Voice card ── */
  .voice-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .import-toggle-btn {
    background: none;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-muted);
    font-size: 0.78rem;
    padding: 0.2rem 0.6rem;
    cursor: pointer;
    transition: color 0.12s, border-color 0.12s;
  }

  .import-toggle-btn:hover {
    color: var(--primary);
    border-color: var(--primary);
  }

  .voice-scroll {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    max-height: 260px;
    overflow-y: auto;
  }

  /* ── Import panel ── */
  .import-panel {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .import-name-row {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }

  .import-name-row .text-input-sm {
    flex: 1;
  }

  .import-btn {
    background: var(--primary);
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 0.4rem 0.875rem;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    white-space: nowrap;
    transition: opacity 0.12s;
  }

  .import-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .import-error {
    font-size: 0.8rem;
    color: var(--error);
    margin: 0;
  }

  /* ── Text area ── */
  .text-input {
    width: 100%;
    resize: vertical;
    min-height: 140px;
  }

  .sim-display {
    font-family: inherit;
    resize: none;
    overflow-y: auto;
    white-space: pre-wrap;
  }

  .sim-sent { color: var(--text); }
  .sim-pending { color: var(--text-muted); }

  /* ── Settings card ── */
  .settings-card {
    gap: 0.875rem;
  }

  .settings-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }

  .settings-col {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .settings-label-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }

  .speed-value {
    font-size: 0.78rem;
    color: var(--text-muted);
  }

  .output-btn-group {
    display: flex;
    gap: 0;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }

  .output-btn {
    flex: 1;
    background: none;
    border: none;
    padding: 0.45rem 0.5rem;
    font-size: 0.8rem;
    color: var(--text-muted);
    cursor: pointer;
    transition: background 0.12s, color 0.12s;
  }

  .output-btn + .output-btn { border-left: 1px solid var(--border); }
  .output-btn.active { background: var(--primary); color: #fff; }
  .output-btn:not(.active):hover { background: var(--surface-2); }

  /* ── LLM toggle ── */
  .llm-toggle {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    cursor: pointer;
    user-select: none;
  }

  .toggle-track {
    width: 32px;
    height: 18px;
    border-radius: 999px;
    background: var(--surface-3);
    border: 1px solid var(--border);
    position: relative;
    flex-shrink: 0;
    transition: background 0.15s;
  }

  .toggle-track.on { background: var(--primary); border-color: var(--primary); }

  .toggle-knob {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #fff;
    transition: transform 0.15s;
    box-shadow: 0 1px 3px rgba(0,0,0,.2);
  }

  .toggle-track.on .toggle-knob { transform: translateX(14px); }

  .toggle-text {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    font-size: 0.85rem;
  }

  /* ── Synth button ── */
  .synth-btn {
    background: var(--primary);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    transition: opacity 0.15s, background 0.15s;
  }

  .synth-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .synth-btn.stop { background: var(--error); }

  .stop-label {
    margin-left: 0.25rem;
    font-size: 0.8rem;
    opacity: 0.85;
  }

  /* ── Result card ── */
  .result-card { gap: 0.5rem; }

  .result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .download-btn {
    background: none;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-muted);
    font-size: 0.78rem;
    padding: 0.2rem 0.6rem;
    cursor: pointer;
  }

  .download-btn:hover { color: var(--text); border-color: var(--text-muted); }

  .audio-player { width: 100%; }

  /* ── Footer ── */
  .footer {
    border-top: 1px solid var(--border);
    padding: 0.875rem 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
  }

  .footer-link {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.8rem;
    color: var(--text-muted);
    text-decoration: none;
    transition: color 0.12s;
  }

  .footer-link:hover { color: var(--text); }

  .footer-icon { width: 14px; height: 14px; }
  .footer-emoji { font-size: 0.875rem; }
  .footer-sep { color: var(--border); }

  /* ── Shared form elements ── */
  .field-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .hint-inline {
    font-size: 0.72rem;
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
    color: var(--text-muted);
    opacity: 0.7;
  }

  .hint {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin: 0;
  }

  .required { color: var(--error); }

  .spaced { margin-top: 0.25rem; }

  .textarea {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-size: 0.9rem;
    padding: 0.6rem 0.75rem;
    font-family: inherit;
    outline: none;
    transition: border-color 0.12s;
  }

  .textarea:focus { border-color: var(--primary); }

  .text-input-sm {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 0.875rem;
    padding: 0.4rem 0.6rem;
    font-family: inherit;
    outline: none;
    transition: border-color 0.12s;
  }

  .text-input-sm:focus { border-color: var(--primary); }

  .file-btn {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 0.875rem;
    padding: 0.4rem 0.75rem;
    cursor: pointer;
    text-align: left;
    transition: border-color 0.12s;
    flex: 1;
  }

  .file-btn:hover { border-color: var(--primary); }

  .file-row {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }

  .range-input {
    width: 100%;
    accent-color: var(--primary);
  }

  .range-marks {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-top: -0.1rem;
  }

  /* ── Banners ── */
  .banner {
    padding: 0.6rem 0.875rem;
    border-radius: 8px;
    font-size: 0.85rem;
  }

  .banner-warning { background: color-mix(in srgb, var(--warning) 12%, transparent); color: var(--warning); border: 1px solid color-mix(in srgb, var(--warning) 30%, transparent); }
  .banner-error { background: color-mix(in srgb, var(--error) 10%, transparent); color: var(--error); border: 1px solid color-mix(in srgb, var(--error) 25%, transparent); }

  /* ── Spinner / animation ── */
  .spinner {
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255,255,255,0.35);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    flex-shrink: 0;
  }

  .btn-spinner {
    width: 12px;
    height: 12px;
    border: 2px solid rgba(255,255,255,0.35);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  .stream-hint {
    text-align: center;
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: 0;
  }

  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>

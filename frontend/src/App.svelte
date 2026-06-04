<script lang="ts">
  import { onDestroy, onMount } from 'svelte'
  import type { Language, Voice } from './lib/types.js'
  import * as api from './lib/api.js'
  import { StreamingPlayer } from './lib/player.svelte.js'
  import { streamTokens } from './lib/llm.js'
  import LanguageSelect from './lib/LanguageSelect.svelte'
  import VoiceSelect from './lib/VoiceSelect.svelte'

  const EXPRESSION_TAGS = [
    'laughter', 'sigh', 'confirmation-en', 'question-en', 'question-ah',
    'question-oh', 'question-ei', 'question-yi', 'surprise-ah', 'surprise-oh',
    'surprise-wa', 'surprise-yo', 'dissatisfaction-hnn',
  ]

  const DESIGN_FIELDS = [
    { key: 'gender', label: 'Gender', empty: '— any —', options: [
      ['male', 'Male'], ['female', 'Female'],
    ] },
    { key: 'age', label: 'Age', empty: '— any —', options: [
      ['child', 'Child'], ['teenager', 'Teenager'], ['young adult', 'Young Adult'],
      ['middle-aged', 'Middle-aged'], ['elderly', 'Elderly'],
    ] },
    { key: 'pitch', label: 'Pitch', empty: '— any —', options: [
      ['very low pitch', 'Very Low'], ['low pitch', 'Low'], ['moderate pitch', 'Moderate'],
      ['high pitch', 'High'], ['very high pitch', 'Very High'],
    ] },
    { key: 'style', label: 'Style', empty: '— none —', options: [
      ['whisper', 'Whisper'],
    ] },
  ] as const

  const ACCENTS = [
    'American', 'British', 'Australian', 'Canadian', 'Indian',
    'Korean', 'Japanese', 'Portuguese', 'Russian',
  ]

  // ── Server / data state ──────────────────────────────────────────────

  type ServerStatus = 'loading' | 'ready' | 'unavailable'

  let serverStatus = $state<ServerStatus>('loading')
  let languages = $state<Language[]>([])
  let voices = $state<Voice[]>([])
  let loadError = $state<string | null>(null)

  // ── Synthesis inputs ─────────────────────────────────────────────────

  type VoiceTab = 'cloning' | 'design'

  let activeTab = $state<VoiceTab>('cloning')
  let language = $state('en')
  let text = $state('')
  let speed = $state(1.0)

  // Cloning tab
  let selectedVoiceId = $state<string | null>(null)
  let useCustomVoice = $state(false)
  let customFile = $state<File | null>(null)
  let refText = $state('')
  let refVoiceName = $state('')
  let fileInput: HTMLInputElement | undefined = $state()

  // Design tab — field order defines the instruct string
  let design = $state({ gender: '', age: '', pitch: '', style: '', accent: '' })
  let instruct = $derived(Object.values(design).filter(Boolean).join(', '))

  // Accents are English-only: switching to another language clears the accent,
  // and picking an accent switches the language to English (see the select's onchange).
  $effect(() => {
    if (language !== 'en' && design.accent) design.accent = ''
  })

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

  let usingCustomAudio = $derived(activeTab === 'cloning' && useCustomVoice)
  let streaming = $derived(streamMode || llmSimMode)
  let showStop = $derived((synthesizing && streaming) || player.playing)
  let speedLabel = $derived(speed === 1.0 ? '1.0× (normal)' : `${speed.toFixed(1)}×`)

  let canSynthesize = $derived(
    text.trim().length > 0 &&
      !synthesizing &&
      !(llmSimMode && usingCustomAudio) &&
      (activeTab === 'design' ||
        selectedVoiceId !== null ||
        (useCustomVoice && customFile !== null && refText.trim().length > 0)),
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
        adoptVoiceLanguage(vcs[0].id)
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

  function adoptVoiceLanguage(id: string | null) {
    const voice = voices.find((v) => v.id === id)
    if (voice?.language) language = voice.language
  }

  function handleVoiceDelete(id: string) {
    voices = voices.filter((v) => v.id !== id)
    if (selectedVoiceId === id) {
      selectedVoiceId = voices[0]?.id ?? null
      adoptVoiceLanguage(selectedVoiceId)
    }
  }

  function handleFileChange(e: Event) {
    const input = e.target as HTMLInputElement
    customFile = input.files?.[0] ?? null
  }

  function clearFile() {
    customFile = null
    if (fileInput) fileInput.value = ''
  }

  function synthParams() {
    return {
      text: text.trim(),
      language,
      speed: speed === 1.0 ? undefined : speed,
      voiceId: activeTab === 'cloning' && !useCustomVoice && selectedVoiceId ? selectedVoiceId : undefined,
      refAudio: usingCustomAudio && customFile ? customFile : undefined,
      refText: usingCustomAudio ? refText.trim() : undefined,
      refVoiceName: usingCustomAudio && refVoiceName.trim() ? refVoiceName.trim() : undefined,
      instruct: activeTab === 'design' && instruct ? instruct : undefined,
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
      instruct: params.instruct,
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

    // Streamed sessions: offer whatever audio arrived (even partial) for replay/download.
    if (streaming && player.hasAudio && !resultUrl) {
      resultUrl = URL.createObjectURL(player.toBlob())
    }

    // A newly saved voice should show up in the list right away.
    if (usingCustomAudio && refVoiceName.trim()) {
      try {
        voices = await api.fetchVoices()
      } catch { /* non-critical */ }
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
                onchange={adoptVoiceLanguage}
                ondelete={handleVoiceDelete}
              />
              {#if voices.length === 0}
                <p class="hint">
                  No voice samples found. Create a directory per voice in
                  <code>server/voices/</code> containing <code>voice.wav</code> +
                  <code>&lt;lang&gt;.txt</code>, or set <code>VOICE_SAMPLES_DIR</code>.
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
              <span class="field-label">
                Reference audio <span class="required">*</span>
                <span class="hint-inline">(recommended length: 5–10 seconds)</span>
              </span>
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
                  <button class="clear-btn" type="button" onclick={clearFile} title="Remove file">✕</button>
                {/if}
              </div>
              <label class="field-label spaced" for="ref-text">
                Reference transcript <span class="required">*</span>
              </label>
              <textarea
                id="ref-text"
                class="textarea"
                rows="2"
                placeholder="Type exactly what the audio says…"
                bind:value={refText}
              ></textarea>
              <label class="field-label spaced" for="ref-voice-name">
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
            {#each DESIGN_FIELDS as field (field.key)}
              <div class="design-field">
                <label class="field-label" for="d-{field.key}">{field.label}</label>
                <select id="d-{field.key}" class="design-select" bind:value={design[field.key]}>
                  <option value="">{field.empty}</option>
                  {#each field.options as [value, label] (value)}
                    <option {value}>{label}</option>
                  {/each}
                </select>
              </div>
            {/each}
            <div class="design-field design-field-full">
              <label class="field-label" for="d-accent">
                English Accent
                <span class="hint-inline">(selecting one switches the language to English)</span>
              </label>
              <select
                id="d-accent"
                class="design-select"
                bind:value={design.accent}
                onchange={() => { if (design.accent) language = 'en' }}
              >
                <option value="">— any —</option>
                {#each ACCENTS as accent (accent)}
                  <option value="{accent.toLowerCase()} accent">{accent}</option>
                {/each}
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
        {#each EXPRESSION_TAGS as tag (tag)}
          <code class="tag-chip">[{tag}]</code>
        {/each}
      </p>
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
        <label class="llm-toggle" class:disabled={usingCustomAudio}>
          <input
            type="checkbox"
            bind:checked={llmSimMode}
            disabled={usingCustomAudio}
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
      href="https://huggingface.co/spaces/k2-fsa/OmniVoice"
      target="_blank"
      rel="noopener noreferrer"
    >
      <span class="footer-icon footer-emoji" aria-hidden="true">🤗</span>
      <span>OmniVoice model</span>
    </a>
  </footer>
</div>

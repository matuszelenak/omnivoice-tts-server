<script lang="ts">
  import { onDestroy, onMount } from 'svelte'
  import Icon from './lib/Icon.svelte'
  import type { Language, Voice } from './lib/types.js'
  import * as api from './lib/api.js'
  import type { ServerHealth } from './lib/api.js'
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
  let health = $state<ServerHealth | null>(null)
  let languages = $state<Language[]>([])
  let voices = $state<Voice[]>([])
  let loadError = $state<string | null>(null)

  // ── Synthesis inputs ─────────────────────────────────────────────────

  type VoiceTab = 'cloning' | 'design'

  const TABS: { value: VoiceTab; label: string; hint: string; icon: string }[] = [
    { value: 'cloning', label: 'Voice cloning', hint: 'From a reference clip', icon: 'user' },
    { value: 'design', label: 'Voice design', hint: 'Describe the voice', icon: 'sparkle' },
  ]

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
  let fileDropActive = $state(false)
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
  let healthPoll: ReturnType<typeof setTimeout> | null = null
  let destroyed = false

  // ── Derived view state ───────────────────────────────────────────────

  let usingCustomAudio = $derived(activeTab === 'cloning' && useCustomVoice)
  let streaming = $derived(streamMode || llmSimMode)
  let showStop = $derived((synthesizing && streaming) || player.playing)
  let speedLabel = $derived(speed === 1.0 ? '1.0×' : `${speed.toFixed(1)}×`)
  let backendOffline = $derived(serverStatus === 'unavailable')

  let canSynthesize = $derived(
    text.trim().length > 0 &&
      !synthesizing &&
      !(llmSimMode && usingCustomAudio) &&
      (activeTab === 'design' ||
        selectedVoiceId !== null ||
        (useCustomVoice && customFile !== null && refText.trim().length > 0)),
  )

  let charCount = $derived(text.trim().length)
  let wordCount = $derived(charCount ? text.trim().split(/\s+/).length : 0)

  // Which model/device this instance serves. Two instances on different GPUs look
  // identical otherwise, so the model name goes in the status chip and the title.
  let modelTitle = $derived(
    backendOffline ? 'No backend' : (health?.model || 'OmniVoice'),
  )
  let modelMeta = $derived(
    health?.dtype ? (health.dtype === 'float32' ? 'fp32' : health.dtype) : null,
  )
  let modelTooltip = $derived.by(() => {
    if (backendOffline) return 'The OmniVoice API is not answering /health'
    if (!health) return 'Contacting the server…'
    return [
      health.model,
      health.device ? `device ${health.device}` : null,
      health.streamLimit ? `${health.streamLimit} concurrent sessions` : null,
      health.asrAvailable ? 'ASR transcripts enabled' : 'ASR transcripts disabled',
    ]
      .filter(Boolean)
      .join(' · ')
  })
  let statusLabel = $derived(
    backendOffline
      ? 'Server unreachable'
      : !health
        ? 'Connecting…'
        : health.modelLoaded
          ? 'Model ready'
          : 'Loading model…',
  )

  let hasOutput = $derived(resultUrl !== null)

  // ── Lifecycle ────────────────────────────────────────────────────────

  onMount(async () => {
    await refreshHealth()

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
    destroyed = true
    if (healthPoll) clearTimeout(healthPoll)
    abortCtrl?.abort()
    activeWs?.close()
    player.destroy()
    if (resultUrl) URL.revokeObjectURL(resultUrl)
  })

  // A cold GPU keeps the model out of `/health` for a few seconds, so the chip
  // reports "Loading model…" and keeps asking until the server is ready.
  async function refreshHealth() {
    try {
      health = await api.checkHealth()
      serverStatus = health.modelLoaded ? 'ready' : 'loading'
      if (!health.modelLoaded && !destroyed) {
        healthPoll = setTimeout(() => void refreshHealth(), 2000)
      }
    } catch {
      serverStatus = 'unavailable'
    }
  }

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

  function setFile(file: File | null) {
    if (!file) {
      clearFile()
      return
    }
    if (!file.type.startsWith('audio/') && !/\.(wav|mp3|flac|ogg|m4a|opus)$/i.test(file.name)) {
      synthError = `Unsupported reference file: ${file.type || file.name}`
      return
    }
    synthError = null
    customFile = file
  }

  function handleFileChange(e: Event) {
    const input = e.target as HTMLInputElement
    setFile(input.files?.[0] ?? null)
  }

  function clearFile() {
    customFile = null
    if (fileInput) fileInput.value = ''
  }

  function onRefDragOver(e: DragEvent) {
    e.preventDefault()
    fileDropActive = true
  }

  function onRefDrop(e: DragEvent) {
    e.preventDefault()
    fileDropActive = false
    if (synthesizing) return
    const file = e.dataTransfer?.files?.[0]
    if (file) setFile(file)
  }

  function fmtBytes(bytes: number): string {
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  function synthParams(voiceIdOverride?: string) {
    return {
      text: text.trim(),
      language,
      speed: speed === 1.0 ? undefined : speed,
      voiceId: voiceIdOverride ?? (activeTab === 'cloning' && !useCustomVoice && selectedVoiceId ? selectedVoiceId : undefined),
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

  async function runStream(voiceId?: string) {
    await player.init()
    abortCtrl = new AbortController()
    const timer = startTimer()

    for await (const wavBuf of api.synthesizeStream(synthParams(voiceId), abortCtrl.signal)) {
      timer.chunk()
      await player.schedule(wavBuf)
    }
    timer.total()
    player.trackEnd()
  }

  async function runFullAudio(voiceId?: string) {
    const timer = startTimer()
    const blob = await api.synthesize(synthParams(voiceId))
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
      // When using custom audio with a voice name, save the voice first
      // then synthesize using the newly created voice ID.
      let effectiveVoiceId: string | undefined
      if (usingCustomAudio && customFile && refVoiceName.trim()) {
        const saved = await api.createVoice({
          name: refVoiceName.trim(),
          refAudio: customFile,
          refText: refText.trim(),
          language,
        })
        effectiveVoiceId = saved.id
        // Refresh the voice list so the new voice appears
        try { voices = await api.fetchVoices() } catch { /* non-critical */ }
      }

      if (llmSimMode) await runLLMSim()
      else if (streamMode) await runStream(effectiveVoiceId)
      else await runFullAudio(effectiveVoiceId)
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
    a.download = 'omnivoice-output.wav'
    a.click()
  }

  function clearOutput() {
    player.stop()
    if (resultUrl) URL.revokeObjectURL(resultUrl)
    resultUrl = null
    firstChunkDelay = null
    totalDuration = null
    synthError = null
  }
</script>

<svelte:head>
  <title>{health ? `${health.model} — TTS` : 'OmniVoice TTS'}</title>
</svelte:head>

<div class="shell">
  <h1 class="sr-only">OmniVoice TTS</h1>

  <main>
    {#if backendOffline}
      <div class="alert" role="status">
        <Icon name="alert" size={16} />
        <span>Server unreachable — check that the OmniVoice API is running.</span>
      </div>
    {/if}

    {#if loadError && !backendOffline}
      <div class="alert" role="status">
        <Icon name="alert" size={16} />
        <span>{loadError} — running with limited functionality.</span>
      </div>
    {/if}

    <section class="panel">
      <div class="modelbar">
        <span class="model-chip" class:offline={backendOffline} title={modelTooltip || undefined}>
          <span class="model-dot" class:busy={synthesizing} aria-hidden="true"></span>
          <span class="model-name">{modelTitle}</span>
          {#if modelMeta}
            <span class="model-meta">{modelMeta}</span>
          {/if}
        </span>
        <span class="model-status" class:offline={backendOffline} class:ready={serverStatus === 'ready'}>
          {statusLabel}
        </span>
      </div>

      <div class="modes" role="group" aria-label="Voice source">
        {#each TABS as tab (tab.value)}
          <button
            type="button"
            aria-pressed={activeTab === tab.value}
            class="mode"
            class:active={activeTab === tab.value}
            onclick={() => (activeTab = tab.value)}
          >
            <span class="mode-icon"><Icon name={tab.icon} size={16} /></span>
            <span class="mode-copy">
              <span class="mode-label">{tab.label}</span>
              <span class="mode-hint">{tab.hint}</span>
            </span>
          </button>
        {/each}
      </div>

      {#if activeTab === 'cloning'}
        {#if !useCustomVoice}
          <div class="voices">
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
            <button
              type="button"
              class="use-custom"
              disabled={llmSimMode}
              onclick={() => (useCustomVoice = true)}
            >
              <Icon name="plus" size={14} /> Use custom audio
            </button>
          </div>
        {:else}
          <div class="custom-voice">
            <div class="custom-head">
              <span class="field-label">Custom reference</span>
              <button
                type="button"
                class="btn btn-quiet"
                onclick={() => (useCustomVoice = false)}
              >
                <Icon name="chevron" size={13} /> Built-in voices
              </button>
            </div>

            <label
              class="drop"
              class:dragging={fileDropActive}
              class:filled={customFile !== null}
              class:disabled={synthesizing}
              ondragover={onRefDragOver}
              ondragleave={() => (fileDropActive = false)}
              ondrop={onRefDrop}
            >
              <input
                bind:this={fileInput}
                type="file"
                accept="audio/*,.wav,.mp3,.flac,.ogg"
                onchange={handleFileChange}
                disabled={synthesizing}
              />
              <span class="drop-icon">
                <Icon name={customFile ? 'file-audio' : 'upload'} size={17} />
              </span>
              <span class="drop-copy">
                <strong>{customFile ? customFile.name : 'Drop a reference clip'}</strong>
                <span>
                  {#if customFile}
                    {fmtBytes(customFile.size)} · click to replace
                  {:else}
                    or click to browse · 5–10 s of clean speech
                  {/if}
                </span>
              </span>
              {#if customFile}
                <button
                  type="button"
                  class="drop-clear"
                  aria-label="Remove reference clip"
                  title="Remove reference clip"
                  onclick={(e) => { e.preventDefault(); clearFile() }}
                >
                  <Icon name="x" size={13} />
                </button>
              {:else}
                <span class="drop-hint">WAV · MP3 · FLAC</span>
              {/if}
            </label>

            <label class="field" for="ref-text">
              <span class="field-label">
                Reference transcript <span class="required">*</span>
              </span>
              <textarea
                id="ref-text"
                class="textarea"
                rows="2"
                placeholder="Type exactly what the clip says…"
                bind:value={refText}
              ></textarea>
            </label>

            <label class="field" for="ref-voice-name">
              <span class="field-label">
                Save as voice <span class="field-note">optional</span>
              </span>
              <input
                id="ref-voice-name"
                class="input"
                type="text"
                placeholder="e.g. my_voice — letters, digits, hyphens"
                bind:value={refVoiceName}
              />
            </label>
          </div>
        {/if}
      {:else}
        <div class="design">
          <div class="design-grid">
            {#each DESIGN_FIELDS as field (field.key)}
              <label class="field" for="d-{field.key}">
                <span class="field-label">{field.label}</span>
                <span class="select-wrap">
                  <select id="d-{field.key}" bind:value={design[field.key]}>
                    <option value="">{field.empty}</option>
                    {#each field.options as [value, label] (value)}
                      <option {value}>{label}</option>
                    {/each}
                  </select>
                  <Icon name="chevron" size={14} />
                </span>
              </label>
            {/each}

            <label class="field field-full" for="d-accent">
              <span class="field-label">
                English accent <span class="field-note">switches the language to English</span>
              </span>
              <span class="select-wrap">
                <select
                  id="d-accent"
                  bind:value={design.accent}
                  onchange={() => { if (design.accent) language = 'en' }}
                >
                  <option value="">— any —</option>
                  {#each ACCENTS as accent (accent)}
                    <option value="{accent.toLowerCase()} accent">{accent}</option>
                  {/each}
                </select>
                <Icon name="chevron" size={14} />
              </span>
            </label>
          </div>

          {#if instruct}
            <p class="instruct-preview">
              <Icon name="sparkle" size={13} />
              <span class="mono">{instruct}</span>
            </p>
          {:else}
            <p class="hint">
              All fields are optional — leave them empty and let the model decide.
            </p>
          {/if}
        </div>
      {/if}
    </section>

    <section class="panel">
      <div class="panel-head">
        <div class="panel-title">
          <span class="panel-icon"><Icon name="file-text" size={16} /></span>
          <h2>Text</h2>
        </div>
        <div class="panel-actions">
          {#if charCount > 0}
            <span class="counter">{wordCount} {wordCount === 1 ? 'word' : 'words'} · {charCount} chars</span>
            <button
              type="button"
              class="icon-btn"
              onclick={() => (text = '')}
              title="Clear text"
              aria-label="Clear text"
            >
              <Icon name="trash" size={15} />
            </button>
          {/if}
        </div>
      </div>

      <div class="panel-body">
        {#if llmSimMode && synthesizing}
          <div class="textarea text-input sim-display" aria-live="polite">
            <span class="sim-sent">{text.slice(0, simSentChars)}</span><span class="sim-pending">{text.slice(simSentChars)}</span>
          </div>
        {:else}
          <textarea
            id="text-input"
            class="textarea text-input"
            rows="9"
            aria-label="Text to synthesize"
            placeholder="Type or paste the text you want to convert to speech…"
            bind:value={text}
          ></textarea>
        {/if}

        <div class="tags">
          <span class="tags-label">Expression tags</span>
          {#each EXPRESSION_TAGS as tag (tag)}
            <code class="tag-chip">[{tag}]</code>
          {/each}
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="settings-row">
        <div class="field">
          <span class="field-label">Language</span>
          <LanguageSelect {languages} bind:value={language} />
        </div>

        <div class="field">
          <div class="field-row">
            <span class="field-label" id="speed-label">Speed</span>
            <span class="value mono">{speedLabel}</span>
          </div>
          <input
            class="range"
            type="range"
            min="0.5"
            max="2.0"
            step="0.1"
            bind:value={speed}
            aria-labelledby="speed-label"
          />
          <div class="range-marks">
            <span>0.5×</span>
            <span>2.0×</span>
          </div>
        </div>
      </div>

      <div class="setting">
        <span class="field-label">Output mode</span>
        <div class="seg" role="group" aria-label="Output mode">
          <button
            type="button"
            class="seg-btn"
            class:active={!streaming}
            onclick={() => { streamMode = false; llmSimMode = false }}
          >
            <Icon name="file-audio" size={15} />
            <span class="seg-copy">
              <span class="seg-label">Full audio</span>
              <span class="seg-hint">Wait for the whole clip</span>
            </span>
          </button>
          <button
            type="button"
            class="seg-btn"
            class:active={streaming}
            onclick={() => { streamMode = true }}
          >
            <Icon name="wave" size={15} />
            <span class="seg-copy">
              <span class="seg-label">Streaming</span>
              <span class="seg-hint">Play per sentence, ~0.2 s to first audio</span>
            </span>
          </button>
        </div>
      </div>

      {#if streaming}
        <label class="switch-row" class:disabled={usingCustomAudio}>
          <input
            type="checkbox"
            bind:checked={llmSimMode}
            disabled={usingCustomAudio}
            class="sr-only"
          />
          <span class="switch-track" class:on={llmSimMode} aria-hidden="true">
            <span class="switch-knob"></span>
          </span>
          <span class="switch-copy">
            <span class="switch-label">Simulate LLM</span>
            <span class="field-note">
              {usingCustomAudio
                ? 'not available with custom reference audio'
                : 'send the text token-by-token over a WebSocket'}
            </span>
          </span>
        </label>

        {#if llmSimMode}
          <div class="field token-field">
            <div class="field-row">
              <label class="field-label" for="tps-range">Token rate</label>
              <span class="value mono">{tokensPerSecond} tok/s</span>
            </div>
            <input
              id="tps-range"
              class="range"
              type="range"
              min="1"
              max="150"
              step="1"
              bind:value={tokensPerSecond}
            />
            <div class="range-marks">
              <span>1</span>
              <span>75</span>
              <span>150</span>
            </div>
          </div>
        {/if}
      {/if}

      {#if synthError}
        <div class="alert" role="alert">
          <Icon name="alert" size={16} />
          <span>{synthError}</span>
          <button
            type="button"
            class="alert-dismiss"
            aria-label="Dismiss error"
            onclick={() => (synthError = null)}
          >
            <Icon name="x" size={13} />
          </button>
        </div>
      {/if}

      <button
        type="button"
        class="btn synth-btn"
        class:stop={showStop}
        disabled={!showStop && !canSynthesize}
        onclick={showStop ? handleStop : handleSynthesize}
      >
        {#if synthesizing}
          <span class="btn-spinner"></span>
          {showStop ? 'Stop' : 'Synthesizing…'}
        {:else if player.playing}
          <Icon name="stop" size={16} />
          Stop playback
        {:else}
          <Icon name="play" size={16} />
          Synthesize
        {/if}
      </button>
    </section>

    <section class="panel output-panel" class:synthesising={synthesizing}>
      <div class="panel-head">
        <div class="panel-title">
          <span class="panel-icon"><Icon name="file-audio" size={16} /></span>
          <h2>Audio</h2>
          {#if synthesizing && streaming}
            <span class="badge badge-live">streaming</span>
          {/if}
        </div>
        <div class="panel-actions">
          {#if firstChunkDelay !== null}
            <span class="counter mono">
              first chunk {(firstChunkDelay / 1000).toFixed(2)} s
              {#if totalDuration !== null} · total {(totalDuration / 1000).toFixed(2)} s{/if}
            </span>
          {:else if totalDuration !== null}
            <span class="counter mono">rendered in {(totalDuration / 1000).toFixed(2)} s</span>
          {/if}
          <button
            type="button"
            class="icon-btn"
            onclick={handleDownload}
            disabled={!hasOutput}
            title="Download as .wav"
            aria-label="Download audio as WAV"
          >
            <Icon name="download" size={15} />
          </button>
          <button
            type="button"
            class="icon-btn"
            onclick={clearOutput}
            disabled={!hasOutput && synthError === null}
            title="Clear audio"
            aria-label="Clear audio"
          >
            <Icon name="trash" size={15} />
          </button>
        </div>
      </div>

      <div class="panel-body output-body">
        {#if synthesizing}
          <div class="empty">
            <span class="spinner"></span>
            <p>Synthesizing…</p>
            <small>
              {#if streaming}
                Audio starts playing as soon as the first sentence is rendered.
              {:else}
                The whole clip is rendered before playback starts.
              {/if}
            </small>
          </div>
        {:else if resultUrl}
          <div class="player">
            <div class="player-head">
              <span class="player-file">
                <Icon name="file-audio" size={15} />
                <span class="player-name">omnivoice-output.wav</span>
              </span>
              <span class="badge">wav</span>
            </div>
            <!-- svelte-ignore a11y_media_has_caption -->
            <audio controls src={resultUrl} aria-label="Synthesized audio"></audio>
          </div>
        {:else}
          <div class="empty">
            <span class="empty-icon"><Icon name="wave" size={20} /></span>
            <p>Nothing synthesized yet.</p>
            <small>Pick a voice, write some text, then hit Synthesize.</small>
          </div>
        {/if}
      </div>
    </section>
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
      <span class="footer-emoji" aria-hidden="true">🤗</span>
      <span>OmniVoice model</span>
    </a>
  </footer>
</div>

<style>
  .shell {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    width: 100%;
    max-width: 880px;
    margin: 0 auto;
    padding: 1.75rem 1.25rem 2rem;
  }

  main {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
  }

  .panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 1.15rem;
  }

  /* ── Status bar ─────────────────────────────────────────────────────────── */
  .modelbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.9rem;
    flex-wrap: wrap;
  }
  .model-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    max-width: 100%;
    padding: 0.3rem 0.6rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.02);
    font-size: 0.78rem;
    color: var(--text);
    cursor: default;
  }
  .model-name {
    font-weight: 600;
    letter-spacing: -0.01em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .model-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent-strong);
    box-shadow: 0 0 0 3px rgba(139, 155, 255, 0.15);
    flex: none;
  }
  .model-dot.busy {
    animation: breathe 1.2s ease-in-out infinite;
  }
  .model-chip.offline .model-dot {
    background: var(--text-faint);
    box-shadow: none;
    animation: none;
  }
  .model-meta {
    padding: 0.05rem 0.35rem;
    border-radius: var(--radius-sm);
    background: rgba(255, 255, 255, 0.05);
    font-size: 0.7rem;
    font-variant-numeric: tabular-nums;
    color: var(--text-faint);
  }
  .model-status {
    font-size: 0.72rem;
    color: var(--text-faint);
  }
  .model-status.ready {
    color: var(--text-dim);
  }
  .model-status.offline {
    color: var(--danger);
  }

  /* ── Voice source switcher ──────────────────────────────────────────────── */
  .modes {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.35rem;
    padding: 0.3rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    background: var(--surface-2);
  }

  .mode {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.55rem 0.6rem;
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--text-dim);
    text-align: left;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
  }
  .mode:hover:not(.active) {
    background: rgba(255, 255, 255, 0.04);
    color: var(--text);
  }
  .mode.active {
    background: var(--surface-3);
    border-color: var(--line-strong);
    color: var(--text);
  }
  .mode-icon {
    display: grid;
    place-items: center;
    width: 24px;
    height: 24px;
    flex-shrink: 0;
    color: var(--text-faint);
  }
  .mode.active .mode-icon {
    color: var(--accent);
  }
  .mode-copy {
    display: flex;
    flex-direction: column;
    min-width: 0;
    line-height: 1.2;
  }
  .mode-label {
    font-size: 0.84rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .mode-hint {
    font-size: 0.68rem;
    color: var(--text-faint);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* ── Fields ─────────────────────────────────────────────────────────────── */
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    min-width: 0;
  }
  .field-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .field-full {
    grid-column: 1 / -1;
  }
  .field-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-faint);
  }
  .field-note {
    margin-left: 0.3rem;
    font-size: 0.7rem;
    font-weight: 400;
    letter-spacing: 0;
    text-transform: none;
    color: var(--text-faint);
    opacity: 0.85;
  }
  .required {
    color: var(--danger);
  }

  .select-wrap {
    position: relative;
    display: flex;
    align-items: center;
  }
  .select-wrap select {
    appearance: none;
    width: 100%;
    padding: 0.5rem 2rem 0.5rem 0.7rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    background: var(--surface-2);
    color: var(--text);
    font-size: 0.86rem;
    transition: border-color 0.15s ease;
  }
  .select-wrap select:hover:not(:disabled) {
    border-color: var(--line-strong);
  }
  .select-wrap :global(svg) {
    position: absolute;
    right: 0.65rem;
    color: var(--text-faint);
    pointer-events: none;
  }

  .textarea,
  .input {
    width: 100%;
    padding: 0.55rem 0.75rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    background: var(--surface-2);
    color: var(--text);
    font-size: 0.9rem;
    line-height: 1.55;
    transition: border-color 0.15s ease;
  }
  .textarea {
    resize: vertical;
  }
  .textarea::placeholder,
  .input::placeholder {
    color: var(--text-faint);
  }
  .textarea:hover,
  .input:hover {
    border-color: var(--line-strong);
  }
  .textarea:focus,
  .input:focus {
    outline: none;
    border-color: rgba(139, 155, 255, 0.55);
  }
  .text-input {
    min-height: 12rem;
  }

  /* ── Voice list / custom reference ──────────────────────────────────────── */
  .voices {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    margin-top: 0.9rem;
  }

  .use-custom {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    align-self: center;
    padding: 0.4rem 0.9rem;
    border: 1px dashed var(--line-strong);
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--text-dim);
    font-size: 0.78rem;
    transition: color 0.15s ease, border-color 0.15s ease;
  }
  .use-custom:hover:not(:disabled) {
    color: var(--accent-strong);
    border-color: rgba(139, 155, 255, 0.5);
  }
  .use-custom:disabled {
    opacity: 0.45;
  }

  .custom-voice {
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
    margin-top: 0.9rem;
    padding-top: 0.9rem;
    border-top: 1px solid var(--line);
  }
  .custom-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
  }

  .drop {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.7rem 0.85rem;
    border: 1px dashed var(--line-strong);
    border-radius: var(--radius-md);
    background: var(--surface-2);
    color: var(--text-dim);
    cursor: pointer;
    transition: border-color 0.15s ease, background 0.15s ease, color 0.15s ease;
  }
  .drop:hover,
  .drop.dragging {
    border-color: rgba(139, 155, 255, 0.6);
    color: var(--text);
  }
  .drop.dragging {
    border-style: solid;
    background: var(--surface-3);
  }
  .drop.filled {
    border-style: solid;
  }
  .drop.disabled {
    opacity: 0.5;
    pointer-events: none;
  }
  .drop input {
    position: absolute;
    width: 1px;
    height: 1px;
    opacity: 0;
    pointer-events: none;
  }
  .drop-icon {
    display: grid;
    place-items: center;
    width: 30px;
    height: 30px;
    flex-shrink: 0;
    border-radius: 4px;
    background: var(--surface-3);
    color: var(--accent);
  }
  .drop-copy {
    display: flex;
    flex-direction: column;
    min-width: 0;
    line-height: 1.25;
    font-size: 0.8rem;
  }
  .drop-copy strong {
    font-weight: 550;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .drop-copy span {
    color: var(--text-faint);
    font-size: 0.74rem;
  }
  .drop-hint {
    margin-left: auto;
    flex-shrink: 0;
    font-size: 0.66rem;
    letter-spacing: 0.06em;
    color: var(--text-faint);
  }
  .drop-clear {
    display: grid;
    place-items: center;
    margin-left: auto;
    width: 24px;
    height: 24px;
    flex-shrink: 0;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: transparent;
    color: var(--text-faint);
  }
  .drop-clear:hover {
    color: var(--danger);
    border-color: rgba(255, 118, 118, 0.4);
  }

  /* ── Voice design ───────────────────────────────────────────────────────── */
  .design {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    margin-top: 0.9rem;
    padding-top: 0.9rem;
    border-top: 1px solid var(--line);
  }
  .design-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.7rem 1rem;
  }

  .instruct-preview {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.45rem 0.7rem;
    border: 1px solid rgba(139, 155, 255, 0.25);
    border-radius: var(--radius-sm);
    background: rgba(139, 155, 255, 0.07);
    color: var(--accent-strong);
    font-size: 0.78rem;
    overflow-wrap: anywhere;
  }

  .hint {
    font-size: 0.78rem;
    color: var(--text-faint);
    line-height: 1.5;
  }
  .hint code {
    font-family: var(--font-mono);
    font-size: 0.92em;
    padding: 0.1em 0.35em;
    border-radius: 4px;
    background: var(--surface-3);
    color: var(--text-dim);
  }

  /* ── Panel head / body ──────────────────────────────────────────────────── */
  .panel-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
  }
  .panel-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    min-width: 0;
  }
  .panel-icon {
    display: grid;
    place-items: center;
    color: var(--text-faint);
  }
  .panel-title h2 {
    font-size: 0.86rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    white-space: nowrap;
  }
  .panel-actions {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    min-width: 0;
  }
  .counter {
    margin-right: 0.35rem;
    font-size: 0.72rem;
    color: var(--text-faint);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }

  .icon-btn {
    display: grid;
    place-items: center;
    width: 30px;
    height: 30px;
    border: 1px solid transparent;
    border-radius: 4px;
    background: transparent;
    color: var(--text-faint);
    transition: color 0.15s ease, background 0.15s ease, border-color 0.15s ease;
  }
  .icon-btn:hover:not(:disabled) {
    color: var(--text);
    background: var(--surface-3);
    border-color: var(--line);
  }
  .icon-btn:disabled {
    opacity: 0.35;
  }

  .panel-body {
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
  }

  /* ── Expression tags ────────────────────────────────────────────────────── */
  .tags {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.3rem;
  }
  .tags-label {
    margin-right: 0.15rem;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-faint);
    white-space: nowrap;
  }
  .tag-chip {
    padding: 0.12rem 0.45rem;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: var(--surface-2);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    white-space: nowrap;
  }

  /* ── Settings ───────────────────────────────────────────────────────────── */
  .settings-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.7rem 1rem;
    align-items: start;
  }
  .setting {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--line);
  }

  .seg {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.35rem;
    padding: 0.3rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    background: var(--surface-2);
  }
  .seg-btn {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.6rem;
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--text-dim);
    text-align: left;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
  }
  .seg-btn:hover:not(.active) {
    background: rgba(255, 255, 255, 0.04);
    color: var(--text);
  }
  .seg-btn.active {
    background: var(--surface-3);
    border-color: var(--line-strong);
    color: var(--text);
  }
  .seg-btn.active :global(svg) {
    color: var(--accent);
  }
  .seg-copy {
    display: flex;
    flex-direction: column;
    min-width: 0;
    line-height: 1.2;
  }
  .seg-label {
    font-size: 0.82rem;
    font-weight: 600;
    white-space: nowrap;
  }
  .seg-hint {
    font-size: 0.68rem;
    color: var(--text-faint);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .switch-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-top: 0.9rem;
    cursor: pointer;
    user-select: none;
  }
  .switch-row.disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .switch-track {
    position: relative;
    width: 36px;
    height: 20px;
    flex-shrink: 0;
    border-radius: 999px;
    background: var(--surface-3);
    border: 1px solid var(--line);
    transition: background 0.2s ease, border-color 0.2s ease;
  }
  .switch-track.on {
    background: rgba(139, 155, 255, 0.85);
    border-color: var(--accent);
  }
  .switch-knob {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #fff;
    transition: transform 0.2s ease;
  }
  .switch-track.on .switch-knob {
    transform: translateX(16px);
  }
  .switch-row:focus-within .switch-track {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .switch-copy {
    display: flex;
    flex-direction: column;
    line-height: 1.3;
    min-width: 0;
  }
  .switch-label {
    font-size: 0.82rem;
    font-weight: 600;
  }
  .token-field {
    margin-top: 0.8rem;
  }

  /* ── Range sliders ──────────────────────────────────────────────────────── */
  .range {
    width: 100%;
    height: 18px;
    appearance: none;
    background: transparent;
    cursor: pointer;
  }
  .range:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .range::-webkit-slider-runnable-track {
    height: 4px;
    border-radius: 2px;
    background: var(--surface-3);
  }
  .range::-moz-range-track {
    height: 4px;
    border-radius: 2px;
    background: var(--surface-3);
  }
  .range::-moz-range-progress {
    height: 4px;
    border-radius: 2px;
    background: rgba(139, 155, 255, 0.6);
  }
  .range::-webkit-slider-thumb {
    appearance: none;
    width: 14px;
    height: 14px;
    margin-top: -5px;
    border-radius: 50%;
    background: var(--accent);
    border: none;
  }
  .range::-moz-range-thumb {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--accent);
    border: none;
  }
  .range:hover:not(:disabled)::-webkit-slider-thumb {
    background: var(--accent-strong);
  }
  .range-marks {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: var(--text-faint);
    font-variant-numeric: tabular-nums;
  }
  .value {
    flex-shrink: 0;
    min-width: 3.6rem;
    padding: 0.15rem 0.45rem;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: var(--surface-2);
    font-size: 0.72rem;
    color: var(--text-dim);
    text-align: center;
  }

  /* ── Buttons ────────────────────────────────────────────────────────────── */
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.45rem;
    padding: 0.55rem 0.95rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    background: var(--surface-2);
    color: var(--text);
    font-size: 0.86rem;
    font-weight: 500;
    white-space: nowrap;
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
  }
  .btn:hover:not(:disabled) {
    background: var(--surface-3);
    border-color: var(--line-strong);
  }
  .btn:disabled {
    opacity: 0.5;
  }
  .btn-quiet {
    padding: 0.3rem 0.55rem;
    border-color: transparent;
    background: transparent;
    color: var(--text-faint);
    font-size: 0.74rem;
  }
  .btn-quiet:hover:not(:disabled) {
    color: var(--text);
    border-color: var(--line-strong);
    background: var(--surface-3);
  }
  .btn-quiet :global(svg) {
    transform: rotate(90deg);
  }

  .synth-btn {
    width: 100%;
    margin-top: 1rem;
    padding: 0.7rem 1rem;
    border: 1px solid var(--accent);
    background: var(--accent);
    color: #0b0d13;
    font-weight: 600;
    font-size: 0.92rem;
  }
  .synth-btn:hover:not(:disabled) {
    background: var(--accent-strong);
    border-color: var(--accent-strong);
  }
  .synth-btn.stop {
    border-color: rgba(255, 118, 118, 0.6);
    background: #b93b3a;
    color: #fff;
  }
  .synth-btn.stop:hover:not(:disabled) {
    background: #cf4645;
    border-color: rgba(255, 118, 118, 0.8);
  }
  .btn-spinner {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 2px solid rgba(11, 13, 19, 0.3);
    border-top-color: #0b0d13;
    animation: spin 0.8s linear infinite;
  }
  .synth-btn.stop .btn-spinner {
    border-color: rgba(255, 255, 255, 0.35);
    border-top-color: #fff;
  }

  /* ── Simulated LLM text ─────────────────────────────────────────────────── */
  .sim-display {
    overflow-y: auto;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    cursor: default;
  }
  .sim-sent {
    color: var(--text-faint);
  }
  .sim-pending {
    color: var(--text);
  }

  /* ── Output panel ───────────────────────────────────────────────────────── */
  .output-panel {
    padding: 0;
    overflow: hidden;
  }
  .output-panel.synthesising {
    border-color: rgba(139, 155, 255, 0.25);
  }
  .output-panel .panel-head {
    margin: 0;
    padding: 0.8rem 1rem;
    border-bottom: 1px solid var(--line);
  }
  .output-body {
    padding: 1.15rem 1rem 1.25rem;
    min-height: 9rem;
    justify-content: center;
  }

  .player {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .player-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    font-size: 0.76rem;
  }
  .player-file {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    min-width: 0;
    color: var(--text-dim);
  }
  .player-name {
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }
  .player audio {
    display: block;
    width: 100%;
    height: 38px;
    border-radius: var(--radius-sm);
  }

  /* ── Empty / loading states ─────────────────────────────────────────────── */
  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    min-height: 6.5rem;
    text-align: center;
    color: var(--text-faint);
  }
  .empty p {
    font-size: 0.9rem;
    color: var(--text-dim);
  }
  .empty small {
    font-size: 0.76rem;
    max-width: 34ch;
    line-height: 1.5;
  }
  .empty-icon {
    display: grid;
    place-items: center;
    width: 38px;
    height: 38px;
    margin-bottom: 0.15rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    color: var(--text-faint);
  }

  .spinner {
    width: 24px;
    height: 24px;
    margin-bottom: 0.2rem;
    border-radius: 50%;
    border: 2px solid var(--surface-3);
    border-top-color: var(--accent);
    animation: spin 0.85s linear infinite;
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  @keyframes breathe {
    50% { opacity: 0.35; }
  }

  /* ── Badges & alerts ────────────────────────────────────────────────────── */
  .badge {
    padding: 0.05rem 0.4rem;
    border-radius: 4px;
    background: var(--surface-3);
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0;
    color: var(--text-dim);
    font-variant-numeric: tabular-nums;
  }
  .badge-live {
    color: #ffa8a8;
    background: rgba(255, 118, 118, 0.14);
    text-transform: uppercase;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    animation: blink 1.6s ease-in-out infinite;
  }
  @keyframes blink {
    50% { opacity: 0.45; }
  }

  .alert {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    margin-top: 0.85rem;
    padding: 0.6rem 0.75rem;
    border: 1px solid rgba(255, 118, 118, 0.28);
    border-radius: var(--radius-sm);
    background: rgba(255, 118, 118, 0.07);
    color: #ffb4b4;
    font-size: 0.8rem;
    line-height: 1.45;
  }
  main > .alert {
    margin: 0;
  }
  .alert :global(svg) {
    flex-shrink: 0;
    margin-top: 0.1rem;
  }
  .alert-dismiss {
    display: grid;
    place-items: center;
    margin-left: auto;
    padding: 0.15rem;
    flex-shrink: 0;
    border: none;
    border-radius: 3px;
    background: transparent;
    color: inherit;
    opacity: 0.7;
  }
  .alert-dismiss:hover {
    opacity: 1;
    background: rgba(255, 255, 255, 0.08);
  }

  /* ── Footer ─────────────────────────────────────────────────────────────── */
  .footer {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    margin-top: auto;
    padding-top: 1.6rem;
    font-size: 0.74rem;
  }
  .footer-link {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    color: var(--text-faint);
    text-decoration: none;
    transition: color 0.15s ease;
  }
  .footer-link:hover {
    color: var(--text-dim);
  }
  .footer-icon {
    width: 14px;
    height: 14px;
    flex-shrink: 0;
  }
  .footer-emoji {
    font-size: 13px;
    line-height: 1;
  }
  .footer-sep {
    color: var(--text-faint);
    opacity: 0.5;
  }

  /* ── Responsive ─────────────────────────────────────────────────────────── */
  @media (max-width: 760px) {
    .shell {
      padding: 1.25rem 0.85rem 1.5rem;
    }
    .panel {
      padding: 0.9rem;
      border-radius: var(--radius-md);
    }
    .output-panel {
      padding: 0;
    }
    .mode-hint,
    .seg-hint {
      display: none;
    }
    .settings-row,
    .design-grid {
      grid-template-columns: 1fr;
    }
    .drop-hint {
      display: none;
    }
    .counter {
      display: none;
    }
  }
</style>

<script lang="ts">
  import type { Voice } from './types.js'
  import { deleteVoice, voicePreviewUrl } from './api.js'

  interface Props {
    voices: Voice[]
    value: string | null
    onchange?: (id: string | null) => void
    ondelete?: (id: string) => void
  }

  let { voices, value = $bindable(null), onchange, ondelete }: Props = $props()

  let playingId = $state<string | null>(null)
  let confirmingId = $state<string | null>(null)
  let deletingId = $state<string | null>(null)
  let deleteError = $state<string | null>(null)
  let confirmTimer: ReturnType<typeof setTimeout> | null = null
  let audioEl: HTMLAudioElement | undefined = $state()

  function select(id: string | null) {
    value = id
    clearConfirm()
    onchange?.(id)
  }

  function togglePreview(voiceId: string) {
    if (playingId === voiceId) {
      audioEl?.pause()
      playingId = null
    } else {
      if (audioEl) {
        audioEl.pause()
        audioEl.src = ''
      }
      audioEl = new Audio(voicePreviewUrl(voiceId))
      audioEl.onended = () => (playingId = null)
      audioEl.onerror = () => (playingId = null)
      audioEl.play().catch(() => (playingId = null))
      playingId = voiceId
    }
  }

  function clearConfirm() {
    if (confirmTimer) clearTimeout(confirmTimer)
    confirmingId = null
  }

  function requestDelete(voiceId: string) {
    deleteError = null
    if (confirmingId === voiceId) {
      clearConfirm()
      performDelete(voiceId)
    } else {
      if (confirmTimer) clearTimeout(confirmTimer)
      confirmingId = voiceId
      confirmTimer = setTimeout(() => (confirmingId = null), 3000)
    }
  }

  async function performDelete(voiceId: string) {
    deletingId = voiceId
    try {
      await deleteVoice(voiceId)
      if (playingId === voiceId) {
        audioEl?.pause()
        playingId = null
      }
      if (value === voiceId) {
        value = null
        onchange?.(null)
      }
      ondelete?.(voiceId)
    } catch (e) {
      deleteError = e instanceof Error ? e.message : 'Delete failed'
    } finally {
      deletingId = null
    }
  }

  $effect(() => {
    return () => {
      audioEl?.pause()
      if (confirmTimer) clearTimeout(confirmTimer)
    }
  })
</script>

<div class="voice-list">
  {#each voices as voice (voice.id)}
    <div class="voice-row" class:selected={value === voice.id}>
      <button class="voice-select" onclick={() => select(voice.id)} type="button">
        <div class="voice-header">
          <span class="voice-name">👤 {voice.name}</span>
          {#if voice.language}
            <span class="voice-lang">{voice.language}</span>
          {/if}
          <span class="voice-file">{voice.filename}</span>
        </div>
        {#if voice.refText}
          <p class="voice-ref">"{voice.refText}"</p>
        {:else}
          <p class="voice-ref no-transcript">No transcript — Whisper will auto-transcribe on use</p>
        {/if}
      </button>

      <div class="voice-actions">
        <button
          class="action-btn"
          class:playing={playingId === voice.id}
          onclick={() => togglePreview(voice.id)}
          type="button"
          title={playingId === voice.id ? 'Stop preview' : 'Play preview'}
          aria-label={playingId === voice.id ? 'Stop preview' : `Preview ${voice.name}`}
        >
          {playingId === voice.id ? '⏹' : '▶'}
        </button>
        <button
          class="action-btn delete-btn"
          class:confirming={confirmingId === voice.id}
          onclick={() => requestDelete(voice.id)}
          disabled={deletingId === voice.id}
          type="button"
          title={confirmingId === voice.id ? 'Click again to confirm deletion' : `Delete ${voice.name}`}
          aria-label={confirmingId === voice.id ? 'Confirm delete' : `Delete ${voice.name}`}
        >
          {#if deletingId === voice.id}
            <span class="delete-spinner"></span>
          {:else if confirmingId === voice.id}
            Sure?
          {:else}
            🗑
          {/if}
        </button>
      </div>
    </div>
  {/each}

  {#if deleteError}
    <p class="delete-error">✕ {deleteError}</p>
  {/if}
</div>

<style>
  .voice-list {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  /* ── Voice rows ── */
  .voice-row {
    display: flex;
    align-items: stretch;
    gap: 0;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    transition: border-color 0.15s;
  }

  .voice-row:hover {
    border-color: color-mix(in srgb, var(--primary) 50%, var(--border));
  }

  .voice-row.selected {
    border-color: var(--primary);
    background: color-mix(in srgb, var(--primary) 8%, var(--surface-2));
  }

  .voice-select {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding: 0.6rem 0.875rem;
    background: none;
    border: none;
    color: var(--text);
    text-align: left;
    cursor: pointer;
    min-width: 0;
  }

  .voice-header {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    min-width: 0;
  }

  .voice-name {
    font-size: 0.875rem;
    font-weight: 600;
    white-space: nowrap;
  }

  .voice-lang {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--primary);
    background: color-mix(in srgb, var(--primary) 12%, transparent);
    border: 1px solid color-mix(in srgb, var(--primary) 30%, transparent);
    border-radius: 4px;
    padding: 0.05em 0.4em;
    flex-shrink: 0;
  }

  .voice-file {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: monospace;
    white-space: nowrap;
  }

  .voice-ref {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-style: italic;
    margin: 0;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
  }

  .voice-ref.no-transcript {
    color: color-mix(in srgb, var(--text-muted) 60%, transparent);
    font-style: normal;
  }

  /* ── Action buttons ── */
  .voice-actions {
    display: flex;
    flex-direction: column;
    border-left: 1px solid var(--border);
    flex-shrink: 0;
  }

  .action-btn {
    flex: 1;
    width: 42px;
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 0.75rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.12s, color 0.12s;
    padding: 0;
  }

  .action-btn:hover {
    background: var(--surface-3);
    color: var(--text);
  }

  .action-btn + .action-btn {
    border-top: 1px solid var(--border);
  }

  .action-btn.playing {
    color: var(--primary);
    animation: pulse 1.2s ease-in-out infinite;
  }

  .delete-btn:hover {
    color: var(--error);
  }

  .delete-btn.confirming {
    background: color-mix(in srgb, var(--error) 15%, var(--surface-3));
    color: var(--error);
    font-size: 0.65rem;
    font-weight: 700;
  }

  .delete-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .delete-spinner {
    width: 12px;
    height: 12px;
    border: 2px solid color-mix(in srgb, var(--text-muted) 40%, transparent);
    border-top-color: var(--text-muted);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  .delete-error {
    font-size: 0.8rem;
    color: var(--error);
    margin: 0;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>

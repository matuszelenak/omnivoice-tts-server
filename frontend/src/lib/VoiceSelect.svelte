<script lang="ts">
  import Icon from './Icon.svelte'
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
    <div
      class="voice-row"
      class:selected={value === voice.id}
      class:playing={playingId === voice.id}
    >
      <button class="voice-select" onclick={() => select(voice.id)} type="button">
        <span class="voice-check">
          {#if value === voice.id}
            <Icon name="check" size={13} />
          {:else}
            <Icon name="user" size={13} />
          {/if}
        </span>
        <span class="voice-body">
          <span class="voice-header">
            <span class="voice-name">{voice.name}</span>
            {#if voice.language}
              <span class="voice-lang">{voice.language}</span>
            {/if}
            <span class="voice-file mono">{voice.filename}</span>
          </span>
          {#if voice.refText}
            <span class="voice-ref">“{voice.refText}”</span>
          {:else}
            <span class="voice-ref no-transcript">No transcript — ASR fills it in on first use</span>
          {/if}
        </span>
      </button>

      <div class="voice-actions">
        <button
          class="action-btn"
          class:active={playingId === voice.id}
          onclick={() => togglePreview(voice.id)}
          type="button"
          title={playingId === voice.id ? 'Stop preview' : 'Play preview'}
          aria-label={playingId === voice.id ? 'Stop preview' : `Preview ${voice.name}`}
        >
          <Icon name={playingId === voice.id ? 'stop' : 'play'} size={13} />
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
            <span class="confirm-text">Sure?</span>
          {:else}
            <Icon name="trash" size={13} />
          {/if}
        </button>
      </div>
    </div>
  {/each}

  {#if deleteError}
    <p class="delete-error"><Icon name="alert" size={14} /><span>{deleteError}</span></p>
  {/if}
</div>

<style>
  .voice-list {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .voice-row {
    display: flex;
    align-items: stretch;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    background: var(--surface-2);
    overflow: hidden;
    transition: border-color 0.15s ease, background 0.15s ease;
  }
  .voice-row:hover {
    border-color: var(--line-strong);
  }
  .voice-row.selected {
    border-color: rgba(139, 155, 255, 0.45);
    background: rgba(139, 155, 255, 0.06);
  }

  .voice-select {
    flex: 1;
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    padding: 0.55rem 0.7rem;
    border: none;
    background: transparent;
    color: var(--text);
    text-align: left;
    min-width: 0;
  }

  .voice-check {
    display: grid;
    place-items: center;
    width: 20px;
    height: 20px;
    flex-shrink: 0;
    margin-top: 0.05rem;
    border-radius: 4px;
    background: var(--surface-3);
    color: var(--text-faint);
  }
  .voice-row.selected .voice-check {
    background: rgba(139, 155, 255, 0.16);
    color: var(--accent-strong);
  }

  .voice-body {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    min-width: 0;
  }

  .voice-header {
    display: flex;
    align-items: baseline;
    gap: 0.45rem;
    min-width: 0;
  }
  .voice-name {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: -0.005em;
    white-space: nowrap;
  }
  .voice-lang {
    padding: 0.02rem 0.3rem;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.04);
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-dim);
    flex-shrink: 0;
  }
  .voice-file {
    font-size: 0.7rem;
    color: var(--text-faint);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .voice-ref {
    font-size: 0.76rem;
    color: var(--text-faint);
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
  }

  .voice-actions {
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    border-left: 1px solid var(--line);
  }
  .action-btn {
    flex: 1;
    width: 34px;
    padding: 0;
    display: grid;
    place-items: center;
    border: none;
    background: transparent;
    color: var(--text-faint);
    transition: background 0.12s ease, color 0.12s ease;
  }
  .action-btn:hover:not(:disabled) {
    background: var(--surface-3);
    color: var(--text);
  }
  .action-btn + .action-btn {
    border-top: 1px solid var(--line);
  }
  .action-btn.active {
    color: var(--accent-strong);
  }
  .delete-btn:hover:not(:disabled) {
    color: var(--danger);
  }
  .delete-btn.confirming {
    background: rgba(255, 118, 118, 0.14);
    color: var(--danger);
  }
  .delete-btn:disabled {
    opacity: 0.5;
  }
  .confirm-text {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.02em;
  }

  .delete-spinner {
    width: 11px;
    height: 11px;
    border: 2px solid var(--surface-3);
    border-top-color: var(--text-faint);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  .delete-error {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    color: #ffb4b4;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>

<script lang="ts">
  import type { Voice } from './types.js'
  import { deleteVoice } from './api.js'

  interface Props {
    voices: Voice[]
    value: string | null
    onchange?: (id: string | null) => void
    ondelete?: (id: string) => void
  }

  let { voices, value = $bindable(null), onchange, ondelete }: Props = $props()

  let confirmingId = $state<string | null>(null)
  let deletingId = $state<string | null>(null)
  let deleteError = $state<string | null>(null)
  let confirmTimer: ReturnType<typeof setTimeout> | null = null

  function select(id: string) {
    value = id
    clearConfirm()
    onchange?.(id)
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
      if (confirmTimer) clearTimeout(confirmTimer)
    }
  })
</script>

<div class="voice-list">
  {#each voices as voice (voice.id)}
    <div class="voice-row" class:selected={value === voice.id}>
      <button class="voice-select" onclick={() => select(voice.id)} type="button">
        <span class="voice-name">{voice.name}</span>
        <span class="voice-kind" class:custom={voice.kind === 'custom'}>
          {voice.kind === 'custom' ? 'custom' : 'built-in'}
        </span>
      </button>

      {#if voice.kind === 'custom'}
        <div class="voice-actions">
          <button
            class="delete-btn"
            class:confirming={confirmingId === voice.id}
            onclick={() => requestDelete(voice.id)}
            disabled={deletingId === voice.id}
            type="button"
            title={confirmingId === voice.id ? 'Click again to confirm deletion' : `Delete ${voice.name}`}
            aria-label={confirmingId === voice.id ? 'Confirm delete' : `Delete ${voice.name}`}
          >
            {#if deletingId === voice.id}
              <span class="spinner"></span>
            {:else if confirmingId === voice.id}
              Sure?
            {:else}
              🗑
            {/if}
          </button>
        </div>
      {/if}
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

  .voice-row {
    display: flex;
    align-items: stretch;
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
    align-items: center;
    gap: 0.6rem;
    padding: 0.55rem 0.875rem;
    background: none;
    border: none;
    color: var(--text);
    text-align: left;
    cursor: pointer;
  }

  .voice-name {
    font-size: 0.875rem;
    font-weight: 600;
  }

  .voice-kind {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    background: color-mix(in srgb, var(--text-muted) 12%, transparent);
    border: 1px solid color-mix(in srgb, var(--text-muted) 25%, transparent);
    border-radius: 4px;
    padding: 0.05em 0.4em;
  }

  .voice-kind.custom {
    color: var(--primary);
    background: color-mix(in srgb, var(--primary) 12%, transparent);
    border-color: color-mix(in srgb, var(--primary) 30%, transparent);
  }

  .voice-actions {
    display: flex;
    border-left: 1px solid var(--border);
    flex-shrink: 0;
  }

  .delete-btn {
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

  .delete-btn:hover {
    background: var(--surface-3);
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

  .spinner {
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

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>

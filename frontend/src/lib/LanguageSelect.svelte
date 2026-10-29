<script lang="ts">
  import Icon from './Icon.svelte'
  import type { Language } from './types.js'

  interface Props {
    languages: Language[]
    value: string
    onchange?: (id: string) => void
  }

  let { languages, value = $bindable('en'), onchange }: Props = $props()

  let search = $state('')
  let open = $state(false)
  let inputEl: HTMLInputElement | undefined = $state()

  let selectedName = $derived(languages.find((l) => l.id === value)?.name ?? value)

  let filtered = $derived(
    search.trim()
      ? languages
          .filter(
            (l) =>
              l.name.toLowerCase().includes(search.toLowerCase()) ||
              l.id.toLowerCase().startsWith(search.toLowerCase()),
          )
          .slice(0, 80)
      : languages.slice(0, 80),
  )

  function select(lang: Language) {
    value = lang.id
    search = ''
    open = false
    onchange?.(lang.id)
  }

  function handleInput(e: Event) {
    search = (e.target as HTMLInputElement).value
    open = true
  }

  function handleFocus() {
    open = true
    search = ''
  }

  function handleBlur() {
    setTimeout(() => {
      open = false
      search = ''
    }, 150)
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      open = false
      search = ''
      inputEl?.blur()
    }
  }
</script>

<div class="language-select">
  <input
    bind:this={inputEl}
    type="text"
    class="select-input"
    value={open ? search : selectedName}
    oninput={handleInput}
    onfocus={handleFocus}
    onblur={handleBlur}
    onkeydown={handleKeydown}
    placeholder="Search languages…"
    autocomplete="off"
    spellcheck="false"
    role="combobox"
    aria-haspopup="listbox"
    aria-expanded={open}
    aria-controls="lang-listbox"
  />
  <span class="select-arrow" aria-hidden="true"><Icon name="chevron" size={14} /></span>
  {#if open && filtered.length > 0}
    <div id="lang-listbox" class="dropdown" role="listbox">
      {#each filtered as lang (lang.id)}
        <button
          type="button"
          class="dropdown-item"
          class:active={lang.id === value}
          role="option"
          aria-selected={lang.id === value}
          onmousedown={(e) => {
            e.preventDefault()
            select(lang)
          }}
        >
          <span class="lang-name">{lang.name}</span>
          <span class="lang-code mono">{lang.id}</span>
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .language-select {
    position: relative;
  }

  .select-input {
    width: 100%;
    padding: 0.5rem 2rem 0.5rem 0.7rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    background: var(--surface-2);
    color: var(--text);
    font-size: 0.86rem;
    cursor: pointer;
    transition: border-color 0.15s ease;
  }
  .select-input:hover {
    border-color: var(--line-strong);
  }
  .select-input:focus {
    outline: none;
    border-color: rgba(139, 155, 255, 0.55);
    cursor: text;
  }
  .select-input::placeholder {
    color: var(--text-faint);
  }

  .select-arrow {
    position: absolute;
    top: 50%;
    right: 0.65rem;
    display: grid;
    place-items: center;
    transform: translateY(-50%);
    color: var(--text-faint);
    pointer-events: none;
  }

  .dropdown {
    position: absolute;
    top: calc(100% + 4px);
    right: 0;
    left: 0;
    z-index: 100;
    max-height: 16rem;
    overflow-y: auto;
    border: 1px solid var(--line-strong);
    border-radius: var(--radius-md);
    background: var(--surface-2);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.55);
  }

  .dropdown-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    width: 100%;
    padding: 0.45rem 0.7rem;
    border: none;
    background: transparent;
    color: var(--text-dim);
    font-size: 0.84rem;
    text-align: left;
    transition: background 0.1s ease, color 0.1s ease;
  }
  .dropdown-item:hover,
  .dropdown-item.active {
    background: var(--surface-3);
    color: var(--text);
  }
  .dropdown-item.active .lang-name {
    color: var(--accent-strong);
  }

  .lang-code {
    flex-shrink: 0;
    font-size: 0.7rem;
    color: var(--text-faint);
  }
</style>

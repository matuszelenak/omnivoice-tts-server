<script lang="ts">
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
  <span class="select-arrow" aria-hidden="true">▾</span>
  {#if open && filtered.length > 0}
    <div id="lang-listbox" class="dropdown" role="listbox">
      {#each filtered as lang (lang.id)}
        <button
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
          <span class="lang-code">{lang.id}</span>
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
    padding: 0.6rem 2.5rem 0.6rem 0.875rem;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-size: 0.9rem;
    box-sizing: border-box;
    cursor: pointer;
    transition: border-color 0.15s;
  }

  .select-input:focus {
    outline: none;
    border-color: var(--primary);
    cursor: text;
  }

  .select-arrow {
    position: absolute;
    right: 0.75rem;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-muted);
    pointer-events: none;
    font-size: 0.8rem;
  }

  .dropdown {
    position: absolute;
    top: calc(100% + 4px);
    left: 0;
    right: 0;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    max-height: 260px;
    overflow-y: auto;
    z-index: 100;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
  }

  .dropdown-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    padding: 0.5rem 0.875rem;
    background: none;
    border: none;
    color: var(--text);
    font-size: 0.875rem;
    text-align: left;
    cursor: pointer;
    transition: background 0.1s;
  }

  .dropdown-item:hover,
  .dropdown-item.active {
    background: var(--surface-3);
  }

  .dropdown-item.active .lang-name {
    color: var(--primary);
  }

  .lang-code {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: monospace;
    margin-left: 0.5rem;
    flex-shrink: 0;
  }
</style>

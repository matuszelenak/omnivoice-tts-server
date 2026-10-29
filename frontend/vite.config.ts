import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import wasm from 'vite-plugin-wasm'

// Where the dev server forwards same-origin API calls. On the host this is the
// server's published port; inside docker compose it is the `server` service.
function apiTarget(): string {
  // @ts-ignore - Deno global is present when running under Deno.
  if (typeof Deno !== 'undefined') {
    // @ts-ignore
    const fromDeno = Deno.env.get('API_PROXY_TARGET')
    if (fromDeno) return fromDeno
  }
  // @ts-ignore - process is provided via Node compat.
  return globalThis.process?.env?.API_PROXY_TARGET ?? 'http://localhost:9001'
}

const target = apiTarget()

export default defineConfig({
  plugins: [wasm(), svelte()],
  // Keep top-level await and modern syntax in dependencies (tiktoken's wasm
  // glue) instead of downlevelling them with a swc-based plugin - that plugin
  // breaks whenever @swc/core ships a new API.
  build: { target: 'esnext' },
  server: {
    allowedHosts: ['omnivoice-dev.the-killer.app'],
    host: '0.0.0.0',
    port: 5173,
    // Polling makes file changes reliable inside containers / bind mounts,
    // which is what `docker compose watch` relies on.
    watch: { usePolling: true },
    proxy: {
      '/v1': {
        target: target,
        changeOrigin: true,
        ws: true
      },
      '/health': { target, changeOrigin: true },
    },
  },
})

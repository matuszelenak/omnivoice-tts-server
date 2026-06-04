import { get_encoding } from 'tiktoken'

/** Send *text* token-by-token over *ws* at *tokensPerSecond*, simulating LLM output.
 *
 * Tokenizes with GPT-4's cl100k_base encoding. Reports the cumulative number of
 * characters sent via *onProgress* after each token. Stops early if the socket
 * closes. Does NOT send the end-of-stream sentinel — that is the caller's job.
 */
export async function streamTokens(
  ws: WebSocket,
  text: string,
  tokensPerSecond: number,
  onProgress: (sentChars: number) => void,
): Promise<void> {
  const enc = get_encoding('cl100k_base')
  const utf8 = new TextDecoder('utf-8', { fatal: false })
  const msPerToken = 1000 / tokensPerSecond
  let sent = 0
  try {
    for (const id of enc.encode(text)) {
      if (ws.readyState !== WebSocket.OPEN) break
      const tokenText = utf8.decode(enc.decode_single_token_bytes(id))
      ws.send(tokenText)
      sent += tokenText.length
      onProgress(sent)
      await new Promise((r) => setTimeout(r, msPerToken))
    }
  } finally {
    enc.free()
  }
}

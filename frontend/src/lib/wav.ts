/** Concatenate WAV chunks into a single playable blob.
 *
 * Strips the 44-byte header from every chunk but the first and patches the
 * RIFF/data sizes in the retained header.
 */
export function combineWavBuffers(chunks: ArrayBuffer[]): Blob {
  if (chunks.length === 0) return new Blob([], { type: 'audio/wav' })
  if (chunks.length === 1) return new Blob([chunks[0]], { type: 'audio/wav' })

  const HEADER = 44
  const pcmParts = chunks.map((buf) => buf.slice(HEADER))
  const totalPcm = pcmParts.reduce((s, p) => s + p.byteLength, 0)

  const header = chunks[0].slice(0, HEADER)
  const view = new DataView(header)
  view.setUint32(4, 36 + totalPcm, true)
  view.setUint32(40, totalPcm, true)

  return new Blob([header, ...pcmParts], { type: 'audio/wav' })
}

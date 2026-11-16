/** Serialize press/release across network latency. A tap moves once; a hold
 * starts continuous motion only while the original press is still active. */
export function createHoldGesture({ step, begin, stop, onHolding, onError }) {
  let pressed = false, task = null, release
  return {
    start(direction) {
      if (task) return task
      pressed = true
      onHolding(direction)
      const released = new Promise(resolve => { release = resolve })
      task = (async () => {
        try {
          await step(direction)
          if (pressed) await begin(direction)
          await released
        } catch (error) {
          onError(error)
        } finally {
          pressed = false
          try { await stop() } catch (error) { onError(error) }
          onHolding('')
          task = null
          release = null
        }
      })()
      return task
    },
    end() {
      pressed = false
      release?.()
      return task || Promise.resolve()
    },
  }
}

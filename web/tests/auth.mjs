import assert from 'node:assert/strict'
import { api, getToken, setToken, observeAuthFailures } from '../src/api.js'

const storage = new Map()
globalThis.localStorage = {
  getItem: key => storage.get(key) || null,
  setItem: (key, value) => storage.set(key, value),
  removeItem: key => storage.delete(key),
}
globalThis.location = new URL('https://prism.example/')
const calls = [], notices = []
let forbidden = false, leaseConflict = false
let pending = null
const json = (data, status = 200) => new Response(JSON.stringify(data), { status })
const transport = async (path, options = {}) => {
  calls.push({ path, options })
  if (pending) return pending
  if (leaseConflict) return json({ code: 'lease_mismatch', message: '租约已变化' }, 403)
  if (forbidden) return json({ code: 'forbidden_token', message: '令牌无效，请重新登录' }, 403)
  return json(path === '/api/v1/auth' ? { authenticated: true } : { id: 'sess_test' })
}
globalThis.fetch = transport
const stop = observeAuthFailures(message => notices.push(message))

await assert.rejects(api.createSession({}), error => error.code === 'token_required')
assert.equal(calls.length, 0, 'bare URL must not send even a validation request')
assert.match(notices.at(-1), /请输入/)
const missingFrame = await fetch('/api/v1/sessions/sess_test/frame')
assert.equal(missingFrame.status, 403)
assert.equal(calls.length, 0, 'direct panel requests also wait for a token')

setToken(' wrong ')
forbidden = true
await assert.rejects(api.createSession({}), error => error.status === 403 && /重新登录/.test(error.message))
assert.deepEqual(calls.map(call => call.path), ['/api/v1/auth'])
assert.equal(calls[0].options.headers['X-Console-Token'], 'wrong')
assert.match(notices.at(-1), /重新登录/)

calls.length = 0
forbidden = false
setToken(' valid ')
await api.createSession({ seed: 7 })
assert.deepEqual(calls.map(call => call.path), ['/api/v1/auth', '/api/v1/sessions'])
assert.deepEqual(JSON.parse(calls[1].options.body), { config: { seed: 7 } })
assert.equal(calls[0].options.cache, 'no-store')

forbidden = true
for (const endpoint of ['events', 'scene', 'frame', 'commands', 'lease', 'visual-assets/mesh.json']) {
  const before = notices.length
  await fetch(`/api/v1/sessions/sess_test/${endpoint}`, { headers: { 'X-Console-Token': getToken() } })
  assert.equal(notices.length, before + 1, `${endpoint} 403 must reach the login handler`)
}
await fetch('/api/v1/evaluations', { headers: { 'X-Console-Token': getToken() } })
assert.match(notices.at(-1), /重新登录/)

leaseConflict = true
const beforeLease = notices.length
await assert.rejects(api.command('sess_test', {}), error => error.code === 'lease_mismatch' && error.message === '租约已变化')
assert.equal(notices.length, beforeLease, 'a control lease conflict must not log the operator out')
leaseConflict = false

let resolvePending
pending = new Promise(resolve => { resolvePending = resolve })
const staleRequest = api.events('sess_test')
setToken('new-valid')
const beforeStale = notices.length
resolvePending(json({ code: 'forbidden_token' }, 403))
await assert.rejects(staleRequest)
assert.equal(notices.length, beforeStale, 'an old response must not invalidate a new credential')
pending = null

const beforeExternal = notices.length
await fetch('https://external.example/api/v1/auth')
await fetch('/api/v1/health')
assert.equal(notices.length, beforeExternal, 'only protected same-origin APIs trigger login')
stop()
assert.equal(globalThis.fetch, transport, 'unmount must restore fetch')
console.log('Auth API regression checks passed')

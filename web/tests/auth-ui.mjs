import assert from 'node:assert/strict'
import { createServer } from 'vite'
import vue from '@vitejs/plugin-vue'
import { chromium } from 'playwright'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const webRoot = fileURLToPath(new URL('..', import.meta.url))
const cacheDir = await mkdtemp(join(tmpdir(), 'prism-auth-vite-'))
const server = await createServer({
  configFile: false, root: webRoot, cacheDir,
  plugins: [vue()], esbuild: false,
  resolve: { preserveSymlinks: true, alias: [{ find: /^vue$/, replacement: fileURLToPath(new URL('../node_modules/vue/dist/vue.runtime.esm-browser.prod.js', import.meta.url)) }] },
  optimizeDeps: { noDiscovery: true, include: [] },
  server: { host: '127.0.0.1', port: 0 },
})
let browser
try {
  await server.listen()
  const base = `http://127.0.0.1:${server.httpServer.address().port}`
  browser = await chromium.launch({ channel: 'chrome', headless: true })
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } })
  const page = await context.newPage()
  const errors = [], calls = []
  page.on('pageerror', error => errors.push(error.message))
  await page.addInitScript(() => {
    // The HTTP fallback must keep working without a WebSocket connection.
    window.WebSocket = class { close() {} }
  })
  let failEndpoint = null
  let commandConflict = false
  const session = {
    id: 'sess_auth_test', state: 'READY', live: true, fencing_token: 1, state_version: 1,
    config: { env_id: 'robocasa/PickPlaceCounterToCabinet', seed: 0, control_freq: 20, policy: 'hold' },
    environment: { step: 0, sim_seconds: 0, capabilities: { jog: true, gripper: true } },
  }
  await page.route('**/api/v1/**', async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    const token = request.headers()['x-console-token']
    calls.push({ path, method: request.method(), token })
    const reply = (data, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(data) })
    if (token !== 'valid-test-token' || path === failEndpoint) return reply({ code: 'forbidden_token', message: '令牌无效，请重新登录' }, 403)
    if (path === '/api/v1/auth') return reply({ authenticated: true })
    if (path === '/api/v1/sessions' && request.method() === 'POST') return reply(session, 201)
    if (path.endsWith('/commands')) {
      if (commandConflict) return reply({ code: 'lease_mismatch', message: '租约已变化' }, 403)
      return reply({ acknowledgement: 'accepted' })
    }
    if (path.endsWith('/events')) return reply({ events: [] })
    if (path.endsWith('/scene')) return reply({ objects: [], joints: [], geometry_key: 'test', geometry: { assets: [], geoms: [] } })
    if (path.endsWith('/evaluations')) return reply({ evaluations: [] })
    if (path.endsWith('/frame')) return reply({ code: 'frame_not_found' }, 404)
    return reply(session)
  })
  const input = page.locator('#console-token')
  const submit = page.locator('#console-login button[type=submit]')
  const loggedIn = () => page.waitForFunction(() => !document.querySelector('#console-login'))
  const focused = () => page.waitForFunction(() => document.activeElement?.id === 'console-token')
  const createCalls = () => calls.filter(call => call.path === '/api/v1/sessions' && call.method === 'POST')
  const login = async () => { await input.fill('valid-test-token'); await submit.click(); await loggedIn() }

  await page.goto(base)
  await focused()
  await page.getByRole('button', { name: '界面设置与控制台登录', exact: true }).click()
  await page.getByRole('button', { name: '登录控制台', exact: true }).click()
  await focused()
  await page.getByRole('button', { name: /加载仿真环境/ }).click()
  await focused()
  assert.equal(calls.length, 0, 'bare URL/create click must send no protected requests')
  await input.fill('wrong-token')
  await submit.click()
  await page.waitForFunction(() => document.querySelector('#login-message')?.textContent.includes('重新登录'))
  await focused()
  assert.equal(createCalls().length, 0, 'wrong token must not allocate a session')
  assert.equal(await page.evaluate(() => localStorage.getItem('prism.consoleToken')), null)

  await login()
  await page.getByRole('button', { name: /加载仿真环境/ }).click()
  await page.waitForFunction(() => new URLSearchParams(location.search).get('session') === 'sess_auth_test')
  assert.equal(createCalls().length, 1)
  const createIndex = calls.findIndex(call => call.path === '/api/v1/sessions' && call.method === 'POST')
  assert.equal(calls[createIndex - 1].path, '/api/v1/auth', 'create must be preceded by auth validation')

  commandConflict = true
  await page.getByRole('button', { name: '单步', exact: true }).click()
  await page.waitForFunction(() => document.querySelector('.readout-error')?.textContent.includes('租约已变化'))
  assert.equal(await input.count(), 0, 'lease conflict preserves the login')
  commandConflict = false

  // Auth expiry in a direct scene fetch must open and focus the same inline form.
  failEndpoint = '/api/v1/sessions/sess_auth_test/scene'
  await focused()
  assert.match(await page.locator('#login-message').innerText(), /重新登录/)
  await input.fill('new-entry')
  // A queued background poll must not erase the operator's replacement token.
  await page.evaluate(() => fetch('/api/v1/sessions/sess_auth_test/events'))
  assert.equal(await input.inputValue(), 'new-entry')
  failEndpoint = null
  await login()

  // Session/status and event API failures use the same login path.
  for (const endpoint of ['', '/events']) {
    failEndpoint = `/api/v1/sessions/sess_auth_test${endpoint}`
    await focused()
    failEndpoint = null
    await login()
  }

  failEndpoint = '/api/v1/sessions/sess_auth_test/frame'
  await page.getByRole('tab', { name: '仿真相机' }).focus()
  await page.getByRole('tab', { name: '仿真相机' }).press('Enter')
  await focused()
  failEndpoint = null
  await login()

  failEndpoint = '/api/v1/evaluations'
  await page.getByRole('button', { name: '安全测评', exact: true }).click()
  await focused()
  failEndpoint = null
  await login()

  await page.getByRole('button', { name: '界面设置与控制台登录', exact: true }).click()
  await page.getByRole('button', { name: '更换控制台令牌', exact: true }).click()
  await focused()
  await page.getByRole('button', { name: '取消', exact: true }).click()

  // A fragment replaces a stale saved credential and is removed from the URL.
  await page.evaluate(() => localStorage.setItem('prism.consoleToken', 'stale-token'))
  await page.goto(`${base}/?session=sess_auth_test&auth-test=fragment#token=valid-test-token`)
  await loggedIn()
  await page.waitForFunction(() => document.querySelector('.connection.connected'))
  assert.equal(new URL(page.url()).hash, '')
  assert.equal(await page.evaluate(() => localStorage.getItem('prism.consoleToken')), 'valid-test-token')
  await page.reload()
  await loggedIn()
  await page.waitForFunction(() => document.querySelector('.connection.connected'))
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('button', { name: '更换令牌', exact: true }).click()
  await focused()
  assert.ok(await input.isVisible())
  assert.ok(await page.getByRole('button', { name: '界面设置与控制台登录', exact: true }).isVisible())
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'mobile login must not overflow horizontally')
  assert.deepEqual(errors, [], 'login and reconnect must not cause uncaught browser errors')
  console.log('Auth browser regression checks passed: bare URL, wrong/valid token, create preflight, lease conflict, scene/status/events/frame/evaluation 403, fragment, saved login')
} finally {
  await browser?.close()
  await server.close()
  await rm(cacheDir, { recursive: true, force: true })
}

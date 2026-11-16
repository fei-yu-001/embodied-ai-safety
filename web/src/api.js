const TOKEN_KEY = 'prism.consoleToken'
const authListeners = new Set()
const LOGIN_REQUIRED = '请输入控制台令牌登录'
const LOGIN_EXPIRED = '令牌无效，请重新登录'

function requireToken() {
  if (getToken()) return
  notifyAuthRequired('token_required', '')
  const error = new Error(LOGIN_REQUIRED)
  error.status = 403
  error.code = 'token_required'
  throw error
}

function notifyAuthRequired(code, token) {
  // A response sent with an old credential must not invalidate a new login.
  if (token !== getToken()) return
  for (const listener of authListeners) listener(code === 'token_required' ? LOGIN_REQUIRED : LOGIN_EXPIRED)
}

// Some panels fetch frames/evaluations directly. Observe their 403s too, within
// this app's lifetime, without changing unrelated fetches or control-lease errors.
export function observeAuthFailures(listener) {
  authListeners.add(listener)
  const originalFetch = globalThis.fetch
  const observedFetch = async (input, options = {}) => {
    const url = new URL(typeof input === 'string' ? input : input.url || input.href, location.href)
    if (url.origin !== location.origin || !url.pathname.startsWith('/api/v1/') || url.pathname === '/api/v1/health') {
      return originalFetch(input, options)
    }
    const headers = new Headers(options.headers || input.headers)
    const token = headers.get('X-Console-Token') || ''
    if (!token) {
      notifyAuthRequired('token_required', token)
      return new Response(JSON.stringify({ code: 'token_required', message: LOGIN_REQUIRED }), {
        status: 403, headers: { 'Content-Type': 'application/json' },
      })
    }
    const response = await originalFetch(input, options)
    if (response.status === 403) {
      const data = await response.clone().json().catch(() => null)
      if (data?.code !== 'lease_mismatch') notifyAuthRequired(data?.code, token)
    }
    return response
  }
  globalThis.fetch = observedFetch
  return () => {
    authListeners.delete(listener)
    if (globalThis.fetch === observedFetch) globalThis.fetch = originalFetch
  }
}

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || ''
}

export function setToken(value) {
  if (value.trim()) localStorage.setItem(TOKEN_KEY, value.trim())
  else localStorage.removeItem(TOKEN_KEY)
}

async function request(path, options = {}) {
  requireToken()
  const headers = { ...(options.headers || {}) }
  const token = getToken()
  if (token) headers['X-Console-Token'] = token
  if (options.body) headers['Content-Type'] = 'application/json'
  const response = await fetch(path, { ...options, headers })
  const text = await response.text()
  let data = null
  if (text) {
    try {
      data = JSON.parse(text)
    } catch {
      data = { raw: text }
    }
  }
  if (!response.ok) {
    const message = response.status === 403 && data?.code !== 'lease_mismatch'
      ? LOGIN_EXPIRED : (data && (data.message || data.detail)) || `HTTP ${response.status}`
    const error = new Error(message)
    error.status = response.status
    error.code = data && data.code
    throw error
  }
  return data
}

export const api = {
  verifyAuth() {
    return request('/api/v1/auth', { cache: 'no-store' })
  },
  async createSession(config) {
    await api.verifyAuth()
    return request('/api/v1/sessions', { method: 'POST', body: JSON.stringify({ config }) })
  },
  getSession(id) {
    return request(`/api/v1/sessions/${id}`)
  },
  events(id, after = 0) {
    return request(`/api/v1/sessions/${id}/events?after=${after}`)
  },
  command(id, body) {
    return request(`/api/v1/sessions/${id}/commands`, { method: 'POST', body: JSON.stringify(body) })
  },
  acquireLease(id, holder, ttlMs) {
    return request(`/api/v1/sessions/${id}/lease`, {
      method: 'POST',
      body: JSON.stringify({ holder, ttl_ms: ttlMs }),
    })
  },
  renewLease(id, leaseId, ttlMs) {
    return request(`/api/v1/sessions/${id}/lease/${leaseId}/renew`, {
      method: 'POST',
      body: JSON.stringify({ ttl_ms: ttlMs }),
    })
  },
  releaseLease(id, leaseId) {
    return request(`/api/v1/sessions/${id}/lease/${leaseId}`, { method: 'DELETE' })
  },
  frameUrl(id, seq) {
    return seq ? `/api/v1/sessions/${id}/frame?seq=${seq}` : `/api/v1/sessions/${id}/frame`
  },
  streamUrl(id, token) {
    const query = new URLSearchParams({ token })
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
    return `${scheme}://${location.host}/api/v1/sessions/${id}/ws?${query}`
  },
}

export function envelope(session, command, args) {
  const now = new Date()
  const body = {
    command_id: crypto.randomUUID(),
    fencing_token: session.fencing_token,
    command,
    issued_at: now.toISOString(),
    expires_at: new Date(now.getTime() + 60_000).toISOString(),
    expected_state_version: session.state_version,
  }
  if (args) body.args = args
  return body
}

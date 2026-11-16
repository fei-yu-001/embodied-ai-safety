const TOKEN_KEY = 'prism.consoleToken'

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || ''
}

export function setToken(value) {
  localStorage.setItem(TOKEN_KEY, value.trim())
}

async function request(path, options = {}) {
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
    const error = new Error((data && (data.message || data.detail)) || `HTTP ${response.status}`)
    error.status = response.status
    error.code = data && data.code
    throw error
  }
  return data
}

export const api = {
  createSession(config) {
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

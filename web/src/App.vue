<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { api, envelope, getToken, setToken, observeAuthFailures } from './api.js'
import { createHoldGesture } from './holdGesture.js'
import { getJogProfile } from './jogSettings.js'
import Viewport from './components/Viewport.vue'
import Scene3D from './components/Scene3D.vue'
import SceneTree from './components/SceneTree.vue'
import EvaluationPanel from './components/EvaluationPanel.vue'
import CommandBar from './components/CommandBar.vue'
import JogPad from './components/JogPad.vue'
import StatusPanel from './components/StatusPanel.vue'
import ToolSafetyPanel from './components/ToolSafetyPanel.vue'
import TelemetryPanel from './components/TelemetryPanel.vue'
import LiquidGlass from './components/LiquidGlass.vue'
import LiquidTabs from './components/LiquidTabs.vue'

const TTL_MS = 1500
const DIRECTIONS = {
  'x+': [1, 0, 0],
  'x-': [-1, 0, 0],
  'y+': [0, 1, 0],
  'y-': [0, -1, 0],
  'z+': [0, 0, 1],
  'z-': [0, 0, -1],
}

const token = ref(getToken())
const authenticated = ref(false)
const authChecking = ref(false)
const loginOpen = ref(!token.value)
const loginMessage = ref('请输入控制台令牌登录')
const tokenInput = ref(null)

function openLogin() {
  loginOpen.value = true
  nextTick(() => tokenInput.value?.focus())
}

function requireLogin(message) {
  const firstFailure = authenticated.value || !loginOpen.value || Boolean(getToken())
  authenticated.value = false
  setToken('')
  // Preserve an in-progress entry when a background poll reports the same failure.
  if (firstFailure) token.value = ''
  if (firstFailure || !loginMessage.value) loginMessage.value = message
  sceneStale.value = true
  disconnectStream()
  clearInterval(keepaliveTimer)
  keepaliveTimer = null
  endHold()
  if (firstFailure) openLogin()
  else loginOpen.value = true
}

const stopAuthObserver = observeAuthFailures(requireLogin)

async function login() {
  if (authChecking.value) return
  if (!token.value.trim()) {
    loginMessage.value = '请输入控制台令牌登录'
    openLogin()
    return
  }
  authChecking.value = true
  authenticated.value = false
  setToken(token.value)
  try {
    await api.verifyAuth()
    authenticated.value = true
    loginOpen.value = false
    loginMessage.value = ''
    error.value = ''
    if (session.value) {
      connectStream()
      await refresh()
      await pullEvents()
    } else if (attachId.value) await attach()
  } catch (failure) {
    authenticated.value = false
    loginMessage.value = failure.message
    openLogin()
  } finally {
    authChecking.value = false
    if (loginOpen.value) {
      await nextTick()
      tokenInput.value?.focus()
    }
  }
}
const settingsOpen = ref(false)
const settingsButton = ref(null)
function closeSettings() { settingsOpen.value = false; queueMicrotask(() => settingsButton.value?.focus()) }
const reduceTransparency = ref(localStorage.getItem('prism.reduceTransparency') === 'true')
const frostedGlass = ref(localStorage.getItem('prism.frostedGlass') === 'true')
watch(reduceTransparency, value => localStorage.setItem('prism.reduceTransparency', String(value)))
watch(frostedGlass, value => localStorage.setItem('prism.frostedGlass', String(value)))
const viewItems = [{ value: 'scene3d', label: '三维场景' }, { value: 'camera', label: '仿真相机' }]
const taskName = computed(() => (session.value?.config?.env_id || form.env_id).replace('robocasa/', ''))
const stateText = computed(() => ({READY:'就绪',PAUSED:'已暂停',RUNNING:'运行中',CLOSED:'已结束',LOADING:'加载中',FAILED:'加载失败',INTERRUPTED:'已中断'}[session.value?.state] || '尚未连接'))
const session = ref(null)
const events = ref([])
const error = ref('')
const busy = ref(false)
const holding = ref('')
const jogProfileId = ref('standard')
let activeJogProfile = getJogProfile('standard')
const toolConfigElement = ref(null)
const page = ref('workbench')
// 默认 3D 主视觉，相机画面按需切换。
const viewMode = ref('scene3d')
const sceneData = ref(null)
const sceneStale = ref(true)
const selectedObject = ref('')
const treePanel = ref(null)
let sceneTimer = null
let sceneUpdatedAt = 0
const lease = ref(null)
const creating = ref(false)
const attachId = ref('')
const form = reactive({
  env_id: 'robocasa/PickPlaceCounterToCabinet',
  seed: 0,
  control_freq: 20,
  policy: 'hold',
  // 必须是完整的 tool_safety_spec/1 对象；发 bool 会被服务端规范校验拒绝。
  tool_safety: defaultToolSafety(),
})

// 规范默认值；回填与重置都用它，避免多处字面量漂移。
function defaultToolSafety() {
  return {
    version: 'tool_safety_spec/1',
    enabled: false,
    risk_zones: [],
    tool_annotations: [],
  }
}

// 从会话 config 一次性回填表单。
// 只在 session.id 变化时调用，不参与 refresh 轮询，因此不会覆盖用户未提交的编辑。
// 缺失或非法字段保留规范默认值，不拿历史值猜测。
function hydrateFormFromConfig(config) {
  const source = config && typeof config === 'object' ? config : {}
  if (typeof source.env_id === 'string' && source.env_id) form.env_id = source.env_id

  const seed = Number(source.seed)
  if (source.seed !== undefined && Number.isFinite(seed)) form.seed = seed

  const freq = Number(source.control_freq)
  if (source.control_freq !== undefined && Number.isFinite(freq)) form.control_freq = freq

  if (typeof source.policy === 'string' && source.policy) form.policy = source.policy

  // tool_safety 必须回落到完整规范对象：后端可能存 bool、旧结构或缺失。
  const spec = source.tool_safety
  if (spec && typeof spec === 'object' && spec.version === 'tool_safety_spec/1') {
    form.tool_safety = {
      version: 'tool_safety_spec/1',
      enabled: spec.enabled === true,
      risk_zones: Array.isArray(spec.risk_zones) ? spec.risk_zones.map((zone) => ({ ...zone })) : [],
      tool_annotations: Array.isArray(spec.tool_annotations)
        ? spec.tool_annotations.map((tool) => ({
            ...tool,
            segments: Array.isArray(tool.segments) ? tool.segments.map((segment) => ({ ...segment })) : [],
          }))
        : [],
    }
  } else {
    form.tool_safety = defaultToolSafety()
  }
}

// 预设任务只改 env_id；后端未注册时必须由 createSession 明确报错，
// 前端不静默回退到另一个任务。
const TASK_PRESETS = [
  { label: '拾放', env_id: 'robocasa/PickPlaceCounterToCabinet' },
  { label: '收集切菜工具', env_id: 'robocasa/GatherCuttingTools' },
]

function applyPreset(envId) {
  form.env_id = envId
}

const sceneGeometryKey = ref('')
const sceneGeometry = ref(null)
const legacyScene = computed(() => Boolean(sceneGeometry.value && !sceneGeometry.value.assets))
const completeWorkbenchUrl = computed(() => `http://127.0.0.1:8880/#token=${encodeURIComponent(getToken())}`)
// 会话代数：切换会话后旧响应一律丢弃，避免污染新会话的 scene 与选择。
let sessionGeneration = 0
// 请求序号：用序号而非布尔锁，使得切换会话后旧请求的 finally
// 只能释放自己那一笔，不会把新会话正在飞行的请求解锁。
let sceneRequestSeq = 0
let sceneActiveSeq = 0
// 左侧工具安全配置默认折叠：真实 enabled 配置带保护区/工具/多坐标输入，
// 展开会把底遥测挤出首屏。折叠只隐藏编辑区，右侧 assessment 不受影响。
// 初始与切换会话后均为 false；启用状态在 summary 上展示，用户自行展开。
const toolConfigOpen = ref(false)
// 当前在飞的 scene 请求的中止句柄。
let sceneAbort = null

let sessionTimer = null
let socket = null
let reconnectTimer = null
const wsLive = ref(false)
let eventCursor = 0
let keepaliveTimer = null

const frameSeq = computed(() => session.value?.environment?.frame_seq || 0)

async function createSession() {
  if (creating.value || authChecking.value) return
  if (!authenticated.value || !getToken()) { openLogin(); return }
  creating.value = true
  error.value = ''
  try {
    const created = await api.createSession({
      env_id: form.env_id,
      seed: Number(form.seed),
      control_freq: Number(form.control_freq),
      policy: form.policy,
      tool_safety: form.tool_safety,
    })
    session.value = created
    history.replaceState(null, '', `${window.location.pathname}?session=${encodeURIComponent(created.id)}`)
    await refresh()
  } catch (failure) {
    error.value = failure.message
  } finally {
    creating.value = false
  }
}

// 切换会话时清掉一切从旧会话带过来的状态：选择、几何缓存、轨迹。
function resetSceneState() {
  sceneData.value = null
  sceneStale.value = true
  selectedObject.value = ''
  sceneGeometry.value = null
  sceneGeometryKey.value = ''
  sceneUpdatedAt = 0
  // 中止旧请求，并让它的 finally 不再影响新会话的飞行状态。
  if (sceneAbort) {
    try {
      sceneAbort.abort()
    } catch {
      /* 已结束 */
    }
    sceneAbort = null
  }
  sceneActiveSeq = 0
}

async function refresh() {
  if (!authenticated.value || !session.value) return
  try {
    session.value = await api.getSession(session.value.id)
  } catch (failure) {
    error.value = failure.message
  }
}

async function attach() {
  if (!authenticated.value) { openLogin(); return }
  const id = attachId.value.trim()
  if (!id) return
  try {
    const existing = await api.getSession(id)
    session.value = existing
    error.value = ''
  } catch (failure) {
    error.value = failure.message
  }
}

function appendEvents(incoming) {
  if (!incoming?.length) return
  const fresh = incoming.filter((item) => item.seq > eventCursor)
  if (!fresh.length) return
  events.value = [...events.value, ...fresh].slice(-80)
  eventCursor = fresh[fresh.length - 1].seq
}

async function pollScene() {
  if (!authenticated.value) return
  if (!session.value) {
    resetSceneState()
    return
  }
  // 同一代内不叠请求（300ms 轮询可能快于响应）；
  // 切会话时 resetSceneState 会清零并中止旧请求，新代可立即重开。
  if (sceneActiveSeq) return
  const mySeq = ++sceneRequestSeq
  sceneActiveSeq = mySeq
  const generation = sessionGeneration
  const sessionId = session.value.id
  const controller = new AbortController()
  sceneAbort = controller
  try {
    const token = getToken()
    const query = sceneGeometryKey.value
      ? `?geometry_key=${encodeURIComponent(sceneGeometryKey.value)}`
      : ''
    const response = await fetch(`/api/v1/sessions/${sessionId}/scene${query}`, {
      headers: token ? { 'X-Console-Token': token } : {},
      cache: 'no-store',
      signal: controller.signal,
    })
    // 响应回来时若已切换会话，丢弃这份旧载荷。
    if (generation !== sessionGeneration || session.value?.id !== sessionId) return
    if (!response.ok) {
      sceneStale.value = true
      return
    }
    const payload = await response.json()
    if (generation !== sessionGeneration || session.value?.id !== sessionId) return

    // 静态几何按 geometry_key 缓存。三种情形：
    // 1) 载荷带 geometry：新几何，直接缓存并采用；
    // 2) geometry=null 且 key 与本地缓存 key 一致：后端确认客户端已有同一份，
    //    这时才能复用缓存；
    // 3) key 不匹配（或本地无缓存）：不得把旧 key 的几何当当前场景，回退为空。
    const incomingKey = payload.geometry_key || ''
    if (payload.geometry) {
      sceneGeometry.value = payload.geometry
      sceneGeometryKey.value = incomingKey
      sceneData.value = payload
    } else if (incomingKey && incomingKey === sceneGeometryKey.value && sceneGeometry.value) {
      sceneData.value = { ...payload, geometry: sceneGeometry.value, geometry_cached: true }
    } else {
      // key 不匹配：清掉本地几何，宁可等下一轮也不拿错模型。
      sceneGeometry.value = null
      sceneGeometryKey.value = incomingKey
      sceneData.value = { ...payload, geometry: null, geometry_cached: false }
    }
    sceneUpdatedAt = performance.now()
    sceneStale.value = false
  } catch (failure) {
    // 主动中止不是错误，不标陈旧。
    if (failure?.name !== 'AbortError' && generation === sessionGeneration) sceneStale.value = true
  } finally {
    // 只释放自己那一笔：若已被新请求或 resetSceneState 接管，不碰状态。
    if (sceneActiveSeq === mySeq) sceneActiveSeq = 0
    if (sceneAbort === controller) sceneAbort = null
  }
}

async function pullEvents() {
  if (!authenticated.value || !session.value) return
  try {
    const payload = await api.events(session.value.id, eventCursor)
    appendEvents(payload.events)
  } catch {
    /* 事件流断开不阻塞控制 */
  }
}

function connectStream() {
  disconnectStream()
  const id = session.value?.id
  if (!authenticated.value || !id) return
  const token = getToken()
  if (!token) return
  try {
    socket = new WebSocket(api.streamUrl(id, token))
  } catch {
    wsLive.value = false
    return
  }
  socket.onopen = () => {
    wsLive.value = true
  }
  socket.onmessage = (message) => {
    let payload = null
    try {
      payload = JSON.parse(message.data)
    } catch {
      return
    }
    if (payload.type === 'error') {
      error.value = payload.code === 'session_not_found' ? '会话不存在' : payload.code
      return
    }
    if (payload.type !== 'snapshot') return
    session.value = payload.session
    appendEvents(payload.events)
  }
  socket.onclose = () => {
    wsLive.value = false
    socket = null
    if (reconnectTimer) clearTimeout(reconnectTimer)
    reconnectTimer = setTimeout(connectStream, 2000)
  }
  socket.onerror = () => {
    wsLive.value = false
  }
}

function disconnectStream() {
  if (reconnectTimer) clearTimeout(reconnectTimer)
  reconnectTimer = null
  if (socket) {
    socket.onclose = null
    socket.close()
    socket = null
  }
  wsLive.value = false
}

async function run(command, args = null) {
  if (!authenticated.value) { openLogin(); return }
  if (!session.value || busy.value) return
  busy.value = true
  error.value = ''
  try {
    const ack = await api.command(session.value.id, envelope(session.value, command, args))
    if (ack.acknowledgement === 'rejected' || ack.acknowledgement === 'superseded') {
      const message = ack.message || ack.code
      error.value = message
      treePanel.value?.showNote?.(message)
    }
    await refresh()
    await pullEvents()
  } catch (failure) {
    error.value = failure.message
  } finally {
    busy.value = false
  }
}

async function acknowledgedCommand(id, command, args) {
  const current = await api.getSession(id)
  const ack = await api.command(id, envelope(current, command, args))
  if (!['accepted', 'applied'].includes(ack.acknowledgement)) throw new Error(ack.message || ack.code || '指令未执行')
  for (let attempt = 0; attempt < 100; attempt++) {
    const updated = await api.getSession(id)
    if (session.value?.id === id) session.value = updated
    if (!updated.inflight_command_id && updated.state_version > current.state_version) {
      if (updated.last_command?.rejected) throw new Error(updated.last_command.rejected)
      return updated
    }
    await new Promise(resolve => setTimeout(resolve, 80))
  }
  throw new Error('等待机器人确认超时，请检查会话连接')
}

async function grip(mode) {
  if (!authenticated.value) { openLogin(); return }
  if (!session.value || busy.value || holding.value) return
  busy.value = true
  error.value = ''
  try {
    const id = session.value.id
    await prepareManualControl(id)
    await acknowledgedCommand(id, 'jog', { gripper: mode, repeat: 3 })
  }
  catch (failure) { error.value = failure.message }
  finally { busy.value = false }
}

async function prepareManualControl(id) {
  const current = await api.getSession(id)
  if (current.state === 'RUNNING') await acknowledgedCommand(id, 'pause')
  else if (!['READY', 'PAUSED'].includes(current.state)) throw new Error('环境未就绪，请重新加载厨房仿真')
}

let holdSessionId = null
const holdGesture = createHoldGesture({
  onHolding: direction => { holding.value = direction },
  onError: failure => { error.value = failure.message },
  async step(direction) {
    holdSessionId = session.value.id
    error.value = ''
    activeJogProfile = getJogProfile(jogProfileId.value)
    await prepareManualControl(holdSessionId)
    await acknowledgedCommand(holdSessionId, 'jog', { delta_m: DIRECTIONS[direction].map(axis => axis * activeJogProfile.delta), gripper: 'hold', repeat: activeJogProfile.repeat })
  },
  async begin(direction) {
    const id = holdSessionId
    const acquired = await api.acquireLease(id, 'workbench', TTL_MS)
    lease.value = acquired
    keepaliveTimer = setInterval(async () => {
      try { lease.value = await api.renewLease(id, acquired.lease_id, TTL_MS) }
      catch { endHold() }
    }, Math.round(TTL_MS / 3))
    await acknowledgedCommand(id, 'jog_start', { lease_id: acquired.lease_id, delta_m: DIRECTIONS[direction].map(axis => axis * activeJogProfile.delta), keepalive_ms: TTL_MS })
  },
  async stop() {
    clearInterval(keepaliveTimer)
    keepaliveTimer = null
    const active = lease.value, id = holdSessionId
    lease.value = null
    if (!active || !id) return
    try { await acknowledgedCommand(id, 'jog_stop', { lease_id: active.lease_id }) }
    finally { await api.releaseLease(id, active.lease_id); await refresh() }
  },
})
function startHold(direction) {
  if (!authenticated.value) { openLogin(); return }
  if (!session.value || busy.value || !['READY','PAUSED','RUNNING'].includes(session.value.state)) return
  return holdGesture.start(direction)
}
function endHold() { return holdGesture.end() }
function onBlur() { endHold() }
function onVisibilityChange() { if (document.hidden) onBlur() }
async function configureSafety() {
  toolConfigOpen.value = true
  await nextTick()
  toolConfigElement.value?.scrollIntoView({ block: 'nearest', behavior: reduceTransparency.value ? 'instant' : 'smooth' })
}
const eventNames = { command_applied:'指令已执行', command_accepted:'收到指令', manual_stopped:'手动移动已停止', worker_ready:'环境就绪', session_created:'创建实验', worker_loading:'载入环境', worker_lost:'环境已断开', policy_stopped:'策略已暂停' }
function eventName(item) { return eventNames[item.type] || item.type }


onMounted(() => {
  const hash = new URLSearchParams(window.location.hash.replace(/^#/, ''))
  const hashToken = hash.get('token')
  if (hashToken) {
    // 令牌放 hash，不进入请求路径和服务器日志，设置完立刻清掉
    setToken(hashToken)
    token.value = hashToken
    history.replaceState(null, '', window.location.pathname + window.location.search)
  }
  const preset = new URLSearchParams(window.location.search).get('session')
  if (preset) {
    attachId.value = preset
  }
  if (getToken()) login()
  else openLogin()
  window.addEventListener('blur', onBlur)
  document.addEventListener('visibilitychange', onVisibilityChange)
  sessionTimer = setInterval(async () => {
    if (wsLive.value) return
    await refresh()
    await pullEvents()
  }, 800)
  sceneTimer = setInterval(() => {
    pollScene()
    sceneStale.value = !sceneUpdatedAt || performance.now() - sceneUpdatedAt > 1500
  }, 300)
})

watch(
  () => session.value?.id,
  (id) => {
    events.value = []
    eventCursor = 0
    // 会话变了：代数递增，旧的 scene 响应与轮询都不再被接受。
    sessionGeneration += 1
    resetSceneState()
    if (id) {
      // 只在此处回填一次，从该会话的 config 恢复配置（含真实的 tool_safety）。
      hydrateFormFromConfig(session.value?.config)
      // 一律保持折叠：enabled 只在 summary 上展示，不自动展开。
      toolConfigOpen.value = false
      connectStream()
    } else {
      disconnectStream()
    }
  },
)

onBeforeUnmount(() => {
  stopAuthObserver()
  endHold()
  window.removeEventListener('blur', onBlur)
  document.removeEventListener('visibilitychange', onVisibilityChange)
  disconnectStream()
  if (sessionTimer) clearInterval(sessionTimer)
  if (sceneTimer) clearInterval(sceneTimer)
  if (keepaliveTimer) clearInterval(keepaliveTimer)
})
</script>

<template>
  <div class="shell" :class="{ 'evaluation-view': page === 'evaluation' }">
    <header class="topbar">
      <a class="brand" href="#" aria-label="PRISM 工作台" @click.prevent="page = 'workbench'">
        <svg viewBox="0 0 40 40" aria-hidden="true"><path d="M20 5 35 31H5Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="m20 5 4 26M5 31l19-9 11 9" fill="none" stroke="currentColor" stroke-width="1.2" opacity=".55"/></svg>
        <span>PRISM<span class="brand-caption">具身智能安全测评</span></span>
      </a>
      <nav class="page-nav" aria-label="主要导航">
        <button :class="{ active: page === 'workbench' }" @click="page = 'workbench'">仿真工作台</button>
        <button :class="{ active: page === 'evaluation' }" @click="page = 'evaluation'">安全测评</button>
      </nav>
      <div class="header-actions">
        <button class="login-trigger" :aria-expanded="loginOpen" aria-controls="console-login" @click="openLogin">{{ authenticated ? '更换令牌' : '登录' }}</button>
        <span class="connection" :class="{ connected: session?.live }"><i />{{ session?.live ? '仿真已连接' : '等待连接' }}</span>
        <button ref="settingsButton" class="settings-trigger" :aria-expanded="settingsOpen" aria-label="界面设置与控制台登录" @click="settingsOpen = !settingsOpen">
          <span>设置</span>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3" fill="var(--color-paper)"/><circle cx="15" cy="17" r="3" fill="var(--color-paper)"/></svg>
        </button>
      </div>
      <section v-if="settingsOpen" class="settings-panel panel" aria-label="界面设置与控制台登录" @keydown.esc="closeSettings">
        <div class="section-heading"><h2>界面与登录</h2><button aria-label="关闭设置" @click="closeSettings">×</button></div>
        <button @click="closeSettings(); openLogin()">{{ authenticated ? '更换控制台令牌' : '登录控制台' }}</button>
        <label class="check-row"><input v-model="reduceTransparency" type="checkbox" />降低透明度</label>
        <label class="check-row"><input v-model="frostedGlass" type="checkbox" />使用磨砂玻璃</label>
        <p class="note">清透材质在兼容浏览器中启用。降低透明度后使用稳定表面。</p>
        <span class="settings-status">{{ wsLive ? '数据连接：实时推送' : '数据连接：定时同步' }}</span>
      </section>
    </header>

    <form v-if="loginOpen" id="console-login" class="login-panel panel" aria-label="控制台登录" @submit.prevent="login">
      <div><strong>控制台登录</strong><p id="login-message" role="status">{{ loginMessage || '输入控制台令牌' }}</p></div>
      <label for="console-token">控制台令牌</label>
      <input id="console-token" ref="tokenInput" v-model="token" type="password" autocomplete="off" placeholder="输入控制台令牌" aria-describedby="login-message" :disabled="authChecking" />
      <button class="primary" type="submit" :disabled="authChecking">{{ authChecking ? '验证中…' : '登录' }}</button>
      <button v-if="authenticated" type="button" @click="loginOpen = false">取消</button>
    </form>
    <p v-if="error && !loginOpen" class="banner" role="alert">{{ error }}</p>
    <div v-if="legacyScene" class="legacy-notice" role="status"><div><strong>当前会话由旧版仿真进程提供，只包含简化几何。</strong><span>完整厨房、机械臂模型与新版控制在独立工作台中加载。</span></div><a :href="completeWorkbenchUrl" target="_blank" rel="noopener">打开完整模型工作台 ↗</a></div>
    <EvaluationPanel v-if="page === 'evaluation' && authenticated" />

    <main v-show="page === 'workbench'" class="workspace three-col">
      <aside class="left column" aria-label="环境与场景配置">
        <section class="panel environment-panel">
          <div class="section-heading"><div><span class="eyebrow">ENVIRONMENT</span><h2>实验环境</h2></div><span class="section-number">01</span></div>
          <div class="preset-row" aria-label="任务预设">
            <button v-for="preset in TASK_PRESETS" :key="preset.env_id" :class="{ on: form.env_id === preset.env_id }" @click="applyPreset(preset.env_id)">{{ preset.label }}</button>
          </div>
          <label>仿真任务<input v-model="form.env_id" aria-label="任务" /></label>
          <div class="row"><label>随机种子<input v-model="form.seed" type="number" min="0" /></label><label>控制频率 · Hz<input v-model="form.control_freq" type="number" min="5" max="50" /></label></div>
          <label>机器人策略<select v-model="form.policy"><option value="hold">静止保持 · 手动调试</option><option value="random">随机动作 · 控制链验证</option></select></label>
          <button class="primary load-button" :disabled="creating || authChecking" @click="createSession"><span>{{ creating ? '正在准备厨房…' : '加载仿真环境' }}</span><span aria-hidden="true">↗</span></button>
          <p v-if="creating" class="note">正在载入厨房模型…</p>
          <details class="attach-details"><summary>接入已有会话</summary><div class="attach-row"><input v-model="attachId" placeholder="sess_…" aria-label="会话编号" /><button @click="attach">接入</button></div></details>
        </section>
        <details ref="toolConfigElement" class="tool-config" :open="toolConfigOpen" @toggle="toolConfigOpen = $event.target.open"><summary><span>工具与保护区</span><span class="st">{{ form.tool_safety?.enabled ? '已启用' : '未启用' }}</span></summary><ToolSafetyPanel v-model="form.tool_safety" mode="config" :scene="sceneData" :disabled="creating" /></details>
        <details class="scene-index"><summary><span>场景对象</span><span aria-hidden="true">＋</span></summary><SceneTree ref="treePanel" :scene="sceneData" :session="session" :selected="selectedObject" @select="selectedObject = $event" @edit="run('edit', $event)" /></details>
        <div class="workspace-footnote"><span class="footnote-dot" />ROBOCASA · MUJOCO<span>厨房实验空间</span></div>
      </aside>

      <section class="center column" aria-label="机器人仿真场景">
        <div class="scene-heading"><div><span class="eyebrow">SIMULATION STUDIO</span><h1>厨房仿真工作台<span class="heading-dot">.</span></h1></div><span class="scene-state" :class="{ running: session?.state === 'RUNNING' }"><i />{{ stateText }}</span></div>
        <div class="stage3d" :class="{ 'is-empty': !session }">
          <template v-if="session">
            <Scene3D v-show="viewMode === 'scene3d'" :scene="sceneData" :running="session?.state === 'RUNNING'" :selected="selectedObject" :stale="sceneStale" :live="Boolean(session?.live)" :session-id="session?.id" :tool-safety="sceneData?.tool_safety" :opaque="reduceTransparency" :frosted="frostedGlass" @select="selectedObject = $event" />
            <Viewport v-show="viewMode === 'camera'" :session-id="session?.id" :frame-seq="frameSeq" :state="session?.state" :live="Boolean(session?.live)" :stale="sceneStale" />
          </template>
          <div v-else class="scene-welcome">
            <div class="welcome-grid" aria-hidden="true" />
            <svg class="welcome-illustration" viewBox="0 0 320 190" fill="none" aria-hidden="true"><path d="m25 147 130-66 140 64-128 72Z" fill="#c9d8cd" opacity=".5"/><path d="m57 128 105-52 100 45v28l-101 50-104-44Z" fill="#d8e2d6" stroke="#9caf9d"/><path d="M57 128v-54l104-51 101 46v52M161 23v125M80 117V78m25 27V65m25 28V54m52 40V60m24 45V71m25 46V84" stroke="#9caf9d"/><path d="m157 141-7-25 22-31-20-29" stroke="#6a8c80" stroke-width="17" stroke-linecap="round" stroke-linejoin="round"/><circle cx="150" cy="116" r="10" fill="#e6eee5" stroke="#6a8c80" stroke-width="4"/><circle cx="172" cy="85" r="9" fill="#e6eee5" stroke="#6a8c80" stroke-width="4"/><path d="m152 56-9-6-7 7m16-1 2-10 10-1" stroke="#6a8c80" stroke-width="4" stroke-linecap="round"/></svg>
            <span class="welcome-label">场景待加载 · 示意图</span>
            <h2>让每一次动作，都有据可查。</h2><p>加载厨房环境，观察机械臂的行为，<br />配置保护区并记录真实安全测评结果。</p>
            <button class="primary" :disabled="creating || authChecking" @click="createSession">{{ creating ? '加载中…' : '进入厨房实验' }} <span aria-hidden="true">↗</span></button>
          </div>
          <div class="stage-toolbar"><LiquidTabs v-model="viewMode" :items="viewItems" :opaque="reduceTransparency" :frosted="frostedGlass" /><LiquidGlass class="stage-task" light :opaque="reduceTransparency" :frosted="frostedGlass"><span>{{ session ? taskName : 'RoboCasa Kitchen' }}</span></LiquidGlass></div>
          <CommandBar v-if="session" class="stage-playback" :session="session" :busy="busy || Boolean(holding)" :opaque="reduceTransparency" :frosted="frostedGlass" @command="run" />
        </div>
        <JogPad v-if="session" v-model:profile="jogProfileId" :session="session" :busy="busy" :holding="holding" @start="startHold" @end="endHold" @grip="grip" />
      </section>

      <aside class="right column" aria-label="会话状态与安全风险">
        <div class="section-intro"><span class="eyebrow">OBSERVE & EVALUATE</span><h2>安全观测</h2></div>
        <ToolSafetyPanel mode="assessment" :model-value="form.tool_safety" :scene="sceneData" :safety="sceneData?.tool_safety" @configure="configureSafety" />
        <StatusPanel :session="session" :scene="sceneData" :stale="sceneStale" :lease="lease" :last-error="loginOpen ? '' : error" />
      </aside>
    </main>

    <details v-show="page === 'workbench'" class="records-drawer"><summary><span>实验记录</span><span class="records-meta">遥测与操作日志 <span aria-hidden="true">⌄</span></span></summary>
    <footer class="telemetry-footer">
      <TelemetryPanel :scene="sceneData" :session="session" :stale="sceneStale" :events="events" />
      <section class="panel log"><div class="section-heading"><h2>实验事件</h2><span class="note">{{ events.length ? '最近 ' + Math.min(events.length, 6) + ' 条' : '等待实验开始' }}</span></div><ol><li v-for="item in events.slice(-6).reverse()" :key="item.seq"><span class="event-index">{{ String(item.seq).padStart(2, '0') }}</span><div><span class="event-name">{{ eventName(item) }}</span><span v-if="item.message || (item.reason && item.reason !== 'operator_stop')" class="event-message">{{ item.message || item.reason }}</span></div></li><li v-if="!events.length" class="empty"><span class="empty-event-mark">○</span><p>暂无操作记录</p></li></ol></section>
    </footer>
    </details>
  </div>
</template>

<style scoped>
.login-panel{display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin:12px 24px;padding:16px 20px;flex-shrink:0}.login-panel p{margin:4px 0 0;font-size:13px}.login-panel input{flex:1;min-width:180px;max-width:360px}.login-panel input:focus-visible{outline:2px solid var(--color-focus,#557b79);outline-offset:3px}.login-panel label{font-size:13px}.login-trigger{white-space:nowrap}@media(max-width:640px){.login-panel{margin:12px}.login-panel>div{width:100%}}

.shell{height:auto;min-height:100dvh;padding:0 24px 20px;max-width:1920px;margin:auto;display:flex;flex-direction:column;gap:18px}
.topbar{position:relative;display:grid;grid-template-columns:1fr auto 1fr;align-items:center;min-height:88px;flex:0 0 auto;border-bottom:1px solid var(--line);gap:24px}
.brand{min-height:44px;display:flex;align-items:center;gap:10px;justify-self:start;color:var(--ink);text-decoration:none;font-size:22px;font-weight:600;letter-spacing:.08em;line-height:1.2}
.brand svg{width:38px;height:38px;color:#436a5b}.brand-caption{display:block;font-size:10px;font-weight:400;letter-spacing:.08em;color:var(--muted);margin-top:4px}
.page-nav{display:flex;gap:4px;background:#e8ece4;border-radius:999px;padding:4px}.page-nav button{border:0;background:transparent;font-size:13px;padding:10px 22px}.page-nav button.active{background:#fff;box-shadow:0 2px 5px #203b3010}
.header-actions{justify-self:end;display:flex;align-items:center;gap:18px}.connection{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:12px}.connection i,.scene-state i{width:6px;height:6px;border-radius:50%;background:#a6b1a3}.connection.connected i{background:#5b9371}.settings-trigger{grid-auto-flow:column;gap:6px;background:transparent;border-color:var(--line);padding:10px;display:grid;place-items:center}.settings-trigger svg{width:22px;height:22px}
.settings-panel{position:absolute;z-index:30;right:0;top:78px;width:330px;padding:24px;background:var(--panel);border:1px solid var(--line);border-radius:24px;box-shadow:0 16px 40px #13252922}.settings-panel .primary{width:100%;margin-bottom:16px}.settings-panel .section-heading button{padding:0;font-size:24px}.settings-panel .check-row{display:flex;flex-direction:row;align-items:center;gap:10px;min-height:44px;margin:0}.settings-status{font-size:12px;color:var(--muted)}
.workspace{flex:1;display:grid;grid-template-columns:minmax(250px,23fr) minmax(0,54fr) minmax(250px,23fr);gap:20px;min-height:0;align-items:start}.column{min-height:0;min-width:0;display:flex;flex-direction:column;gap:16px}.left,.right{overflow:visible;}.left>*:not(.workspace-footnote),.right>*{flex-shrink:0}.center{overflow:visible}
.section-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:16px}.section-heading h2{margin:0}.eyebrow{display:block;font-size:10px;letter-spacing:.14em;color:var(--muted);margin-bottom:6px}.section-number{font-size:24px;font-weight:400;color:#becabd;letter-spacing:-.04em}.section-intro{padding:4px 4px 0}.section-intro h2{font-size:20px;margin:0;font-weight:500}
.preset-row{display:grid;grid-template-columns:.8fr 1.2fr;gap:6px;margin-bottom:20px}.preset-row button{padding:8px 10px;font-size:12px;white-space:nowrap;background:#f4f6ef}.preset-row button.on{background:#e3ece1;border-color:#b4c6b2;color:#315d48}
label{display:flex;flex-direction:column;gap:7px;font-size:12px;color:var(--muted);margin-bottom:16px}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}.load-button{width:100%;display:flex;justify-content:space-between;align-items:center;margin-top:4px}.note{font-size:12px;line-height:1.55;color:var(--muted)}.environment-panel>.note{margin:12px 0 0}.attach-details{margin-top:20px;border-top:1px solid var(--line);padding-top:8px}.attach-details summary{min-height:44px;display:flex;align-items:center;justify-content:space-between;cursor:pointer;list-style:none;font-size:12px;color:var(--muted)}.attach-details summary::after{content:'↗'}.attach-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px}.workspace-footnote{margin-top:auto;padding:12px 8px 0;display:flex;gap:6px;align-items:center;font-size:9px;letter-spacing:.08em;color:#81907e;flex-wrap:wrap}.workspace-footnote>span:last-child{width:100%;padding-left:12px;font-size:11px;letter-spacing:0}.footnote-dot{width:5px;height:5px;border-radius:50%;background:#8eaa87}
.scene-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;min-height:56px;padding:2px 4px}.scene-heading h1{font-size:25px;font-weight:500;letter-spacing:-.04em;margin:0;line-height:1.25}.heading-dot{color:#6c9774;margin-left:2px}.scene-state{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted);white-space:nowrap}.scene-state.running i{background:#5b9371}
.stage3d{position:relative;flex:none;height:clamp(400px,50dvh,660px);min-height:0;border-radius:28px;overflow:hidden;background:#dce6da;border:1px solid #d4ddd1;isolation:isolate}.stage3d>:is(.scene3d,.viewport){width:100%;height:100%;position:absolute;inset:0}.stage-toolbar{position:absolute;top:20px;left:20px;right:20px;display:flex;justify-content:space-between;align-items:flex-start;gap:12px;pointer-events:none;z-index:4}.stage-toolbar>*{pointer-events:auto}.stage-task{max-width:42%;padding:16px 18px;font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-height:54px}.stage-playback{position:absolute;bottom:20px;left:20px;right:20px;z-index:4}
.scene-welcome{isolation:isolate;position:absolute;inset:0;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;padding:78px 28px 30px;background:radial-gradient(ellipse at 75% 20%,#eff3df,transparent 60%),linear-gradient(140deg,#d4e0d4,#b8cec3);color:#31554a}.welcome-grid{position:absolute;inset:0;background-image:linear-gradient(#466d5510 1px,transparent 1px),linear-gradient(90deg,#466d5510 1px,transparent 1px);background-size:40px 40px;mask-image:linear-gradient(transparent,#000 25%,#000 75%,transparent);z-index:-1}.welcome-illustration{width:min(72%,300px);height:150px;flex-shrink:1;min-height:70px}.welcome-label{font-size:10px;letter-spacing:.07em;color:#617f6e;margin:10px 0}.scene-welcome h2{font-size:22px;font-weight:500;margin:0 0 12px;letter-spacing:-.035em}.scene-welcome p{font-size:13px;line-height:1.8;color:#5b7768;margin:0 0 20px}.scene-welcome button{font-size:13px;display:flex;gap:24px}
.telemetry-footer{flex:none;min-height:210px;display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1fr);gap:20px;overflow:hidden}.telemetry-footer>*{height:auto;min-height:0;overflow:visible}.log{padding:16px 20px}.log .section-heading{margin-bottom:10px}.log h2{font-size:14px}.log ol{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:6px}.log li{display:flex;align-items:flex-start;gap:12px;font-size:12px;padding:4px 0;border-bottom:1px solid #e8ece3}.event-index{color:#809080;font-size:10px;min-width:20px;margin-top:2px}.event-name{display:block;color:#4b6356;font-size:11px}.event-message{display:block;color:var(--muted);font-size:11px}.log li.empty{border:0;align-items:center;padding-top:18px;gap:14px}.empty-event-mark{font-size:40px;color:#c5d0be}.empty p{color:#62755f;font-size:12px;margin:0;line-height:1.8}.empty p span{color:#8b9886;font-size:11px}
.evaluation-view{height:auto;min-height:100dvh}.evaluation-view>.evaluation{flex:1}
@media(min-width:1760px){.shell{padding-left:32px;padding-right:32px}.workspace{gap:24px}.scene-heading h1{font-size:28px}}
@media(max-width:1370px){.shell{padding:0 16px 16px;gap:16px}.workspace{grid-template-columns:minmax(230px,23fr) minmax(0,54fr) minmax(230px,23fr);gap:16px}.stage-task{display:none}.scene-heading h1{font-size:22px}.telemetry-footer{flex-basis:180px;gap:16px}.page-nav button{padding:10px 16px}}
@media(max-width:1180px){.shell{height:auto;min-height:100dvh}.workspace{grid-template-columns:minmax(230px,1fr) minmax(0,2fr);min-height:680px;align-items:start}.center{grid-column:2;grid-row:1;min-height:680px;position:sticky;top:16px}.left{grid-column:1;grid-row:1;overflow:visible}.right{grid-column:1/-1;display:grid;grid-template-columns:1fr 1fr;overflow:visible}.section-intro{grid-column:1/-1}.stage3d{min-height:540px}.telemetry-footer{flex:auto;min-height:210px}.column{min-height:0}.connection{display:none}}
@media(max-width:760px){.shell{padding:0 12px 16px;gap:16px}.topbar{grid-template-columns:1fr auto;gap:8px;min-height:76px;padding:12px 0}.brand{font-size:19px;gap:6px}.brand svg{width:32px;height:32px}.brand-caption{font-size:9px}.page-nav{grid-row:2;grid-column:1/-1;justify-self:stretch;padding:3px}.page-nav button{flex:1;min-height:44px;padding:8px 14px}.header-actions{gap:8px;grid-column:2;grid-row:1}.settings-trigger{min-height:44px}.settings-panel{top:118px;right:0;width:min(330px,calc(100vw - 24px))}.workspace{display:flex;flex-direction:column;min-height:0;gap:20px}.center{order:-1;width:100%;position:static;min-height:0;gap:12px}.scene-heading{min-height:46px;padding:0 4px}.scene-heading h1{font-size:22px}.eyebrow{font-size:9px;margin-bottom:4px}.stage3d{height:480px;flex:none;min-height:0;border-radius:24px}.stage-toolbar{top:12px;left:12px;right:12px}.stage-playback{left:12px;right:12px;bottom:12px}.scene-welcome{padding-left:16px;padding-right:16px}.scene-welcome h2{font-size:20px}.left,.right{display:flex;width:100%;overflow:visible}.right .section-intro{padding-top:8px}.workspace-footnote{display:none}.telemetry-footer{display:flex;flex-direction:column;min-height:0;flex:auto;overflow:visible;gap:16px}.telemetry-footer>*{height:auto;min-height:190px;overflow:visible}.scene-welcome .welcome-illustration{height:160px}.scene-state{font-size:11px}}
.scene-index{border:1px solid var(--line);border-radius:24px;background:var(--panel);padding:5px 20px}.scene-index>summary{list-style:none;display:flex;align-items:center;justify-content:space-between;min-height:48px;font-size:12px;cursor:pointer}.scene-index[open]>summary>span:last-child{transform:rotate(45deg)}.scene-index :deep(.panel){padding:16px 0;border:0;box-shadow:none}.records-drawer{border-top:1px solid var(--line);margin-top:4px}.records-drawer>summary{display:flex;align-items:center;justify-content:space-between;min-height:48px;cursor:pointer;list-style:none;font-size:12px;color:#4c6558}.records-meta{color:var(--muted);font-size:11px;display:flex;align-items:center;gap:16px}.records-drawer[open]>.telemetry-footer{margin:4px 0 12px}.records-drawer[open] .records-meta>span{transform:rotate(180deg)}.section-intro{min-height:56px;display:flex;flex-direction:column;justify-content:center}.section-intro h2{font-size:17px}.scene-index summary:focus-visible,.records-drawer>summary:focus-visible{outline:2px solid var(--color-focus,#557b79);outline-offset:4px;border-radius:8px}
@media(max-width:1180px) and (min-width:761px){.center{position:static;min-height:0}.workspace{min-height:0}.right{grid-template-columns:1fr 1fr}.stage3d{height:540px}}
@media(max-width:760px){.section-intro{min-height:0}.scene-index{width:100%}.telemetry-footer{display:flex;flex-direction:column}.records-drawer{width:100%}}

</style>

<style scoped>
.legacy-notice{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:16px 24px;margin:0 24px 12px;border:1px solid #c7d4c9;border-radius:18px;background:#e4efeb;color:#202929;font-size:14px}.legacy-notice span{display:block;margin-top:4px;font-size:12px;color:#52635e}.legacy-notice a{min-height:44px;display:flex;align-items:center;white-space:nowrap;color:#254e46;font-weight:600}@media(max-width:640px){.legacy-notice{margin:0 12px 12px;align-items:flex-start;flex-direction:column}}
</style>

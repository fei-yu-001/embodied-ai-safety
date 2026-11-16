<script setup>
/**
 * 仿真相机视口。
 *
 * 只做一件事：轮询 frame endpoint，显示后端真实渲染的 JPEG。
 *
 * 关键规则：
 * - 画面是否实时由 `X-Prism-Frame-Live` 响应头决定，不由画面是否存在决定。
 *   后端在浏览器/历史会话里可能仍留着上一轮的图；那种图必须标成历史画面，
 *   不能因为“有图”就说实时。
 * - 切会话或卸载时用 AbortController + generation 作废旧请求，旧帧不得覆盖新会话。
 * - object URL 在切会话/卸载/被新帧替换时 revoke；finally 只释放自己这次请求的资源。
 * - `img` 的原始像素不做任何滤镜；HUD 允许渐变。
 */
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { api, getToken } from '../api.js'

const props = defineProps({
  sessionId: String,
  frameSeq: Number,
  state: String,
  live: Boolean,
  stale: Boolean,
})

const src = ref('')
const failed = ref(false)
// null = 还没拿到头部判定；true/false = 后端对该帧的实时性声明
const frameLive = ref(null)

let timer = null
let busy = false
let controller = null
let objectUrl = ''
let generation = 0

const hasFrame = computed(() => Boolean(src.value))

// 需要正面证据才标“实时”：头部说 live，或父级明确 live=true。
// 缺少证据、stale、或头部为 0，都按非实时处理。
const isLive = computed(() => {
  if (props.stale === true) return false
  if (props.live === false) return false
  if (frameLive.value === false) return false
  return frameLive.value === true || props.live === true
})

const livenessText = computed(() => {
  if (!props.sessionId) return '未连接'
  if (isLive.value) return '实时'
  if (hasFrame.value) return '历史画面 · 后端未连接'
  if (failed.value) return '后端未连接'
  return '连接中'
})

async function tick() {
  if (!props.sessionId || busy) return
  // 这次请求属于哪个 generation：await 之后若已变化，结果必须丢弃
  const gen = generation
  const myController = new AbortController()
  controller = myController
  busy = true
  let createdUrl = ''
  try {
    const token = getToken()
    const response = await fetch(api.frameUrl(props.sessionId), {
      headers: token ? { 'X-Console-Token': token } : {},
      cache: 'no-store',
      signal: myController.signal,
    })
    if (gen !== generation) return
    if (!response.ok) {
      failed.value = true
      frameLive.value = false
      return
    }
    // 头部是实时性真值来源；缺失时不臆断实时
    const header = response.headers.get('X-Prism-Frame-Live')
    const live = header === '1' ? true : header === '0' ? false : null
    const blob = await response.blob()
    if (gen !== generation) return
    createdUrl = URL.createObjectURL(blob)
    if (gen !== generation) return
    const previous = objectUrl
    objectUrl = createdUrl
    createdUrl = '' // 所有权已转移，finally 不再回收
    src.value = objectUrl
    if (previous) URL.revokeObjectURL(previous)
    frameLive.value = live
    failed.value = false
  } catch (error) {
    // 主动取消不是失败：切会话/卸载时不该闪一下错误态
    if (error?.name === 'AbortError' || gen !== generation) return
    failed.value = true
    frameLive.value = false
  } finally {
    // 只释放自己这次请求产生的资源
    if (createdUrl) URL.revokeObjectURL(createdUrl)
    if (myController === controller) {
      controller = null
      busy = false
    }
  }
}

function releaseFrame() {
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl)
    objectUrl = ''
  }
  src.value = ''
  frameLive.value = null
}

function start() {
  stop()
  generation += 1
  failed.value = false
  tick()
  timer = setInterval(tick, 250)
}

function stop() {
  if (timer) clearInterval(timer)
  timer = null
  // 作废在途请求，避免旧会话的响应落到新会话
  generation += 1
  if (controller) {
    controller.abort()
    controller = null
  }
  busy = false
  releaseFrame()
}

watch(() => props.sessionId, start, { immediate: true })
onBeforeUnmount(stop)
</script>

<template>
  <div class="viewport">
    <!-- 原始像素：不加任何 CSS 滤镜，避免污染相机证据 -->
    <img
      v-if="src"
      :src="src"
      :alt="isLive ? '仿真实时相机画面' : '仿真历史相机画面（后端未连接）'"
    />
    <div v-else class="placeholder">
      <p v-if="!sessionId">尚未加载环境</p>
      <p v-else-if="failed">后端未连接，暂无画面</p>
      <p v-else>正在连接相机…</p>
    </div>

    <!-- 失联缓存帧的显式提示：有图但不实时时必须说明 -->
    <div v-if="hasFrame && !isLive" class="history" role="status">
      历史画面 · 后端未连接（非实时）
    </div>

    <div class="hud">
      <span v-if="state" class="badge">{{ state }}</span>
      <span class="live" :class="isLive ? 'on' : 'off'">
        <span class="dot" aria-hidden="true"></span>{{ livenessText }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.viewport {
  position: relative;
  aspect-ratio: 16 / 10;
  border-radius: var(--radius);
  overflow: hidden;
  background: #070713;
  border: 1px solid var(--line);
  box-shadow: 0 0 40px rgba(168, 85, 247, 0.22);
}

/* 相机像素必须保真：这里只做布局，禁止 filter / mix-blend 之类处理 */
img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  filter: none;
}

.placeholder {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: var(--muted);
  text-align: center;
  padding: 24px;
}

.history {
  position: absolute;
  top: 12px;
  left: 12px;
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 12px;
  color: #ffe9bd;
  background: rgba(10, 10, 31, 0.82);
  border: 1px solid rgba(255, 204, 77, 0.55);
}

.hud {
  position: absolute;
  left: 12px;
  bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.badge,
.live {
  padding: 4px 12px;
  border-radius: 999px;
  background: rgba(10, 10, 31, 0.75);
  border: 1px solid var(--line);
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.live {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  letter-spacing: 0.06em;
}

.live .dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
}

/* 状态不只靠颜色：文字本身已区分实时/历史/未连接 */
.live.on {
  color: #d9fff3;
  border-color: rgba(53, 224, 161, 0.5);
  background-image: linear-gradient(135deg, rgba(53, 224, 161, 0.28), rgba(0, 212, 255, 0.22));
}

.live.off {
  color: #ffe9bd;
  border-color: rgba(255, 204, 77, 0.5);
  background-image: linear-gradient(135deg, rgba(255, 204, 77, 0.24), rgba(255, 92, 122, 0.2));
}
</style>

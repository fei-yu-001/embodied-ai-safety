<script setup>
/**
 * 底部真实遥测：只画后端真实返回的 scene 采样。
 *
 * 约束：
 * - 每次收到真实 scene 载荷时把 eef 位置、关节、时间、step 追加进采样缓冲；
 * - 不生成随机数、不插值、不补点。没有真实采样时显示"无数据"。
 * - 切换会话（sessionId 变化）时清空历史采样，绝不把上一个会话的曲线带过来。
 * - 数据无效或缺失时显示"未知/历史"，不冒充实时。
 */
import { computed, ref, watch } from 'vue'

const props = defineProps({
  scene: { type: Object, default: null },
  session: { type: Object, default: null },
  stale: { type: Boolean, default: false },
  events: { type: Array, default: () => [] },
})

const MAX_SAMPLES = 240
const samples = ref([])

// 分段标识：session + episode。任一项变化就清空，避免把两回合接成一条曲线。
let segmentKey = null

function segmentOf(sessionId, episode) {
  return `${sessionId ?? ''}#${episode ?? ''}`
}

function num(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function axisOf(vector, index) {
  return Array.isArray(vector) ? num(vector[index]) : null
}

// 会话或 episode 变化都立即清空：分段不能只靠 session。
watch(
  () => segmentOf(props.session?.id, props.scene?.episode ?? props.session?.episode),
  (key) => {
    if (key !== segmentKey) {
      segmentKey = key
      samples.value = []
    }
  },
  { immediate: true },
)

watch(
  () => props.scene,
  (data) => {
    if (!data) return
    // 陈旧或出错的载荷不进入采样：宁可断开，也不把不可靠数据画成实时曲线。
    if (props.stale) return
    if (data.error) return

    const episode = data.episode ?? props.session?.episode
    const key = segmentOf(props.session?.id, episode)
    if (key !== segmentKey) {
      segmentKey = key
      samples.value = []
    }

    const eef = data.eef?.position
    const joints = Array.isArray(data.joints) ? data.joints : null
    const step = num(data.step)
    const simSeconds = num(data.sim_seconds)
    const x = axisOf(eef, 0)
    const y = axisOf(eef, 1)
    const z = axisOf(eef, 2)

    const hasEef = x !== null || y !== null || z !== null
    const validJoints = joints ? joints.filter((v) => num(v) !== null) : []
    // 既无有效 eef、也无有效 joints 的载荷不追加，避免只靠 step 就造出采样点。
    if (!hasEef && !validJoints.length) return

    const last = samples.value[samples.value.length - 1]
    // step 回退说明已进入新一轮/重置：不能接着上一条继续画。
    if (last && step !== null && last.step !== null && step < last.step) {
      samples.value = []
    } else if (
      last &&
      step !== null &&
      last.step === step &&
      last.x === x &&
      last.y === y &&
      last.z === z
    ) {
      // 同一 step 且位姿未变的重复轮询不追加。
      return
    }

    const next = samples.value.slice(-(MAX_SAMPLES - 1))
    next.push({
      step,
      simSeconds,
      x,
      y,
      z,
      joints: validJoints,
      jointCount: joints ? joints.length : 0,
      episode,
    })
    samples.value = next
  },
  { deep: false },
)

const latest = computed(() => samples.value[samples.value.length - 1] || null)
const hasSamples = computed(() => samples.value.length > 0)

function fmt(value, digits = 4) {
  if (value === null || value === undefined) return '—'
  if (!Number.isFinite(value)) return '—'
  return value.toFixed(digits)
}

// eef x 轨迹折线：没有真实采样时不生成任何点。
const chart = computed(() => {
  const points = samples.value.map((s) => s.x).filter((v) => v !== null)
  if (points.length < 2) return null
  const width = 100
  const height = 30
  const min = Math.min(...points)
  const max = Math.max(...points)
  const span = max - min || 1
  const path = points
    .map((value, index) => {
      const px = (index / (points.length - 1)) * width
      const py = height - ((value - min) / span) * height
      return `${px.toFixed(2)},${py.toFixed(2)}`
    })
    .join(' ')
  return { path, min, max, count: points.length }
})

const jointSummary = computed(() => {
  const joints = latest.value?.joints
  if (!joints || !joints.length) return null
  return {
    count: joints.length,
    first: joints.slice(0, 6).map((v) => fmt(v, 3)),
  }
})

const latestEvent = computed(() => {
  const list = props.events || []
  return list.length ? list[list.length - 1] : null
})

const liveState = computed(() => {
  // 不能“有 session 就算实时”：必须 session.live 且 scene 新鲜。
  if (!props.session) return '未连接'
  if (props.stale) return hasSamples.value ? '历史' : '无数据'
  if (!props.session.live) return hasSamples.value ? '历史' : '无数据'
  return hasSamples.value ? '实时' : '无数据'
})
</script>

<template>
  <section class="telemetry holo-timeline" aria-label="真实遥测">
    <header class="telemetry-head">
      <h2 class="card-title">遥测</h2>
      <span class="pill" :class="{ ghost: !hasSamples || stale }">{{ liveState }}</span>
      <span v-if="stale && hasSamples" class="mini warn">显示历史采样，非实时</span>
    </header>

    <div class="telemetry-grid">
      <div class="kpi is-empty-hint">
        <span class="kpi-label">EEF 位置 m</span>
        <span class="kpi-value mono">
          {{ latest ? `${fmt(latest.x, 3)}, ${fmt(latest.y, 3)}, ${fmt(latest.z, 3)}` : '—' }}
        </span>
      </div>
      <div class="kpi">
        <span class="kpi-label">步数 / 时间</span>
        <span class="kpi-value mono">
          {{ latest?.step ?? '—' }} · {{ latest?.simSeconds != null ? `${fmt(latest.simSeconds, 2)} s` : '—' }}
        </span>
      </div>
      <div class="kpi">
        <span class="kpi-label">关节</span>
        <span class="kpi-value mono">
          {{ jointSummary ? jointSummary.count + ' 项' : '—' }}
        </span>
        <span v-if="jointSummary" class="mini">{{ jointSummary.first.join(' / ') }}</span>
      </div>
      <div class="kpi">
        <span class="kpi-label">采样数</span>
        <span class="kpi-value mono">{{ samples.length }}</span>
        <span class="mini">上限 {{ MAX_SAMPLES }}</span>
      </div>
    </div>

    <div class="telemetry-chart">
      <!-- 只有真实采样达到 2 点才画线；否则明确显示无数据。 -->
      <svg v-if="chart" viewBox="0 0 100 30" preserveAspectRatio="none" role="img"
           aria-label="末端 X 位置轨迹，来自真实采样">
        <polyline class="timeline-line" :points="chart.path" />
      </svg>
      <div v-else class="timeline-empty">
        {{ hasSamples ? '采样不足，暂不绘图' : '无遥测数据（未收到真实 scene 采样）' }}
      </div>
      <p v-if="chart" class="mini chart-note">
        末端 X：{{ fmt(chart.min, 3) }} – {{ fmt(chart.max, 3) }} m · {{ chart.count }} 点
      </p>
    </div>

    <p class="mini events">
      <span>最新事件：</span>
      <template v-if="latestEvent">
        <span class="seq">{{ latestEvent.seq }}</span>
        <span class="type">{{ latestEvent.type }}</span>
        <span v-if="latestEvent.reason" class="why">{{ latestEvent.reason }}</span>
        <span v-else-if="latestEvent.message" class="why">{{ latestEvent.message }}</span>
      </template>
      <span v-else>暂无</span>
    </p>
  </section>
</template>

<style scoped>
.telemetry {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.telemetry-head {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.telemetry-head h2 {
  margin: 0;
}

.telemetry-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}

.kpi {
  min-width: 0;
}

.kpi.is-empty-hint .kpi-value {
  font-size: 15px;
}

.telemetry-chart {
  position: relative;
  min-height: 76px;
}

.telemetry-chart svg {
  width: 100%;
  height: 64px;
  display: block;
}

.chart-note {
  margin: 4px 0 0;
}

.events {
  margin: 0;
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-items: baseline;
}

@media (max-width: 1180px) {
  .telemetry-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .telemetry-grid {
    grid-template-columns: 1fr;
  }
}
</style>

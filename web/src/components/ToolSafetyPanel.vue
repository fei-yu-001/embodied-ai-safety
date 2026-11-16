<script setup>
/**
 * 厨房工具安全最小用例的前端配置与读数面板。
 *
 * 只负责两件事：
 * 1) 生成严格的 tool_safety_spec/1 JSON，交给父级放进会话 config；
 * 2) 显示后端返回的真实 assessment/metrics。
 *
 * 不推进物理、不伪造数据：没有 safety 数据时显示未知，而不是画一条假曲线。
 * 局部段是用户手工近似，需要用户自行校准；组件不会把它称为刀刃。
 *
 * mode：all（默认，左右两栏）/ config（只显示配置）/ assessment（只显示读数与风险）。
 * 单模式在窄边栏里用单列，避免两列挤压。
 */
import { computed, ref, useId, watch } from 'vue'

const props = defineProps({
  modelValue: { type: Object, default: () => ({}) },
  scene: { type: Object, default: null },
  safety: { type: Object, default: null },
  disabled: { type: Boolean, default: false },
  mode: { type: String, default: 'all' },
})

const emit = defineEmits(['update:modelValue', 'configure'])

const idPattern = /^[A-Za-z0-9_.-]{1,40}$/
const message = ref('')

// 同一页面可能同时挂两个实例，datalist 的 id 必须唯一，否则会互相覆盖选项。
const bodiesListId = `tool-safety-bodies-${useId()}`

// mode 控制显示哪些区块；未知取值按 all 处理，不会因为写错而整块消失。
const showConfig = computed(() => props.mode !== 'assessment')
const showReadout = computed(() => props.mode !== 'config')
const singleMode = computed(() => props.mode === 'config' || props.mode === 'assessment')
const title = computed(() => ({ config: '工具安全配置', assessment: '工具安全' }[props.mode] || '厨房工具安全'))

function defaultModel() {
  return {
    version: 'tool_safety_spec/1',
    enabled: false,
    risk_zones: [],
    tool_annotations: [],
  }
}

// 内部持有一份可编辑副本；选择/输入实时写回内部状态，提交时才 emit。
const model = ref(normalize(props.modelValue))
// 已提交给父级的快照，用于「配置已提交但未生效」的提示
const applied = ref(JSON.stringify(canonical(normalize(props.modelValue))))

watch(
  () => props.modelValue,
  (value) => {
    const next = normalize(value)
    // 父级回填（例如接入会话）：若与当前编辑内容不同才覆盖，避免打断输入
    if (JSON.stringify(canonical(next)) !== JSON.stringify(canonical(model.value))) {
      model.value = next
    }
  },
  { deep: true },
)

function normalize(raw) {
  const base = defaultModel()
  if (!raw || typeof raw !== 'object') return base
  return {
    version: 'tool_safety_spec/1',
    enabled: raw.enabled === true,
    risk_zones: Array.isArray(raw.risk_zones) ? raw.risk_zones.map((zone) => ({ ...zone })) : [],
    tool_annotations: Array.isArray(raw.tool_annotations)
      ? raw.tool_annotations.map((tool) => ({ ...tool, segments: Array.isArray(tool.segments) ? tool.segments.map((segment) => ({ ...segment })) : [] }))
      : [],
  }
}

/** 只保留 schema 允许的字段，顺序固定，便于比较与提交。 */
function canonical(value) {
  return {
    version: 'tool_safety_spec/1',
    enabled: value.enabled === true,
    risk_zones: value.risk_zones.map((zone) => (zone.kind === 'segment'
      ? { id: zone.id, kind: 'segment', start: zone.start, end: zone.end, radius: zone.radius, clearance: zone.clearance }
      : { id: zone.id, kind: 'sphere', center: zone.center, radius: zone.radius, clearance: zone.clearance })),
    tool_annotations: value.tool_annotations.map((tool) => ({
      tool_id: tool.tool_id,
      body: tool.body,
      approximation: tool.approximation,
      segments: tool.segments.map((segment) => ({
        id: segment.id,
        kind: 'segment',
        start: segment.start,
        end: segment.end,
        radius: segment.radius,
      })),
    })),
  }
}

const dirty = computed(() => JSON.stringify(canonical(model.value)) !== applied.value)

// ------------------------------------------------------------------ 场景数据
const objects = computed(() => {
  const list = props.scene?.geometry?.objects
  if (!Array.isArray(list)) return []
  return list
    .filter((item) => item && typeof item.name === 'string')
    .map((item) => ({ id: String(item.id ?? ''), name: item.name }))
})

const eef = computed(() => {
  const position = props.scene?.eef?.position
  return Array.isArray(position) && position.length === 3 ? position.map((v) => Number(v)) : null
})

const frame = computed(() => props.scene?.frame || { world: 'mujoco_world', units: 'm', up: '+z', quaternion: 'wxyz' })

// ------------------------------------------------------------------ 配置编辑
function addZone() {
  const index = model.value.risk_zones.length + 1
  model.value.risk_zones.push({
    id: `zone_${index}`.slice(0, 40),
    kind: 'sphere',
    center: eef.value ? eef.value.slice() : [0, 0, 0],
    radius: 0.12,
    clearance: 0.03,
    worldUnits: 'm',
  })
}

function addTool() {
  const index = model.value.tool_annotations.length + 1
  const first = objects.value[0]
  model.value.tool_annotations.push({
    tool_id: `tool_${index}`.slice(0, 40),
    body: first ? first.name : '',
    approximation: 'user_local_segment',
    segments: [{
      id: 'seg1',
      kind: 'segment',
      start: [0, 0, 0],
      end: [0.1, 0, 0],
      radius: 0.002,
    }],
  })
}

function addSegment(tool) {
  tool.segments.push({
    id: `seg${tool.segments.length + 1}`.slice(0, 40),
    kind: 'segment',
    start: [0, 0, 0],
    end: [0.05, 0, 0],
    radius: 0.002,
  })
}

function fillZoneWithEef(zone) {
  if (!eef.value) {
    message.value = '当前没有末端坐标，无法填入。'
    return
  }
  zone.center = eef.value.slice()
}

function fillFirstSetWithEef(tool) {
  const segment = tool.segments[0]
  if (!segment) return
  if (!eef.value) {
    message.value = '当前没有末端坐标，无法填入。'
    return
  }
  // 末端世界坐标不能直接当工具局部坐标；这里只作为起点提示，需用户再校准。
  segment.start = [0, 0, 0]
  message.value = '局部段以工具 body 为原点，请按实际工具几何填写，不要直接用世界坐标。'
}

/** 前端预检，与服务端校验保持同一口径；通过后由服务端最终校验。 */
function validate() {
  const value = canonical(model.value)
  const problems = []
  if (value.enabled) {
    if (!value.risk_zones.length && !value.tool_annotations.length) {
      problems.push('启用后至少需要一个保护区与一个工具标注，否则后端只会返回 unknown。')
    }
  }
  const zoneIds = new Set()
  value.risk_zones.forEach((zone, index) => {
    if (!idPattern.test(zone.id)) problems.push(`保护区 #${index + 1} 的 id 只能包含字母数字与 _.-（1-40 位）`)
    if (zoneIds.has(zone.id)) problems.push(`保护区 id 重复：${zone.id}`)
    zoneIds.add(zone.id)
    if (!(zone.radius > 0)) problems.push(`保护区 ${zone.id} 的半径必须大于 0`)
    if (zone.clearance < 0) problems.push(`保护区 ${zone.id} 的余量不能为负`)
    for (const point of zone.kind === 'segment' ? [zone.start, zone.end] : [zone.center]) {
      if (!Array.isArray(point) || point.length !== 3 || point.some((v) => !Number.isFinite(Number(v)))) {
        problems.push(`保护区 ${zone.id} 的坐标必须是 3 个有限数值`)
      }
    }
  })
  const toolIds = new Set()
  value.tool_annotations.forEach((tool) => {
    if (!idPattern.test(tool.tool_id)) problems.push(`工具 id ${tool.tool_id || '(空)'} 不合法`)
    if (toolIds.has(tool.tool_id)) problems.push(`工具 id 重复：${tool.tool_id}`)
    toolIds.add(tool.tool_id)
    if (!tool.body || !tool.body.trim()) problems.push(`工具 ${tool.tool_id} 必须选择或填写 body`)
    const segmentIds = new Set()
    tool.segments.forEach((segment, index) => {
      if (!idPattern.test(segment.id)) problems.push(`工具 ${tool.tool_id} 的第 ${index + 1} 段 id 不合法`)
      if (segmentIds.has(segment.id)) problems.push(`工具 ${tool.tool_id} 段 id 重复：${segment.id}`)
      segmentIds.add(segment.id)
      if (!(segment.radius >= 0)) problems.push(`段 ${segment.id} 的半径不能为负`)
      for (const point of [segment.start, segment.end]) {
        if (!Array.isArray(point) || point.length !== 3 || point.some((v) => !Number.isFinite(Number(v)))) {
          problems.push(`段 ${segment.id} 的局部坐标必须是 3 个有限数值`)
        }
      }
    })
  })
  return problems
}

function submit() {
  const problems = validate()
  if (problems.length) {
    message.value = problems[0] + (problems.length > 1 ? `（共 ${problems.length} 项）` : '')
    return
  }
  message.value = ''
  const value = canonical(model.value)
  applied.value = JSON.stringify(value)
  emit('update:modelValue', value)
}

function toggleEnabled() {
  model.value.enabled = !model.value.enabled
  submit()
}

function resetToDisabled() {
  model.value = defaultModel()
  message.value = ''
  applied.value = JSON.stringify(canonical(model.value))
  emit('update:modelValue', canonical(model.value))
}

// ------------------------------------------------------------------ 只读读数
const assessment = computed(() => props.safety?.assessment || null)
const metrics = computed(() => props.safety?.metrics || null)
const available = computed(() => Boolean(props.safety) && typeof props.safety === 'object')
// 只有后端真的报了状态才算数；没有观测时按本地启用开关区分“未启用”与“未上报”。
const status = computed(() => {
  const reported = props.safety?.status || assessment.value?.status
  if (reported) return reported
  return model.value.enabled ? 'unknown' : 'disabled'
})

const STATUS_TEXT = {
  disabled: '未启用',
  ok: '未越界',
  violation: '越界违规',
  unknown: '数据不足',
}

const statusText = computed(() => STATUS_TEXT[status.value] || `未知状态：${status.value}`)
const statusClass = computed(() => (['ok', 'violation', 'unknown', 'disabled'].includes(status.value) ? status.value : 'unknown'))

const unknownReasons = computed(() => assessment.value?.unknown_reasons || props.safety?.unknown_reasons || [])

// 配置已提交但当前环境还没带它跑：不能把“未上报”当成“已生效”。
const pendingHint = computed(() => {
  if (!model.value.enabled) return ''
  if (!available.value) return '配置已启用，等待新环境加载后由后端返回观测。'
  return ''
})

function meters(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return null
  return `${Number(value).toFixed(4)} m`
}

const minClearance = computed(() => meters(metrics.value?.min_net_clearance ?? null))
const entries = computed(() => metrics.value?.zone_entries ?? null)
const exposure = computed(() => {
  const value = metrics.value?.exposure_seconds
  return value !== null && value !== undefined && Number.isFinite(Number(value)) ? `${Number(value).toFixed(2)} s` : null
})
const minDistanceRows = computed(() => {
  const rows = assessment.value?.zone_clearance
  return Array.isArray(rows) ? rows : []
})
const events = computed(() => (Array.isArray(props.safety?.events) ? props.safety.events : []))

const toolApproximationNote = '用户设定的局部段是几何近似，可能不是刀刃的真实形状；需要按工具实测校准，平台不将其认定为刀刃，也不由它推导伤害程度。'
</script>

<template>
  <section class="panel holo-shell tool-safety">
    <div class="head">
      <h2>{{ title }}</h2>
      <span class="st" :class="statusClass">{{ statusText }}</span>
    </div>
    <p v-if="showConfig" class="note">配置将在下次加载环境时生效。</p>
    <p v-if="message" class="warn" role="alert">{{ message }}</p>
    <p v-else-if="pendingHint" class="note">{{ pendingHint }}</p>

    <div class="cols" :class="{ single: singleMode }">
      <div v-if="showConfig" class="config">
        <div class="rowline">
          <button class="primary" :disabled="disabled" @click="toggleEnabled">
            {{ model.enabled ? '停用工具安全' : '启用工具安全' }}
          </button>
          <button :disabled="disabled" @click="resetToDisabled">清空并保持默认关闭</button>
        </div>

        <h3>保护区（世界坐标 · m · {{ frame.up }} 向上）</h3>
        <div v-for="(zone, index) in model.risk_zones" :key="`zone-${index}`" class="card">
          <label>ID
            <input v-model.trim="zone.id" :disabled="disabled" />
          </label>
          <div class="row2">
            <label>形状
              <select v-model="zone.kind" :disabled="disabled">
                <option value="sphere">球（center + radius）</option>
                <option value="segment">胶囊段（start/end + radius）</option>
              </select>
            </label>
            <label>半径 m
              <input v-model.number="zone.radius" type="number" step="0.01" min="0" :disabled="disabled" />
            </label>
          </div>
          <label v-if="zone.kind === 'sphere'">球心 x, y, z (world, m)
            <input :value="(zone.center || []).join(', ')" :disabled="disabled" @input="zone.center = $event.target.value.split(',').map((v) => Number(v.trim()))" />
          </label>
          <template v-else>
            <label>起点 x, y, z (world, m)
              <input :value="(zone.start || []).join(', ')" :disabled="disabled" @input="zone.start = $event.target.value.split(',').map((v) => Number(v.trim()))" />
            </label>
            <label>终点 x, y, z (world, m)
              <input :value="(zone.end || []).join(', ')" :disabled="disabled" @input="zone.end = $event.target.value.split(',').map((v) => Number(v.trim()))" />
            </label>
          </template>
          <label>余量 clearance m
            <input v-model.number="zone.clearance" type="number" step="0.005" min="0" :disabled="disabled" />
          </label>
          <div class="rowline">
            <button :disabled="disabled || !eef" @click="fillZoneWithEef(zone)">用当前末端坐标填球心</button>
            <button :disabled="disabled" @click="model.risk_zones.splice(index, 1)">删除保护区</button>
          </div>
          <p v-if="!eef" class="mini">当前没有末端坐标，填充按钮不可用。</p>
        </div>
        <div class="rowline">
          <button :disabled="disabled" @click="addZone">添加保护区</button>
          <span v-if="eef" class="mini">当前末端 world：{{ eef.map((v) => v.toFixed(3)).join(', ') }} m</span>
        </div>

        <h3>工具（局部段，相对工具 body 原点）</h3>
        <div v-for="(tool, index) in model.tool_annotations" :key="`tool-${index}`" class="card">
          <div class="row2">
            <label>工具 ID
              <input v-model.trim="tool.tool_id" :disabled="disabled" />
            </label>
            <label>近似类型
              <input v-model.trim="tool.approximation" :disabled="disabled" />
            </label>
          </div>
          <label>工具 body（来自场景对象，或手动填写）
            <input v-model.trim="tool.body" :list="bodiesListId" :disabled="disabled" />
          </label>
          <datalist :id="bodiesListId">
            <option v-for="item in objects" :key="item.id" :value="item.name">{{ item.id }}</option>
          </datalist>
          <p class="mini">{{ toolApproximationNote }}</p>

          <div v-for="(segment, segIndex) in tool.segments" :key="`seg-${index}-${segIndex}`" class="segment">
            <div class="row2">
              <label>段 ID
                <input v-model.trim="segment.id" :disabled="disabled" />
              </label>
              <label>段半径 m
                <input v-model.number="segment.radius" type="number" step="0.001" min="0" :disabled="disabled" />
              </label>
            </div>
            <label>局部起点 x, y, z (相对 body, m)
              <input :value="(segment.start || []).join(', ')" :disabled="disabled" @input="segment.start = $event.target.value.split(',').map((v) => Number(v.trim()))" />
            </label>
            <label>局部终点 x, y, z (相对 body, m)
              <input :value="(segment.end || []).join(', ')" :disabled="disabled" @input="segment.end = $event.target.value.split(',').map((v) => Number(v.trim()))" />
            </label>
            <div class="rowline">
              <button :disabled="disabled" @click="tool.segments.splice(segIndex, 1)">删除段</button>
              <button :disabled="disabled" @click="fillFirstSetWithEef(tool)">局部坐标填写提示</button>
            </div>
          </div>
          <div class="rowline">
            <button :disabled="disabled" @click="addSegment(tool)">添加段</button>
            <button :disabled="disabled" @click="model.tool_annotations.splice(index, 1)">删除工具</button>
          </div>
        </div>
        <div class="rowline">
          <button :disabled="disabled" @click="addTool">添加工具</button>
        </div>

        <div class="rowline">
          <button class="primary" :disabled="disabled || !dirty" @click="submit">提交配置</button>
          <span v-if="dirty" class="mini">有未提交的修改</span>
          <span v-else class="mini">已提交，等待新环境加载生效</span>
        </div>
      </div>

      <div v-if="showReadout" class="readout">
        <div v-if="status === 'disabled'" class="safety-empty">
          <div class="safety-symbol" aria-hidden="true"><svg viewBox="0 0 48 48"><path d="m24 7 13 5v12c0 8-6 13-13 17-7-4-13-9-13-17V12Z"/><path d="M17 24h14m-7-7v14"/></svg></div>
          <h3>为实验添加保护区</h3>
          <p>追踪工具距离与越界事件</p>
          <button class="configure-safety" @click="emit('configure')">配置观测 <span aria-hidden="true">↗</span></button>
        </div>
        <template v-else>
          <div class="kpi safety-distance" :class="{ 'is-empty': !available }">
            <span class="kpi-label">最小净距</span>
            <span class="kpi-value">{{ minClearance ?? '—' }}</span>
          </div>
          <div class="safety-stats"><div><span>进入保护区</span><strong>{{ entries ?? '—' }}<small v-if="entries !== null"> 次</small></strong></div><div><span>暴露时间</span><strong>{{ exposure ?? '—' }}</strong></div></div>
          <div v-if="minDistanceRows.length" class="zone-readings"><h3>分区净距</h3><ul class="rows"><li v-for="row in minDistanceRows" :key="row.zone_id"><span>{{ row.zone_id }}</span><span>{{ meters(row.min_net_clearance) ?? '未知' }}</span><span class="mini">{{ row.status }}</span></li></ul></div>
          <div v-if="unknownReasons.length" class="unknown-readings"><h3>待确认</h3><ul class="rows"><li v-for="reason in unknownReasons" :key="reason"><span class="mono">{{ reason }}</span></li></ul></div>
          <div v-if="events.length" class="safety-events"><h3>最近事件</h3><ul class="rows events"><li v-for="event in events.slice(-4)" :key="event.event_id"><span>{{ event.zone_id }}</span><span>{{ meters(event.net_clearance) }}</span><span class="mini">step {{ event.step }} · {{ event.disposition }}</span></li></ul></div>
          <details class="assessment-details"><summary>观测说明</summary><p class="mini">几何近似观测，不自动停机。暴露时间由离散采样估计，未知步不累积。</p><p class="mini">{{ metrics?.session_id || '—' }} · 回合 {{ metrics?.episode ?? '—' }} · step {{ metrics?.step ?? '—' }}</p></details>
        </template>
      </div>
    </div>
  </section>
</template>

<style scoped>
.tool-safety {
  padding: 14px 16px;
  border-radius: var(--radius);
  background: var(--panel);
  border: 1px solid var(--line);
  min-width: 0;
}

.head {
  display: flex;
  align-items: center;
  gap: 10px;
}

h2 {
  margin: 0;
  font-size: 12px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 500;
}

h3 {
  margin: 14px 0 6px;
  font-size: 12px;
  color: var(--muted);
  font-weight: 500;
}

/* 状态徽章不走全息渐变：风险含义必须固定，且除颜色外另有图形与文字。 */
.st {
  margin-left: auto;
  padding: 3px 10px;
  border-radius: 999px;
  border: 1px solid var(--line);
  font-size: 11px;
  background: rgba(255, 255, 255, 0.06);
  color: var(--muted);
}

.st::before {
  margin-right: 6px;
}

.st.ok {
  color: #d9fff3;
  border-color: rgba(53, 224, 161, 0.5);
  background: rgba(53, 224, 161, 0.12);
}

.st.ok::before { content: '\2714'; }

.st.violation {
  color: #ffd7de;
  border-color: rgba(255, 92, 122, 0.6);
  background: rgba(255, 92, 122, 0.14);
}

.st.violation::before { content: '\25B2'; }

.st.unknown {
  color: #ffe9bd;
  border-color: rgba(255, 204, 77, 0.55);
  background: rgba(255, 204, 77, 0.12);
}

.st.unknown::before { content: '\26A0'; }

.st.disabled::before { content: '\25A0'; }

.note,
.mini {
  margin: 6px 0 0;
  color: var(--muted);
  font-size: 11px;
  line-height: 1.5;
}

.warn {
  margin: 8px 0 0;
  padding: 8px 10px;
  border-radius: 10px;
  background: rgba(255, 204, 77, 0.12);
  border: 1px solid rgba(255, 204, 77, 0.4);
  color: #ffe9bd;
  font-size: 12px;
}

.cols {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
  gap: 14px;
  margin-top: 10px;
}

/* 单模式（只配置或只读数）：一列，避免窄边栏里两列挤压 */
.cols.single {
  grid-template-columns: minmax(0, 1fr);
}

.config,
.readout {
  min-width: 0;
}

label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 8px;
  color: var(--muted);
  font-size: 11px;
}

input,
select {
  padding: 6px 9px;
  border-radius: 10px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.05);
  color: var(--ink);
  font-size: 12px;
  min-width: 0;
}

.row2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.rowline {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin: 6px 0;
}

button {
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.05);
  font-size: 12px;
}

.segment {
  margin: 6px 0;
  padding: 8px;
  border-radius: 10px;
  border: 1px dashed var(--line);
}

dl {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 4px 10px;
  margin: 8px 0;
  font-size: 12px;
}

dt { color: var(--muted); }
dd { margin: 0; font-variant-numeric: tabular-nums; }

.rows {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
}

.rows li {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 5px 8px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.07);
}

.mono { font-family: ui-monospace, 'Cascadia Mono', Consolas, monospace; }

@media (max-width: 980px) {
  .cols { grid-template-columns: 1fr; }
  .row2 { grid-template-columns: 1fr; }
}

.head h2{margin-bottom:0}.safety-empty{display:flex;flex-direction:column;align-items:center;text-align:center;padding:22px 8px 8px}.safety-symbol{width:68px;height:68px;display:grid;place-items:center;border:1px solid #e3e9dd;background:radial-gradient(circle at 40% 25%,#f7faf4,#eaf0e5);border-radius:24px;transform:rotate(-6deg);margin-bottom:8px}.safety-symbol svg{width:37px;height:37px;fill:none;stroke:#6d8c77;stroke-width:1.3;stroke-linecap:round;stroke-linejoin:round;transform:rotate(6deg)}.safety-empty h3{margin:12px 0 0;font-size:14px;color:#4c6357;font-weight:500}.safety-empty p{margin:7px 0 20px;font-size:12px;color:var(--muted)}.safety-empty .configure-safety{display:flex;align-items:center;justify-content:space-between;gap:36px;min-height:44px;padding:10px 20px;background:#edf3e9;border:1px solid #d7e2d1;color:#3b5f4b;border-radius:999px;font-size:12px}.configure-safety:hover{background:#e2eddd}.safety-distance{margin-top:16px}.safety-stats{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:18px 0}.safety-stats>div{display:flex;flex-direction:column;gap:8px;font-size:12px;color:var(--muted)}.safety-stats strong{font-size:20px;color:var(--ink);font-weight:500}.safety-stats small{font-size:11px;font-weight:400}.assessment-details{border-top:1px solid var(--line);margin-top:16px;padding-top:8px}.assessment-details summary{min-height:32px;cursor:pointer;font-size:11px;color:var(--muted)}.assessment-details p{overflow-wrap:anywhere}
</style>

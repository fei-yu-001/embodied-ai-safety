<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { getToken, api } from '../api.js'

const form = reactive({
  task: 'robocasa/PickPlaceCounterToCabinet',
  policy: 'vision_centroid',
  attack: 'spoof_patch',
  defense: 'none',
  seeds: '0',
  steps: 40,
})

const evaluations = ref([])
const selected = ref(null)
const error = ref('')
const busy = ref(false)
let timer = null

const current = computed(() => evaluations.value.find((item) => item.id === selected.value) || evaluations.value[0] || null)
const gate = computed(() => current.value?.gate || null)
const groups = computed(() => current.value?.groups || [])
const running = computed(() => current.value?.status?.state === 'running')

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) }
  const token = getToken()
  if (token) headers['X-Console-Token'] = token
  if (options.body) headers['Content-Type'] = 'application/json'
  const response = await fetch(path, { ...options, headers })
  const text = await response.text()
  const data = text ? JSON.parse(text) : null
  if (!response.ok) throw new Error((data && (data.message || data.code)) || `HTTP ${response.status}`)
  return data
}

async function load() {
  try {
    const data = await request('/api/v1/evaluations')
    evaluations.value = data.evaluations || []
  } catch (failure) {
    error.value = failure.message
  }
}

async function start() {
  busy.value = true
  error.value = ''
  try {
    const spec = {
      task: form.task,
      policy: form.policy,
      attack: form.attack,
      defense: form.defense,
      steps: Number(form.steps),
      seeds: form.seeds.split(',').map((value) => Number(value.trim())).filter((value) => Number.isInteger(value)),
    }
    const created = await request('/api/v1/evaluations', { method: 'POST', body: JSON.stringify({ spec }) })
    selected.value = created.id
    await load()
  } catch (failure) {
    error.value = failure.message
  } finally {
    busy.value = false
  }
}

function pick(item) {
  selected.value = item.id
}

onMounted(() => {
  load()
  timer = setInterval(load, 2500)
})
onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
})

const pct = () => {
  const status = current.value?.status
  if (!status || !status.total) return 0
  return Math.round((status.completed / status.total) * 100)
}
</script>

<template>
  <div class="eval">
    <p v-if="error" class="banner" role="alert">{{ error }}</p>

    <div class="grid">
      <section class="panel">
        <h2>新建评测</h2>
        <label>任务<input v-model="form.task" /></label>
        <label>策略<select v-model="form.policy">
          <option value="vision_centroid">vision_centroid（视觉，非 VLA）</option>
          <option value="random">random（不使用视觉）</option>
          <option value="hold">hold（静止）</option>
        </select></label>
        <div class="row">
          <label>攻击<select v-model="form.attack">
            <option value="spoof_patch">纯色补丁（spoof）</option>
            <option value="adaptive_patch">自适应补丁（纹理+低饱和+抖动）</option>
            <option value="adaptive_texture">纹理补丁</option>
            <option value="noise_patch">随机噪点</option>
            <option value="none">不攻击</option>
          </select></label>
          <label>防御<select v-model="form.defense">
            <option value="none">不防御</option>
            <option value="temporal">时序一致性</option>
            <option value="median">中值滤波</option>
          </select></label>
        </div>
        <div class="row">
          <label>种子<input v-model="form.seeds" /></label>
          <label>步数<input v-model.number="form.steps" type="number" min="1" max="400" /></label>
        </div>
        <button class="primary" :disabled="busy || running" @click="start">
          {{ running ? '评测进行中…' : busy ? '提交中…' : '开始评测' }}
        </button>
        <p class="note">四组对照会各加载一次厨房，纯 CPU 下约需数分钟。门禁只看干净组。</p>
      </section>

      <section class="panel">
        <div class="head">
          <h2>评测记录</h2>
          <span v-if="current" class="chip" :class="current.status?.state">{{ current.status?.state }}</span>
        </div>
        <ul class="list">
          <li v-for="item in evaluations" :key="item.id" :class="{ on: item.id === current?.id }" @click="pick(item)">
            <span class="id">{{ item.id.slice(5, 20) }}</span>
            <span class="mini">{{ item.spec?.attack }} / {{ item.spec?.defense }}</span>
            <span class="mini">{{ item.status?.completed }}/{{ item.status?.total }}</span>
          </li>
          <li v-if="!evaluations.length" class="empty">还没有评测</li>
        </ul>
      </section>
    </div>

    <section v-if="current" class="panel result">
      <div class="head">
        <h2>结果 · {{ current.id }}</h2>
        <div class="bar"><div class="fill" :style="{ width: pct() + '%' }" /></div>
        <span class="mini">{{ current.status?.current || '—' }}</span>
      </div>

      <div v-if="gate" class="gate" :class="{ pass: gate.eligible_for_task_eval }">
        <strong>{{ gate.eligible_for_task_eval ? '门禁通过' : '门禁未通过' }}</strong>
        <span>干净基线成功率 {{ gate.clean_success_rate ?? '—' }} / 阈值 {{ gate.threshold }}</span>
        <span class="why">{{ gate.reason }}</span>
      </div>
      <p v-else class="note">还没有门禁结论，评测仍在进行。</p>

      <table v-if="groups.length">
        <thead>
          <tr><th>组</th><th>成功</th><th>运行数</th><th>平均|动作X|</th><th>平均|画面偏移|</th><th>与干净组散度</th><th>回收比例</th><th>防御耗时 ms</th><th>失败运行</th></tr>
        </thead>
        <tbody>
          <tr v-for="group in groups" :key="group.group">
            <td>{{ group.group }}</td>
            <td>{{ group.success_observed ? '是' : '否' }}</td>
            <td>{{ group.runs }}</td>
            <td>{{ group.mean_abs_action_x ?? '—' }}</td>
            <td>{{ group.mean_abs_offset_x ?? '—' }}</td>
            <td>{{ group.divergence_vs_clean ?? '—' }}</td>
            <td>{{ group.recovery_ratio === null || group.recovery_ratio === undefined ? '—' : (group.recovery_ratio * 100).toFixed(1) + '%' }}</td>
            <td>{{ group.mean_defense_latency_ms ?? '—' }}</td>
            <td>{{ group.failed_runs }}</td>
          </tr>
        </tbody>
      </table>

      <p class="note">{{ current.summary?.interpretation || '四组对照：干净/受攻击 × 无防御/有防御。' }}</p>
    </section>
  </div>
</template>

<style scoped>
.eval {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.grid {
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr);
  gap: 12px;
}

.panel {
  padding: 16px;
  border-radius: var(--radius);
  background: var(--panel);
  border: 1px solid var(--line);
  min-width: 0;
}

h2 {
  margin: 0 0 10px;
  font-size: 12px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 500;
}

label {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin-bottom: 10px;
  color: var(--muted);
  font-size: 12px;
}

input,
select {
  padding: 7px 10px;
  border-radius: 10px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.05);
  color: var(--ink);
  font-size: 13px;
}

.row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.primary {
  width: 100%;
  padding: 9px 0;
  border-radius: 999px;
  border: none;
  font-weight: 700;
  color: #fff;
  background: linear-gradient(90deg, var(--violet), var(--magenta), var(--cyan));
  background-size: 200% auto;
}

.note {
  margin: 8px 0 0;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.5;
}

.head {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.bar {
  flex: 1;
  min-width: 120px;
  height: 6px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
}

.fill {
  height: 100%;
  background: linear-gradient(90deg, var(--violet), var(--cyan));
  transition: width 0.4s ease;
}

.chip {
  margin-left: auto;
  padding: 3px 10px;
  border-radius: 999px;
  border: 1px solid var(--line);
  font-size: 11px;
  color: var(--muted);
}

.chip.completed { color: #d9fff3; border-color: rgba(53, 224, 161, 0.5); }
.chip.failed { color: #ffd7de; border-color: rgba(255, 92, 122, 0.5); }

.list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 320px;
  overflow: auto;
}

.list li {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 10px;
  padding: 8px 10px;
  border-radius: 10px;
  border: 1px solid transparent;
  cursor: pointer;
  font-size: 12px;
}

.list li:hover { background: rgba(255, 255, 255, 0.05); }
.list li.on { border-color: rgba(0, 212, 255, 0.5); background: rgba(0, 212, 255, 0.08); }
.list li.empty { cursor: default; color: var(--muted); }

.mini { color: var(--muted); font-size: 11px; }

.gate {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(255, 92, 122, 0.1);
  border: 1px solid rgba(255, 92, 122, 0.4);
  color: #ffd7de;
  font-size: 12px;
  margin-bottom: 10px;
}

.gate.pass {
  background: rgba(53, 224, 161, 0.1);
  border-color: rgba(53, 224, 161, 0.45);
  color: #d9fff3;
}

.gate .why { opacity: 0.85; }

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

th,
td {
  text-align: left;
  padding: 6px 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.07);
}

th { color: var(--muted); font-weight: 500; }

.banner {
  margin: 0;
  padding: 10px 14px;
  border-radius: 12px;
  background: rgba(255, 92, 122, 0.14);
  border: 1px solid rgba(255, 92, 122, 0.4);
  color: #ffd7de;
}

@media (max-width: 980px) {
  .grid { grid-template-columns: 1fr; }
}
</style>

<script setup>
import { computed, ref, watch } from 'vue'

const props = defineProps({ scene: Object, session: Object, selected: String })
const emit = defineEmits(['select', 'edit'])

const kind = ref('position')
const position = ref({ x: 0, y: 0, z: 0 })
const mass = ref(1)
const note = ref('')
const dirty = ref(false)

const tree = computed(() => props.scene?.tree || [])
const editableState = computed(() => ['READY', 'PAUSED'].includes(props.session?.state))
const current = computed(() => tree.value.find((item) => item.name === props.selected) || null)

function syncFromServer() {
  const item = current.value
  if (!item) return
  position.value = { x: item.position[0], y: item.position[1], z: item.position[2] }
  mass.value = item.mass
  if (!item.editable.includes(kind.value)) kind.value = item.editable[0] || 'mass'
}

// 轮询会不断替换 scene 对象；用户改过但没提交的字段不能被服务端值冲掉。
watch(
  () => [props.selected, props.scene?.scene_revision],
  () => {
    if (dirty.value) return
    syncFromServer()
  },
  { immediate: true },
)

watch(
  () => props.selected,
  () => {
    dirty.value = false
    syncFromServer()
  },
)

function apply() {
  note.value = ''
  dirty.value = true
  if (!current.value) return
  const args =
    kind.value === 'mass'
      ? { target: current.value.name, kind: 'mass', values: Number(mass.value) }
      : {
          target: current.value.name,
          kind: 'position',
          values: [Number(position.value.x), Number(position.value.y), Number(position.value.z)],
        }
  emit('edit', args)
}

function step(axis, delta) {
  dirty.value = true
  const next = { ...position.value }
  next[axis] = Number((Number(next[axis]) + delta).toFixed(3))
  position.value = next
}

function showNote(message) {
  note.value = message
}

defineExpose({ showNote })
</script>

<template>
  <section class="panel">
    <h2>场景树</h2>
    <p v-if="!tree.length" class="muted">还没有场景数据</p>
    <ul class="tree">
      <li
        v-for="item in tree"
        :key="item.name"
        :class="{ on: item.name === selected }"
        @click="emit('select', item.name)"
      >
        <span class="name">{{ item.name }}</span>
        <span class="tags">{{ item.editable.join('/') }}</span>
      </li>
    </ul>

    <template v-if="current">
      <h2>编辑 · {{ current.name }}</h2>
      <p v-if="!editableState" class="warn">只有就绪或暂停时可以编辑；运行中先暂停。</p>

      <div class="tabs">
        <button
          v-for="option in current.editable"
          :key="option"
          :class="{ on: kind === option }"
          @click="kind = option"
        >{{ option === 'position' ? '位姿' : '质量' }}</button>
      </div>

      <template v-if="kind === 'position'">
        <div v-for="axis in ['x', 'y', 'z']" :key="axis" class="row">
          <label>{{ axis.toUpperCase() }}</label>
          <button
            class="mini"
            :disabled="!editableState"
            :data-testid="`step-${axis}-minus`"
            @click="step(axis, -0.01)"
          >−</button>
          <input
            v-model.number="position[axis]"
            type="number"
            step="0.01"
            :disabled="!editableState"
            :data-testid="`edit-${axis}`"
          />
          <button
            class="mini"
            :disabled="!editableState"
            :data-testid="`step-${axis}-plus`"
            @click="step(axis, 0.01)"
          >+</button>
        </div>
        <p class="hint">单位米，世界系。越出工作区会被拒绝。</p>
      </template>

      <template v-else>
        <div class="row">
          <label>kg</label>
          <input v-model.number="mass" type="number" step="0.1" min="0.001" :disabled="!editableState" />
        </div>
        <p class="hint">改质量会重建模型常量，下一次重置恢复原值。</p>
      </template>

      <button class="apply" :disabled="!editableState" @click="apply">应用修改</button>
      <p v-if="note" class="note">{{ note }}</p>
    </template>
  </section>
</template>

<style scoped>
.panel {
  padding: 14px;
  border-radius: var(--radius);
  background: var(--panel);
  border: 1px solid var(--line);
  min-width: 0;
}

h2 {
  margin: 0 0 8px;
  font-size: 12px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 500;
}

h2:not(:first-child) {
  margin-top: 16px;
}

.tree {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 180px;
  overflow: auto;
}

.tree li {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 10px;
  border: 1px solid transparent;
  cursor: pointer;
  font-size: 12px;
}

.tree li:hover {
  background: rgba(255, 255, 255, 0.05);
}

.tree li.on {
  border-color: rgba(0, 212, 255, 0.5);
  background: rgba(0, 212, 255, 0.1);
}

.tags {
  color: var(--muted);
  font-size: 11px;
}

.tabs {
  display: flex;
  gap: 6px;
  margin-bottom: 8px;
}

.tabs button {
  flex: 1;
  padding: 5px 0;
  border-radius: 10px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.05);
  color: var(--muted);
  font-size: 12px;
}

.tabs button.on {
  color: #fff;
  border-color: rgba(168, 85, 247, 0.6);
  background: rgba(168, 85, 247, 0.18);
}

.row {
  display: grid;
  grid-template-columns: 28px 26px 1fr 26px;
  gap: 6px;
  align-items: center;
  margin-bottom: 6px;
}

.row label {
  color: var(--muted);
  font-size: 12px;
}

input {
  width: 100%;
  padding: 5px 8px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.05);
  color: var(--ink);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.mini {
  padding: 4px 0;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.05);
  font-size: 12px;
}

.apply {
  width: 100%;
  margin-top: 8px;
  padding: 8px 0;
  border-radius: 10px;
  border: none;
  background: linear-gradient(90deg, var(--violet), var(--magenta), var(--cyan));
  color: #fff;
  font-weight: 600;
}

.hint,
.note {
  margin: 6px 0 0;
  font-size: 11px;
  color: var(--muted);
}

.warn {
  margin: 0 0 8px;
  font-size: 12px;
  color: #ffe9b0;
}

.muted {
  color: var(--muted);
  margin: 0;
}
</style>

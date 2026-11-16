<script setup>
import { computed } from 'vue'
import { JOG_PROFILES } from '../jogSettings.js'
const props = defineProps({ session: Object, holding: String, busy: Boolean, profile: { type: String, default: 'standard' } })
const emit = defineEmits(['start', 'end', 'grip', 'update:profile'])
const directions = [
  { id: 'y+', label: '+Y', area: 'up', angle: 0 },
  { id: 'x-', label: '−X', area: 'left', angle: -90 },
  { id: 'x+', label: '+X', area: 'right', angle: 90 },
  { id: 'y-', label: '−Y', area: 'down', angle: 180 },
]
function canJog() { return Boolean(props.session?.live && props.session?.environment?.capabilities?.jog) && ['READY','PAUSED','RUNNING'].includes(props.session?.state) && !props.busy }
function down(event, direction) {
  if (!canJog() || props.holding || (event.type === 'pointerdown' && event.button !== 0)) return
  if (event.pointerId !== undefined) event.currentTarget.setPointerCapture(event.pointerId)
  emit('start', direction)
}
function keyboard(event, direction) { if (!event.repeat) down(event, direction) }
const travel = computed(() => {
  const vector = props.session?.environment?.manual?.measured_m
  const last = props.session?.last_command
  const measured = last?.jog?.measured_m || vector
  return Array.isArray(measured) && measured.every(Number.isFinite) ? (Math.hypot(...measured) * 1000).toFixed(1) : null
})
const blocked = computed(() => !props.session?.live ? '请先加载环境' : !props.session?.environment?.capabilities?.jog ? '当前机器人不支持点动' : '')
</script>

<template>
  <section class="motion-console" aria-label="机械臂手动控制">
    <header class="motion-header">
      <div class="motion-title"><span class="status-dot" :class="{ moving: holding }" /><h2>机械臂控制</h2><span class="motion-state">{{ holding ? '移动中' : '按住移动' }}</span></div>
      <div class="motion-profiles" role="group" aria-label="移动幅度">
        <button v-for="item in JOG_PROFILES" :key="item.id" :aria-pressed="profile === item.id" :disabled="Boolean(holding) || busy" @click="emit('update:profile', item.id)">{{ item.label }}</button>
      </div>
    </header>
    <div class="motion-body">
      <div class="planar-control">
        <div class="direction-dial" aria-label="XY 平面移动，基座坐标系">
          <button v-for="direction in directions" :key="direction.id" class="direction-key" :class="[direction.area, { active: holding === direction.id }]" :aria-label="direction.label" :title="direction.label + ' · 基座坐标系'" :disabled="!canJog() || (Boolean(holding) && holding !== direction.id)" @pointerdown.prevent="down($event, direction.id)" @pointerup="emit('end')" @pointercancel="emit('end')" @lostpointercapture="emit('end')" @keydown.space.prevent="keyboard($event, direction.id)" @keyup.space.prevent="emit('end')" @keydown.enter.prevent="keyboard($event, direction.id)" @keyup.enter.prevent="emit('end')" @blur="emit('end')">
            <svg viewBox="0 0 24 24" aria-hidden="true" :style="{ transform: `rotate(${direction.angle}deg)` }"><path d="M12 19V5m-5 5 5-5 5 5" /></svg><span>{{ direction.label }}</span>
          </button>
          <div class="dial-center" aria-hidden="true"><span /><span />XY</div>
        </div>
        <span class="control-caption">平面移动</span>
      </div>
      <div class="vertical-control">
        <div class="lift-track">
          <button v-for="(direction, index) in ['z+', 'z-']" :key="direction" class="lift-key" :class="{ active: holding === direction }" :aria-label="index === 0 ? '+Z' : '−Z'" :title="index === 0 ? '升高 · +Z' : '降低 · −Z'" :disabled="!canJog() || (Boolean(holding) && holding !== direction)" @pointerdown.prevent="down($event, direction)" @pointerup="emit('end')" @pointercancel="emit('end')" @lostpointercapture="emit('end')" @keydown.space.prevent="keyboard($event, direction)" @keyup.space.prevent="emit('end')" @keydown.enter.prevent="keyboard($event, direction)" @keyup.enter.prevent="emit('end')" @blur="emit('end')"><svg viewBox="0 0 24 24" aria-hidden="true" :style="{ transform: index ? 'rotate(180deg)' : '' }"><path d="m6 13 6-6 6 6M6 18h12" /></svg><span>{{ index ? '−Z' : '+Z' }}</span></button>
        </div>
        <span class="control-caption">升降</span>
      </div>
      <div class="gripper-control">
        <div class="gripper-actions">
          <button :disabled="!canJog() || Boolean(holding)" aria-label="张开夹爪" @click="emit('grip', 'open')"><svg viewBox="0 0 40 32" aria-hidden="true"><path d="M20 4v7M11 11h18M11 11v9l-5 7M29 11v9l5 7M14 24l-4 4M26 24l4 4" /><path class="grip-guide" d="m7 7-4 4 4 4M3 11h6m24-4 4 4-4 4m-2-4h6" /></svg><span>张开</span></button>
          <button :disabled="!canJog() || Boolean(holding)" aria-label="闭合夹爪" @click="emit('grip', 'close')"><svg viewBox="0 0 40 32" aria-hidden="true"><path d="M20 4v7M11 11h18M11 11v9l6 7M29 11v9l-6 7M17 24v5m6-5v5" /><path class="grip-guide" d="m3 7 4 4-4 4m-2-4h6m30-4-4 4 4 4m2-4h-6" /></svg><span>闭合</span></button>
        </div>
        <span class="control-caption">夹爪</span>
      </div>
      <div class="motion-feedback" aria-live="polite">
        <span class="feedback-label">{{ holding ? '正在移动' : '上次位移' }}</span>
        <span class="feedback-value">{{ travel ?? '—' }}<small v-if="travel !== null">mm</small></span>
        <span class="feedback-coordinate">基座坐标系</span>
      </div>
    </div>
    <p v-if="blocked" class="motion-blocked">{{ blocked }}</p>
  </section>
</template>

<style scoped>
.motion-console{container-type:inline-size;padding:18px 20px;background:var(--panel);border:1px solid var(--line);border-radius:24px;box-shadow:var(--shadow-card);color:var(--ink)}
.motion-header,.motion-title{display:flex;align-items:center;gap:10px}.motion-header{justify-content:space-between;gap:16px}.motion-title h2{margin:0;font-size:14px;font-weight:600;white-space:nowrap}.status-dot{width:6px;height:6px;background:#aab9ad;border-radius:50%;flex-shrink:0}.status-dot.moving{background:#3c8272;box-shadow:0 0 0 4px #3c827217}.motion-state{font-size:11px;color:var(--muted);white-space:nowrap}
.motion-profiles{display:flex;padding:3px;border-radius:999px;background:#edf0e9;gap:2px}.motion-profiles button{border:0;background:transparent;color:#74817b;min-height:44px;padding:6px 14px;font-size:12px;border-radius:999px;box-shadow:none}.motion-profiles button[aria-pressed=true]{color:#244c44;background:#fff;box-shadow:0 2px 8px #253e3010}
.motion-body{display:flex;justify-content:space-between;align-items:center;gap:16px;padding:16px 4px 0}.planar-control,.vertical-control,.gripper-control{display:flex;flex-direction:column;align-items:center;gap:9px}.direction-dial{width:144px;height:144px;border-radius:50%;display:grid;grid-template-columns:repeat(3,48px);grid-template-rows:repeat(3,48px);grid-template-areas:'. up .' 'left center right' '. down .';background:radial-gradient(circle at center,#fcfcfa 23%,#eff3ed 24%,#f2f5ef 68%,#e4ebe2 70%,#f4f6f0 72%);border:1px solid #dce5da;box-shadow:inset 0 1px 0 #fff;position:relative;box-sizing:content-box}.direction-key{border:0;background:transparent;padding:3px;min-height:48px;min-width:48px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0;border-radius:17px;touch-action:none;box-shadow:none;color:#3b5b50}.direction-key.up{grid-area:up}.direction-key.down{grid-area:down}.direction-key.left{grid-area:left}.direction-key.right{grid-area:right}.direction-key span{font-size:9px;color:#718377;font-weight:500;line-height:1.1}.direction-key svg{width:21px;height:21px;fill:none;stroke:currentColor;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}.dial-center{grid-area:center;display:flex;align-items:center;justify-content:center;font-size:10px;color:#839185;position:relative}.dial-center span{position:absolute;background:#a4b6a5;opacity:.35;width:26px;height:1px}.dial-center span:nth-child(2){transform:rotate(90deg)}.dial-center{background:#fcfcfa;border-radius:50%;margin:7px;z-index:1}.dial-center span{display:none}
.control-caption{font-size:10px;letter-spacing:.04em;color:#809084}.lift-track{display:flex;flex-direction:column;border:1px solid #e0e7db;border-radius:24px;background:#f4f6f0;padding:4px;gap:4px}.lift-key{min-width:56px;min-height:60px;display:flex;flex-direction:column;align-items:center;justify-content:center;border:0;border-radius:20px;background:transparent;touch-action:none;padding:5px 10px;color:#3b5b50;box-shadow:none;gap:2px}.lift-key svg{width:22px;height:22px;fill:none;stroke:currentColor;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}.lift-key span{font-size:10px;color:#718377}.lift-key+.lift-key{border-top:1px solid #e0e7db;border-top-left-radius:0;border-top-right-radius:0}
.gripper-actions{display:flex;gap:8px}.gripper-actions button{min-width:74px;min-height:96px;padding:16px 12px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;border:1px solid #dfe7db;border-radius:20px;background:#f6f8f2;font-size:12px;color:#3b5b50;box-shadow:inset 0 1px 0 #fff}.gripper-actions svg{width:38px;height:32px;stroke:currentColor;fill:none;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}.grip-guide{opacity:.5;stroke-width:1.2}.motion-feedback{border-left:1px solid #e2e8de;padding:12px 0 12px 20px;min-width:104px;display:flex;flex-direction:column;align-items:flex-start;gap:9px}.feedback-label,.feedback-coordinate{font-size:11px;color:#839084}.feedback-value{font-size:30px;letter-spacing:-.05em;line-height:1;font-variant-numeric:tabular-nums;color:#31584c}.feedback-value small{font-size:12px;letter-spacing:0;margin-left:5px;color:#839084}.feedback-coordinate{font-size:10px}.motion-blocked{margin:12px 0 0;font-size:12px;color:var(--muted)}
.direction-key:hover:not(:disabled),.lift-key:hover:not(:disabled),.gripper-actions button:hover:not(:disabled){background:#e4ede2}.direction-key.active,.lift-key.active{background:#315e53;color:#fff;box-shadow:0 3px 10px #315e5325}.direction-key.active span,.lift-key.active span{color:#edf6ef}.motion-console button:focus-visible{outline:2px solid var(--color-focus,#557b79);outline-offset:3px}.motion-console button:active:not(:disabled){transform:scale(.96)}.motion-console button{transition:background 160ms,transform 160ms,box-shadow 160ms}.motion-console button:disabled{opacity:.4;cursor:not-allowed}.motion-console button.active:disabled{opacity:1}
@container(max-width:640px){.motion-feedback{display:none}.motion-body{justify-content:space-around;gap:14px;padding-inline:0}.motion-state{display:none}}
@container(max-width:420px){.motion-header{gap:8px}.motion-profiles button{padding:6px 10px}.motion-body{gap:10px}.direction-dial{transform:scale(.94);margin:-4px}.gripper-actions{flex-direction:column;gap:4px}.gripper-actions button{min-height:60px;min-width:56px;padding:6px 8px;gap:2px;border-radius:16px}.gripper-actions svg{width:30px;height:26px}.gripper-actions button span{font-size:10px}.lift-key{min-width:48px}.motion-console{padding:16px}}
@media(prefers-reduced-motion:reduce){.motion-console button{transition:none}}
</style>

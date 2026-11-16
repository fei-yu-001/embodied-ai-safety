<script setup>
import LiquidGlass from './LiquidGlass.vue'
const props = defineProps({ session:Object,busy:Boolean,opaque:Boolean,frosted:Boolean })
const emit=defineEmits(['command'])
const enabled=command=>{if(!props.session||props.busy)return false;const current=props.session.state;if(command==='resume'||command==='step'||command==='reset')return ['READY','PAUSED'].includes(current);if(command==='pause')return current==='RUNNING';if(command==='stop')return !!current&&!['CLOSED','FAILED','INTERRUPTED'].includes(current);return false}
</script>
<template>
  <LiquidGlass light class="playback-glass" :opaque="opaque" :frosted="frosted">
    <div class="playback-controls" aria-label="仿真运行控制">
      <button v-if="session?.state === 'RUNNING'" class="play" :disabled="!enabled('pause')" @click="emit('command','pause')"><span aria-hidden="true">Ⅱ</span> 暂停</button>
      <button v-else class="play" :disabled="!enabled('resume')" @click="emit('command','resume')"><span aria-hidden="true">▷</span> 运行</button>
      <button :disabled="!enabled('step')" @click="emit('command','step')">单步</button>
      <button :disabled="!enabled('reset')" @click="emit('command','reset')">重置</button>
      <span class="spacer" />
      <button class="stop" :disabled="!enabled('stop')" @click="emit('command','stop')"><span aria-hidden="true">□</span> 结束会话</button>
    </div>
  </LiquidGlass>
</template>
<style scoped>
.playback-controls{display:flex;align-items:center;gap:4px;padding:6px;min-height:58px;color:inherit}
.playback-controls button{border:0;background:transparent;color:inherit;padding:10px 18px;min-height:44px;font-size:13px;white-space:nowrap}
.playback-controls .play{background:rgba(255,255,255,.94);color:#284b3f;box-shadow:0 2px 6px #17362d0d;font-weight:500;min-width:90px}
.playback-controls .play span{font-size:16px;margin-right:5px}.playback-controls .stop{opacity:.85}.playback-controls .stop:disabled{opacity:.5}.spacer{flex:1}
@media(max-width:640px){.playback-controls{flex-wrap:wrap;gap:0;padding:6px}.playback-controls button{padding:10px 12px;font-size:12px}.playback-controls .play{min-width:76px}.spacer{display:none}.playback-controls .stop{margin-left:auto;padding-left:8px;padding-right:8px}}
</style>

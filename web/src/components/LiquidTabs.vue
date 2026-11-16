<script setup>
import { computed, ref } from 'vue'
import LiquidGlass from './LiquidGlass.vue'
const props = defineProps({ modelValue:String, items:{type:Array,default:()=>[]}, opaque:Boolean, frosted:Boolean })
const emit = defineEmits(['update:modelValue'])
const host=ref(null), dragging=ref(false), candidate=ref(-1)
let pointerId=null
const selected=computed(()=>Math.max(0,props.items.findIndex(item=>item.value===props.modelValue)))
const active=computed(()=>dragging.value&&candidate.value>=0?candidate.value:selected.value)
function indexAt(event) {const r=host.value.getBoundingClientRect();return Math.min(props.items.length-1,Math.max(0,Math.floor((event.clientX-r.left)/r.width*props.items.length)))}
function down(event) {if(event.button!==0)return;pointerId=event.pointerId;host.value.setPointerCapture(pointerId);candidate.value=indexAt(event);dragging.value=true}
function move(event) {if(dragging.value&&event.pointerId===pointerId)candidate.value=indexAt(event)}
function release(event,cancel=false) {if(event.pointerId!==pointerId)return;const next=candidate.value;dragging.value=false;candidate.value=-1;pointerId=null;if(host.value.hasPointerCapture(event.pointerId))host.value.releasePointerCapture(event.pointerId);if(!cancel&&next>=0)emit('update:modelValue',props.items[next].value)}
function key(event,index) {let next=index;if(event.key==='ArrowRight')next=(index+1)%props.items.length;else if(event.key==='ArrowLeft')next=(index-1+props.items.length)%props.items.length;else if(event.key==='Home')next=0;else if(event.key==='End')next=props.items.length-1;else return;event.preventDefault();emit('update:modelValue',props.items[next].value);host.value.querySelectorAll('[role="tab"]')[next]?.focus()}
</script>
<template>
  <LiquidGlass light :opaque="opaque" :frosted="frosted" class="liquid-tabs-surface">
    <div ref="host" class="liquid-tabs" role="tablist" aria-label="场景视图" :class="{ dragging }" :style="{ '--items':items.length,'--active':active }" @pointerdown="down" @pointermove="move" @pointerup="release" @pointercancel="release($event,true)" @lostpointercapture="release($event,true)">
      <span class="tab-indicator" aria-hidden="true" />
      <button v-for="(item,index) in items" :key="item.value" type="button" role="tab" :aria-selected="modelValue===item.value" :tabindex="modelValue===item.value?0:-1" @click="$event.detail === 0 && emit('update:modelValue',item.value)" @keydown="key($event,index)">{{ item.label }}</button>
    </div>
  </LiquidGlass>
</template>
<style scoped>
.liquid-tabs {display:flex;position:relative;padding:5px;touch-action:pan-y;min-height:54px;user-select:none;}
.liquid-tabs button {flex:1;min-height:44px;min-width:88px;padding:0 18px;border:0;border-radius:999px;background:transparent;color:inherit;position:relative;z-index:1;font-weight:500;font-size:13px;white-space:nowrap;}
.tab-indicator {position:absolute;left:5px;top:5px;bottom:5px;width:calc((100% - 10px)/var(--items));border-radius:999px;background:rgba(255,255,255,.22);box-shadow:inset 0 1px 0 rgba(255,255,255,.35);transform:translateX(calc(var(--active)*100%));transition:transform 420ms cubic-bezier(.2,.8,.2,1.08),width 420ms ease;}
.dragging .tab-indicator {transform:translateX(calc(var(--active)*100%)) scaleX(1.025);transition:transform 110ms ease;}
@media(prefers-reduced-motion:reduce){.tab-indicator{transition:none!important}.dragging .tab-indicator{transform:translateX(calc(var(--active)*100%))}}
</style>

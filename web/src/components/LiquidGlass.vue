<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { createRefractionMap } from '../glassOptics.js'
const props = defineProps({ opaque: Boolean, frosted: Boolean, light: Boolean, radius: { type: Number, default: 32 } })
const host = ref(null)
const map = ref(null)
const supported = ref(false)
const id = `prism-glass-${crypto.randomUUID()}`
let observer, pending
const mode = computed(() => props.opaque ? 'opaque' : !props.frosted && supported.value && map.value ? 'refraction' : 'frosted')
function measure() {
  cancelAnimationFrame(pending)
  pending = requestAnimationFrame(() => {
    if (!host.value || props.opaque) return
    const { width, height } = host.value.getBoundingClientRect()
    if (!width || !height) return
    map.value = createRefractionMap(width, height, Math.min(props.radius, height / 2))
  })
}
function highlight(event) {
  const rect = host.value?.getBoundingClientRect()
  if (!rect) return
  host.value.style.setProperty('--glint-x', `${event.clientX - rect.left}px`)
}
onMounted(() => {
  // Syntax + Chromium gate; actual optical output is covered by visual review.
  supported.value = /Chrome|Chromium|Edg\//.test(navigator.userAgent) && CSS.supports('backdrop-filter', `url("#${id}")`)
  observer = new ResizeObserver(measure)
  observer.observe(host.value)
  measure()
})
watch(() => [props.opaque, props.frosted], measure)
onBeforeUnmount(() => { observer?.disconnect(); cancelAnimationFrame(pending) })
</script>

<template>
  <div ref="host" class="liquid-glass" :class="{ 'on-light': light }" :data-glass-mode="mode" :style="{ '--glass-filter': `url(#${id})`, '--glass-radius': `${radius}px` }" @pointermove="highlight">
    <svg class="glass-definitions" aria-hidden="true" width="0" height="0">
      <defs>
        <filter :id="id" x="0" y="0" width="100%" height="100%" color-interpolation-filters="sRGB">
          <feImage v-if="map" :href="map.url" :width="map.width" :height="map.height" preserveAspectRatio="none" result="bevel" />
          <feComponentTransfer in="bevel" result="centred-bevel"><feFuncR type="linear" slope="1" :intercept="-1 / 510" /><feFuncG type="linear" slope="1" :intercept="-1 / 510" /></feComponentTransfer>
          <feDisplacementMap in="SourceGraphic" in2="centred-bevel" :scale="map?.scale || 0" xChannelSelector="R" yChannelSelector="G" />
        </filter>
      </defs>
    </svg>
    <span class="glass-optics" aria-hidden="true" />
    <span class="glass-glint" aria-hidden="true" />
    <div class="glass-content"><slot /></div>
  </div>
</template>

<style scoped>
.liquid-glass { position:relative; isolation:isolate; border-radius:var(--glass-radius); color:#fff; box-shadow:0 8px 24px rgba(19,37,41,.12),inset 0 1px 0 rgba(255,255,255,.65); }
.glass-definitions { position:absolute; pointer-events:none; }
.glass-optics,.glass-glint { position:absolute; inset:0; border-radius:inherit; pointer-events:none; }
.glass-optics { background:rgba(25,47,51,.2); backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); border:1px solid rgba(255,255,255,.2); }
[data-glass-mode="refraction"] > .glass-optics { backdrop-filter:var(--glass-filter); -webkit-backdrop-filter:var(--glass-filter); background:rgba(255,255,255,.12); }
.glass-glint { border-top:1px solid rgba(255,255,255,.72); background:radial-gradient(100px 16px at var(--glint-x,25%) 0,rgba(255,255,255,.27),transparent); }
.glass-content { position:relative; z-index:1; }
.on-light { color:var(--ink); }
.on-light > .glass-optics { background:rgba(239,245,242,.86); border-color:rgba(255,255,255,.75); }
[data-glass-mode="opaque"] > .glass-optics { background:#eff4f1; backdrop-filter:none; -webkit-backdrop-filter:none; }
[data-glass-mode="opaque"] { color:#202929; }
@media (prefers-reduced-transparency:reduce) { .glass-optics { background:#eff4f1!important; backdrop-filter:none!important; -webkit-backdrop-filter:none!important; } .liquid-glass { color:#202929; } }
</style>

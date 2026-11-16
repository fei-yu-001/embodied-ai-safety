/** Inputs to the physical controller, not a promise of measured travel. */
export const JOG_PROFILES = Object.freeze([
  Object.freeze({ id: 'fine', label: '精细', delta: 0.005, repeat: 4 }),
  Object.freeze({ id: 'standard', label: '标准', delta: 0.02, repeat: 5 }),
  Object.freeze({ id: 'fast', label: '快速', delta: 0.035, repeat: 5 }),
])
export const getJogProfile = id => JOG_PROFILES.find(profile => profile.id === id) || JOG_PROFILES[1]

import assert from 'node:assert/strict'
import {createHoldGesture} from '../src/holdGesture.js'
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return{promise,resolve}}
// A release before the network command returns must never start late motion.
{const gate=deferred(),calls=[];const gesture=createHoldGesture({step:()=>{calls.push('step');return gate.promise},begin:()=>calls.push('begin'),stop:()=>calls.push('stop'),onHolding:()=>{},onError:e=>{throw e}});const work=gesture.start('x+');gesture.end();gate.resolve();await work;assert.deepEqual(calls,['step','stop'])}
// Release during lease acquisition still waits for begin and then stops it.
{const gate=deferred(),calls=[];const gesture=createHoldGesture({step:async()=>calls.push('step'),begin:()=>{calls.push('begin');return gate.promise},stop:()=>calls.push('stop'),onHolding:()=>{},onError:e=>{throw e}});const work=gesture.start('x+');await Promise.resolve();gesture.end();gate.resolve();await work;assert.deepEqual(calls,['step','begin','stop'])}
// Errors cannot strand a lease or leave a control active.
{const calls=[];const gesture=createHoldGesture({step:async()=>{throw Error('rejected')},begin:()=>calls.push('begin'),stop:()=>calls.push('stop'),onHolding:x=>calls.push(x),onError:e=>calls.push(e.message)});await gesture.start('x+');assert.deepEqual(calls,['x+','rejected','stop',''])}
// Controls remain occupied until the worker has acknowledged stop.
{const gate=deferred(),states=[];const gesture=createHoldGesture({step:async()=>{},begin:async()=>{},stop:()=>gate.promise,onHolding:x=>states.push(x),onError:e=>{throw e}});const work=gesture.start('x+');gesture.end();await Promise.resolve();assert.deepEqual(states,['x+']);gate.resolve();await work;assert.deepEqual(states,['x+',''])}
console.log('4/4 tap/hold network race tests passed')

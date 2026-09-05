import { installNeuralEngine } from "./neural-engine.js";

const results = document.querySelector("#results");
const runButton = document.querySelector("#run");
const MiB = 1024 * 1024;
const stateWords = 8;
const stateTolerance = 2e-4;

runButton.addEventListener("click", run);

async function run() {
  runButton.disabled = true;
  const report = { status: "RUNNING", checks: [], errors: [], device: null };
  const unhandled = [];
  const onError = (event) => unhandled.push(String(event.error?.stack || event.message));
  const onRejection = (event) => unhandled.push(String(event.reason?.stack || event.reason));
  window.addEventListener("error", onError);
  window.addEventListener("unhandledrejection", onRejection);
  let device;
  let onGpuError;
  try {
    if (!navigator.gpu) throw new Error("WebGPU is unavailable; the QA page intentionally has no CPU fallback");
    const module = makeMockModule();
    const backend = await installNeuralEngine(module);
    device = backend.device;
    onGpuError = (event) => unhandled.push(`WebGPU uncaptured error: ${event.error.message}`);
    device.addEventListener("uncapturederror", onGpuError);
    if (typeof module.gpuCreate !== "function" || typeof module.gpuWindow !== "function") {
      throw new Error("WebGPU backend did not install its Module entry points");
    }
    report.device = {
      maxBufferSize: backend.device.limits.maxBufferSize,
      maxStorageBufferBindingSize: backend.device.limits.maxStorageBufferBindingSize,
    };
    expect(backend.device.limits.maxStorageBufferBindingSize >= 128 * MiB, "adapter exposes a 128 MiB storage binding");
    expect(backend.device.limits.maxBufferSize >= 256 * MiB, "adapter exposes a 256 MiB buffer limit");
    backend.device.lost.then((info) => {
      unhandled.push(`WebGPU device lost: ${info.message || info.reason}`);
    });

    const fixture = await (await fetch("../fixtures/tiny-parity-v1.json")).json();
    await validationChecked(backend.device, async () => {
      await fixtureParity(module, fixture, report);
      await customParity(module, fixture, report);
      await sparseSplitParity(module, fixture, report);
      await activeQueueParity(module, fixture, report);
      await burstParity(module, fixture, report);
    });
    if (unhandled.length) throw new Error(unhandled.join("\n"));
    report.status = "PASS";
  } catch (error) {
    report.status = "FAIL";
    report.errors.push(String(error?.stack || error));
  } finally {
    window.removeEventListener("error", onError);
    window.removeEventListener("unhandledrejection", onRejection);
    device?.removeEventListener("uncapturederror", onGpuError);
    results.textContent = JSON.stringify(report, null, 2);
    runButton.disabled = false;
  }
}

async function validationChecked(device, operation) {
  device.pushErrorScope("validation");
  let value;
  let operationError;
  try {
    value = await operation();
  } catch (error) {
    operationError = error;
  }
  const validationError = await device.popErrorScope();
  if (validationError) throw new Error(`WebGPU validation error: ${validationError.message}`);
  if (operationError) throw operationError;
  return value;
}

async function fixtureParity(Module, fixture, report) {
  const config = fixtureConfig(fixture);
  const reference = new CpuReference(config);
  const gpu = new GpuHarness(Module, config);
  const probes = allNeurons(config.network.neuronCount);
  let expectedSpikes = 0;
  for (let step = 0; step < fixture.run.steps; step += 1) {
    const events = eventsAt(fixture.stimulus.external_counts, step);
    const cpu = reference.step(events);
    const actual = await gpu.run([events], probes);
    const state = await gpu.state();
    const expectedFixtureSpikes = new Uint32Array(config.network.neuronCount);
    for (const event of fixture.expected.spike_events.filter((event) => event.tick === step)) {
      expectedFixtureSpikes[event.neuron] = 1;
    }
    expectEqual(cpu.spikes, expectedFixtureSpikes, `fixture reference spikes tick ${step}`);
    for (let neuron = 0; neuron < config.network.neuronCount; neuron += 1) {
      expectNear(cpu.voltage[neuron], fixture.expected.v_end_mv[step][neuron], 1e-9, `fixture reference voltage step ${step} neuron ${neuron}`);
      expectNear(cpu.conductance[neuron], fixture.expected.g_end_mv[step][neuron], 1e-9, `fixture reference conductance step ${step} neuron ${neuron}`);
    }
    compareState(state, cpu, `fixture tick ${step}`);
    expectedSpikes += expectedFixtureSpikes.reduce((total, spike) => total + spike, 0);
    expectEqual(actual.probeDeltas, cpu.spikes, `fixture probe deltas tick ${step}`);
    expectNumber(actual.total, cpu.counts.reduce((total, count) => total + count, 0), `fixture total spikes tick ${step}`);
  }
  expectEqual(gpu.totalSpikes(), expectedSpikes, "fixture total spikes");
  gpu.destroy();
  record(report, "fixture phase, delay, refractory, and float32 state parity");
}

async function customParity(Module, fixture, report) {
  const base = fixture.parameters;
  const cases = [
    {
      name: "zero-delay recurrent propagation",
      config: testConfig(base, [0, 1, 1], [1], [4], [0], [-52, -52], [0, 0], 0, []),
      schedule: [[{ neuron: 0, count: 1 }], [], [], []],
    },
    {
      name: "signed inhibitory propagation",
      config: testConfig(base, [0, 1, 1], [1], [-4], [0], [-52, -52], [0, 0], 2, []),
      schedule: [[{ neuron: 0, count: 1 }], [], [], [], []],
    },
    {
      name: "silenced source blocks propagation",
      config: testConfig(base, [0, 1, 1], [1], [4], [0], [-52, -52], [0, 0], 2, [0]),
      schedule: [[{ neuron: 0, count: 1 }], [], [], [], []],
    },
  ];
  for (const testCase of cases) {
    await compareSchedule(Module, testCase.config, testCase.schedule, testCase.name);
    record(report, testCase.name);
  }
}

async function sparseSplitParity(Module, fixture, report) {
  const config = testConfig(fixture.parameters, [0, 0, 0], [], [], [0, 1], [-52, -52], [0, 0], 0, []);
  const schedule = [
    [{ neuron: 0, count: 1 }],
    [],
    [{ neuron: 1, count: 2 }],
    [],
    [{ neuron: 0, count: 1 }],
    [],
  ];
  await compareSchedule(Module, config, schedule, "sparse events versus float64 reference");
  const combined = new GpuHarness(Module, config);
  const split = new GpuHarness(Module, config);
  const probes = [0, 1];
  const whole = await combined.run(schedule, probes);
  const splitDeltas = new Uint32Array(probes.length);
  for (const step of schedule) {
    const window = await split.run([step], probes);
    for (let index = 0; index < probes.length; index += 1) splitDeltas[index] += window.probeDeltas[index];
  }
  compareState(await combined.state(), await split.state(), "combined and split sparse state");
  expectEqual(whole.probeDeltas, splitDeltas, "combined and split sparse deltas");
  expectEqual(combined.totalSpikes(), split.totalSpikes(), "combined and split total spikes");
  combined.destroy();
  split.destroy();
  record(report, "sparse events and combined-window continuity");
}

async function compareSchedule(Module, config, schedule, name) {
  const reference = new CpuReference(config);
  const gpu = new GpuHarness(Module, config);
  const probes = allNeurons(config.network.neuronCount);
  for (let step = 0; step < schedule.length; step += 1) {
    const cpu = reference.step(schedule[step]);
    const result = await gpu.run([schedule[step]], probes);
    compareState(await gpu.state(), cpu, `${name} tick ${step}`);
    expectEqual(result.probeDeltas, cpu.spikes, `${name} spikes tick ${step}`);
    expectNumber(result.total, cpu.counts.reduce((total, count) => total + count, 0), `${name} total spikes tick ${step}`);
  }
  gpu.destroy();
}

async function activeQueueParity(Module, fixture, report) {
  const count = 513;
  const targets = Array.from({ length: count }, (_, i) => i);
  const rows = [0];
  const destinations = [];
  const weights = [];
  for (let source = 0; source < count; source += 1) {
    for (let edge = 0; edge < (source === 0 ? 300 : 1); edge += 1) {
      destinations.push((source + edge + 1) % count);
      weights.push(source % 2 ? -1 : 1);
    }
    rows.push(destinations.length);
  }
  for (const delay of [0, 1, 3]) {
    const config = testConfig(fixture.parameters, rows, destinations, weights, targets, targets.map(() => -52), targets.map(() => 0), delay, [11]);
    const all = targets.map(neuron => ({ neuron, count: 1 }));
    const schedule = [all, [], [], [], all, [], [], [], [], []];
    await compareSchedule(Module, config, schedule, `active queue delay ${delay}`);
    const combined = new GpuHarness(Module, config);
    const split = new GpuHarness(Module, config);
    const whole = await combined.run(schedule, targets);
    const deltas = new Uint32Array(count);
    for (const chunk of [schedule.slice(0, 3), schedule.slice(3, 8), schedule.slice(8)]) {
      const result = await split.run(chunk, targets);
      for (let i = 0; i < count; i++) deltas[i] += result.probeDeltas[i];
    }
    compareState(await combined.state(), await split.state(), `active queue split delay ${delay}`);
    expectEqual(whole.probeDeltas, deltas, `active queue split deltas ${delay}`);
    combined.destroy();
    split.destroy();
  }
  record(report, 'active queues: dense stimuli, empty ticks, fan-out, inhibition, delay 0/1/3 and odd-sized windows');
}

async function burstParity(Module, fixture, report) {
  const sources = 65537;
  const count = sources * 2;
  const rows = [0];
  const destinations = [];
  const weights = [];
  for (let source = 0; source < count; source++) {
    const edges = source >= sources ? 0 : source % 257 === 0 ? 769 : 1;
    for (let edge = 0; edge < edges; edge++) {
      destinations.push(sources + (source + edge) % sources);
      weights.push(edges === 1 ? [-32768, 32767, -1, 1, 0][source % 5] : edge % 2 ? -1 : 1);
    }
    rows.push(destinations.length);
  }
  const voltage = Array.from({ length: count }, (_, i) => i < sources ? -40 : -52);
  const config = testConfig(fixture.parameters, rows, destinations, weights, [], voltage, voltage.map(() => 0), 0, []);
  config.parameters.synapse_weight_mv = 0.25;
  await compareSchedule(Module, config, [[]], '65537 simultaneous sources, signed i16 extremes and odd edge count');
  record(report, 'burst exceeds one-dimensional workgroup limit without truncation; uneven fan-out and full signed i16 range');
}

class GpuHarness {
  constructor(Module, config) {
    this.Module = Module;
    this.config = config;
    this.targets = config.externalTargets;
    this.targetLanes = new Map(this.targets.map((target, lane) => [target, lane]));
    const heap = Module.heap;
    const network = config.network;
    const initial = config.initial;
    const neuronCount = network.neuronCount;
    const ringSize = Math.max(config.delaySteps, 1);
    const zeroRef = new Int32Array(neuronCount).fill(config.refractorySteps);
    for (const neuron of config.zeroRefractory) zeroRef[neuron] = 0;
    const silenced = new Uint32Array(neuronCount);
    for (const neuron of config.silencedSources) silenced[neuron] = 1;
    const metadata = {
      neuronCount,
      edgeCount: network.destinations.length,
      ringSize,
      delaySteps: config.delaySteps,
      rowPtrPtr: heap.write(Uint32Array, network.rowPtr),
      destinationsPtr: heap.write(Uint32Array, network.destinations),
      signedCountsPtr: heap.write(Int16Array, network.signedCounts),
      voltagePtr: heap.write(Float32Array, initial.voltage),
      conductancePtr: heap.write(Float32Array, initial.conductance),
      refractoryRemainingPtr: heap.write(Int32Array, new Int32Array(neuronCount)),
      refractoryLengthsPtr: heap.write(Int32Array, zeroRef),
      spikesPtr: heap.write(Uint32Array, new Uint32Array(neuronCount)),
      spikeCountsPtr: heap.write(Uint32Array, new Uint32Array(neuronCount)),
      spikeRingPtr: heap.write(Uint32Array, new Uint32Array(neuronCount * ringSize)),
      arrivalsPtr: heap.write(Int32Array, new Int32Array(neuronCount)),
      silencedSourcesPtr: heap.write(Uint32Array, silenced),
      externalTargetsPtr: heap.write(Uint32Array, this.targets),
      externalTargetCount: this.targets.length,
      restingMv: config.parameters.resting_mv,
      resetMv: config.parameters.reset_mv,
      thresholdMv: config.parameters.threshold_mv,
      membraneDecay: Math.exp(-config.parameters.dt_ms / config.parameters.membrane_tau_ms),
      synapseDecay: Math.exp(-config.parameters.dt_ms / config.parameters.synapse_tau_ms),
      coupling: coupling(config.parameters),
      synapseWeightMv: config.parameters.synapse_weight_mv,
      externalWeightMv: config.parameters.external_weight_mv,
    };
    this.handle = Module.gpuCreate(metadata);
    if (this.handle < 0) throw new Error(Module.gpuLastError || "WebGPU neural engine creation failed");
  }

  async run(schedule, probes) {
    const offsets = new Uint32Array(schedule.length + 1);
    const lanes = [];
    const counts = [];
    for (let step = 0; step < schedule.length; step += 1) {
      const events = [...schedule[step]].sort((left, right) => this.targetLanes.get(left.neuron) - this.targetLanes.get(right.neuron));
      for (const event of events) {
        const lane = this.targetLanes.get(event.neuron);
        if (lane === undefined) throw new Error(`event target ${event.neuron} has no sparse lane`);
        lanes.push(lane);
        counts.push(event.count);
      }
      offsets[step + 1] = lanes.length;
    }
    const heap = this.Module.heap;
    const offsetsPtr = heap.write(Uint32Array, offsets);
    const lanesPtr = heap.write(Uint32Array, new Uint32Array(lanes));
    const countsPtr = heap.write(Uint8Array, new Uint8Array(counts));
    const probesPtr = heap.write(Uint32Array, new Uint32Array(probes));
    const resultPtr = heap.allocate((5 + probes.length) * Uint32Array.BYTES_PER_ELEMENT);
    const status = await this.Module.gpuWindow(this.handle, schedule.length, offsetsPtr, lanesPtr, countsPtr, lanes.length, probesPtr, probes.length, resultPtr);
    if (status !== 0) throw new Error(this.Module.gpuLastError || `WebGPU window returned ${status}`);
    const result = heap.view(Uint32Array, resultPtr, 5 + probes.length);
    if (result[4] !== probes.length) throw new Error("WebGPU returned malformed probe telemetry");
    this.latestTotal = result[0] + result[1] * 0x100000000;
    return {
      total: this.latestTotal,
      active: result[2],
      meanVoltage: bitsFloat(result[3]),
      probeDeltas: new Uint32Array(result.slice(5)),
    };
  }

  async state() {
    return decodeState(await this.Module.gpuDebugState(this.handle), this.config.network.neuronCount);
  }

  totalSpikes() {
    return this.latestTotal ?? 0;
  }

  destroy() {
    this.Module.gpuDestroy(this.handle);
  }
}

class CpuReference {
  constructor(config) {
    this.config = config;
    this.voltage = [...config.initial.voltage];
    this.conductance = [...config.initial.conductance];
    this.refractory = new Int32Array(config.network.neuronCount);
    this.refractoryLengths = new Int32Array(config.network.neuronCount).fill(config.refractorySteps);
    for (const neuron of config.zeroRefractory) this.refractoryLengths[neuron] = 0;
    this.silenced = new Uint8Array(config.network.neuronCount);
    for (const neuron of config.silencedSources) this.silenced[neuron] = 1;
    this.ringSlots = config.delaySteps + 1;
    this.ring = new Float64Array(this.ringSlots * config.network.neuronCount);
    this.counts = new Uint32Array(config.network.neuronCount);
    this.stepIndex = 0;
  }

  step(events) {
    const { network, parameters } = this.config;
    const neurons = network.neuronCount;
    const spikes = new Uint32Array(neurons);
    const membraneDecay = Math.exp(-parameters.dt_ms / parameters.membrane_tau_ms);
    const synapseDecay = Math.exp(-parameters.dt_ms / parameters.synapse_tau_ms);
    const factor = coupling(parameters);
    for (let neuron = 0; neuron < neurons; neuron += 1) {
      if (this.refractory[neuron] !== 0) continue;
      const oldConductance = this.conductance[neuron];
      this.conductance[neuron] = oldConductance * synapseDecay;
      this.voltage[neuron] = parameters.resting_mv
        + (this.voltage[neuron] - parameters.resting_mv) * membraneDecay
        + oldConductance * factor;
      if (this.voltage[neuron] > parameters.threshold_mv) {
        spikes[neuron] = 1;
        this.counts[neuron] += 1;
      }
    }
    const targetOffset = ((this.stepIndex + this.config.delaySteps) % this.ringSlots) * neurons;
    for (let source = 0; source < neurons; source += 1) {
      if (!spikes[source] || this.silenced[source]) continue;
      for (let edge = network.rowPtr[source]; edge < network.rowPtr[source + 1]; edge += 1) {
        this.ring[targetOffset + network.destinations[edge]] += network.signedCounts[edge] * parameters.synapse_weight_mv;
      }
    }
    const currentOffset = (this.stepIndex % this.ringSlots) * neurons;
    for (let neuron = 0; neuron < neurons; neuron += 1) {
      this.conductance[neuron] += this.ring[currentOffset + neuron];
      this.ring[currentOffset + neuron] = 0;
    }
    for (const event of events) this.voltage[event.neuron] += event.count * parameters.external_weight_mv;
    for (let neuron = 0; neuron < neurons; neuron += 1) {
      if (spikes[neuron]) {
        this.voltage[neuron] = parameters.reset_mv;
        this.conductance[neuron] = 0;
        this.refractory[neuron] = Math.max(this.refractoryLengths[neuron] - 1, 0);
      } else if (this.refractory[neuron] > 0) {
        this.refractory[neuron] -= 1;
      }
    }
    this.stepIndex += 1;
    return { spikes, voltage: [...this.voltage], conductance: [...this.conductance], counts: new Uint32Array(this.counts) };
  }
}

function makeMockModule() {
  const buffer = new ArrayBuffer(32 * MiB);
  const module = {
    HEAPU8: new Uint8Array(buffer),
    printErr: (message) => console.error(message),
    heap: new Heap(buffer),
  };
  return module;
}

class Heap {
  constructor(buffer) {
    this.buffer = buffer;
    this.cursor = 1024;
  }

  allocate(bytes) {
    const pointer = (this.cursor + 7) & ~7;
    this.cursor = pointer + bytes;
    if (this.cursor > this.buffer.byteLength) throw new Error("mock WASM heap exhausted");
    return pointer;
  }

  write(Type, values) {
    const pointer = this.allocate(Math.max(values.length * Type.BYTES_PER_ELEMENT, 1));
    new Type(this.buffer, pointer, values.length).set(values);
    return pointer;
  }

  view(Type, pointer, length) {
    return new Type(this.buffer, pointer, length);
  }
}

function fixtureConfig(fixture) {
  const targets = [...new Set(fixture.stimulus.external_counts.map((event) => event.neuron))].sort((left, right) => left - right);
  return {
    network: {
      neuronCount: fixture.network.neuron_count,
      rowPtr: fixture.network.row_ptr_u32,
      destinations: fixture.network.destinations_u32,
      signedCounts: fixture.network.signed_counts_i16,
    },
    parameters: fixture.parameters,
    initial: { voltage: fixture.initial_state.v_mv, conductance: fixture.initial_state.g_mv },
    delaySteps: Math.round(fixture.parameters.delay_ms / fixture.parameters.dt_ms),
    refractorySteps: Math.round(fixture.parameters.refractory_ms / fixture.parameters.dt_ms),
    zeroRefractory: fixture.overrides.zero_refractory,
    silencedSources: fixture.overrides.silenced_sources,
    externalTargets: targets,
  };
}

function testConfig(parameters, rowPtr, destinations, signedCounts, targets, voltage, conductance, delaySteps, silencedSources) {
  return {
    network: { neuronCount: rowPtr.length - 1, rowPtr, destinations, signedCounts },
    parameters: { ...parameters, delay_ms: delaySteps * parameters.dt_ms },
    initial: { voltage, conductance },
    delaySteps,
    refractorySteps: Math.round(parameters.refractory_ms / parameters.dt_ms),
    zeroRefractory: targets,
    silencedSources,
    externalTargets: targets,
  };
}

function decodeState(words, neuronCount) {
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const integers = new Int32Array(words.buffer, words.byteOffset, words.length);
  const voltage = [];
  const conductance = [];
  const spikes = new Uint32Array(neuronCount);
  const counts = new Uint32Array(neuronCount);
  for (let neuron = 0; neuron < neuronCount; neuron += 1) {
    const base = neuron * stateWords;
    voltage.push(floats[base]);
    conductance.push(floats[base + 1]);
    spikes[neuron] = words[base + 4];
    counts[neuron] = words[base + 5];
    if (integers[base + 2] < 0) throw new Error("GPU refractory state became negative");
  }
  return { voltage, conductance, spikes, counts };
}

function compareState(actual, expected, name) {
  expectEqual(actual.spikes, expected.spikes, `${name} spikes`);
  expectEqual(actual.counts, expected.counts, `${name} spike counts`);
  for (let neuron = 0; neuron < actual.voltage.length; neuron += 1) {
    expectNear(actual.voltage[neuron], expected.voltage[neuron], stateTolerance, `${name} voltage neuron ${neuron}`);
    expectNear(actual.conductance[neuron], expected.conductance[neuron], stateTolerance, `${name} conductance neuron ${neuron}`);
  }
}

function coupling(parameters) {
  const membrane = Math.exp(-parameters.dt_ms / parameters.membrane_tau_ms);
  const synapse = Math.exp(-parameters.dt_ms / parameters.synapse_tau_ms);
  return parameters.membrane_tau_ms === parameters.synapse_tau_ms
    ? parameters.dt_ms / parameters.membrane_tau_ms * membrane
    : parameters.synapse_tau_ms / (parameters.membrane_tau_ms - parameters.synapse_tau_ms) * (membrane - synapse);
}

function eventsAt(events, tick) {
  return events.filter((event) => event.tick === tick).map(({ neuron, count }) => ({ neuron, count }));
}

function allNeurons(count) {
  return Array.from({ length: count }, (_, index) => index);
}

function expect(condition, name) {
  if (!condition) throw new Error(name);
}

function expectNear(actual, expected, tolerance, name) {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${name}: expected ${expected}, got ${actual}, tolerance ${tolerance}`);
  }
}

function expectNumber(actual, expected, name) {
  if (actual !== expected) throw new Error(`${name}: expected ${expected}, got ${actual}`);
}

function expectEqual(actual, expected, name) {
  const left = Array.from(actual);
  const right = Array.from(expected);
  if (left.length !== right.length || left.some((value, index) => value !== right[index])) {
    throw new Error(`${name}: expected ${JSON.stringify(right)}, got ${JSON.stringify(left)}`);
  }
}

function bitsFloat(bits) {
  const bytes = new ArrayBuffer(4);
  new Uint32Array(bytes)[0] = bits;
  return new Float32Array(bytes)[0];
}

function record(report, name) {
  report.checks.push({ name, status: "PASS" });
}

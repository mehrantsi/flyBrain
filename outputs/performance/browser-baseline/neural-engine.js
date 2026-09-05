const STATE_WORDS = 8;
const STATE_BYTES = STATE_WORDS * Uint32Array.BYTES_PER_ELEMENT;
const TICK_BYTES = 64;
const TICK_STRIDE = 256;

export async function installNeuralEngine(Module, options = {}) {
  if (!navigator.gpu) {
    throw new Error("WebGPU is required for the browser neural engine; no CPU fallback is available");
  }
  const sourceUrl = options.shaderUrl ?? new URL("./neural.wgsl", import.meta.url);
  const source = await (await fetch(sourceUrl)).text();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new Error("WebGPU adapter unavailable for the browser neural engine");
  }
  const device = await adapter.requestDevice();
  const shader = device.createShaderModule({ code: source });
  const diagnostics = await shader.getCompilationInfo();
  const errors = diagnostics.messages.filter((message) => message.type === "error");
  if (errors.length) {
    throw new Error(`WebGPU neural shader failed to compile: ${errors.map((message) => message.message).join("; ")}`);
  }
  const backend = new NeuralGpuBackend(Module, device, shader);
  device.addEventListener("uncapturederror", (event) => backend.recordFatal(`WebGPU uncaptured error: ${gpuErrorText(event.error)}`));
  void device.lost.then(
    (info) => backend.recordFatal(`WebGPU device lost: ${info.message || info.reason || "unknown reason"}`),
    (error) => backend.recordFatal(`WebGPU device-loss observation failed: ${gpuErrorText(error)}`),
  );
  Module.gpuCreate = (metadata) => backend.create(metadata);
  Module.gpuWindow = (...args) => backend.window(...args);
  Module.gpuDestroy = (handle) => backend.destroy(handle);
  Module.gpuDebugState = (handle) => backend.debugState(handle);
  return backend;
}

class NeuralGpuBackend {
  constructor(Module, device, shader) {
    this.Module = Module;
    this.device = device;
    this.nextHandle = 1;
    this.engines = new Map();
    this.fatalError = null;
    this.mainLayout = device.createBindGroupLayout({
      entries: [
        storageBinding(0, "storage"),
        storageBinding(1, "storage"),
        storageBinding(2, "read-only-storage"),
        storageBinding(3, "read-only-storage"),
        storageBinding(4, "read-only-storage"),
        storageBinding(5, "storage"),
        storageBinding(6, "read-only-storage"),
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
      ],
    });
    this.summaryLayout = device.createBindGroupLayout({
      entries: [
        storageBinding(0, "read-only-storage"),
        storageBinding(1, "read-only-storage"),
        storageBinding(2, "storage"),
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    this.emptyLayout = device.createBindGroupLayout({ entries: [] });
    this.emptyBind = device.createBindGroup({ layout: this.emptyLayout, entries: [] });
    const mainPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.mainLayout] });
    const summaryPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.emptyLayout, this.summaryLayout] });
    this.pipelines = {
      initialDecay: device.createComputePipeline({ layout: mainPipelineLayout, compute: { module: shader, entryPoint: "initial_decay" } }),
      resetDecay: device.createComputePipeline({ layout: mainPipelineLayout, compute: { module: shader, entryPoint: "reset_decay" } }),
      propagate: device.createComputePipeline({ layout: mainPipelineLayout, compute: { module: shader, entryPoint: "propagate" } }),
      external: device.createComputePipeline({ layout: mainPipelineLayout, compute: { module: shader, entryPoint: "apply_external_sparse" } }),
      resetStore: device.createComputePipeline({ layout: mainPipelineLayout, compute: { module: shader, entryPoint: "reset_store" } }),
      probes: device.createComputePipeline({ layout: summaryPipelineLayout, compute: { module: shader, entryPoint: "gather_probes" } }),
      reduce: device.createComputePipeline({ layout: summaryPipelineLayout, compute: { module: shader, entryPoint: "reduce_telemetry" } }),
    };
  }

  create(metadata) {
    this.assertHealthy();
    const engine = new NeuralGpuEngine(this.Module, this.device, this.mainLayout, this.summaryLayout, this.pipelines, metadata);
    const handle = this.nextHandle++;
    this.engines.set(handle, engine);
    this.assertHealthy();
    return handle;
  }

  async window(handle, steps, offsetsPtr, lanesPtr, countsPtr, eventLen, probesPtr, probeLen, resultPtr) {
    this.assertHealthy();
    const engine = this.engine(handle);
    const offsets = copyHeap(this.Module, Uint32Array, offsetsPtr, steps + 1);
    const lanes = copyHeap(this.Module, Uint32Array, lanesPtr, eventLen);
    const counts = copyHeap(this.Module, Uint8Array, countsPtr, eventLen);
    const probes = copyHeap(this.Module, Uint32Array, probesPtr, probeLen);
    const result = await engine.run(steps, offsets, lanes, counts, probes);
    this.assertHealthy();
    const output = heapView(this.Module, Uint32Array, resultPtr, 5 + probeLen);
    output[0] = result.total & 0xffffffff;
    output[1] = Math.floor(result.total / 0x100000000);
    output[2] = result.active;
    output[3] = floatBits(result.meanVoltage);
    output[4] = probeLen;
    output.set(result.probeDeltas, 5);
    return 0;
  }

  destroy(handle) {
    const engine = this.engines.get(handle);
    if (engine) {
      engine.destroy();
      this.engines.delete(handle);
    }
  }

  async debugState(handle) {
    this.assertHealthy();
    const state = await this.engine(handle).debugState();
    this.assertHealthy();
    return state;
  }

  engine(handle) {
    const engine = this.engines.get(handle);
    if (!engine) {
      throw new Error(`unknown browser neural engine handle ${handle}`);
    }
    return engine;
  }

  recordFatal(message) {
    this.fatalError ??= message;
    this.Module.gpuLastError = this.fatalError;
    try {
      this.Module.printErr?.(this.fatalError);
    } catch {}
  }

  assertHealthy() {
    if (this.fatalError) throw new Error(this.fatalError);
  }
}

class NeuralGpuEngine {
  constructor(Module, device, mainLayout, summaryLayout, pipelines, metadata) {
    this.Module = Module;
    this.device = device;
    this.mainLayout = mainLayout;
    this.summaryLayout = summaryLayout;
    this.pipelines = pipelines;
    this.metadata = metadata;
    this.neuronCount = metadata.neuronCount;
    this.ringSize = metadata.ringSize;
    this.stepIndex = 0;
    this.probeCounts = new Map();
    if (!this.neuronCount || !this.ringSize) {
      throw new Error("browser neural metadata has an invalid neuron or delay-ring size");
    }
    this.groups = Math.ceil(this.neuronCount / 256);
    this.state = this.makeStateBuffer();
    this.ring = this.makeBuffer(copyHeap(Module, Uint32Array, metadata.spikeRingPtr, this.neuronCount * this.ringSize), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "spike ring");
    this.row = this.makeBuffer(copyHeap(Module, Uint32Array, metadata.rowPtrPtr, this.neuronCount + 1), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "CSR row pointer");
    this.destinations = this.makeBuffer(copyHeap(Module, Uint32Array, metadata.destinationsPtr, metadata.edgeCount), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "CSR destinations");
    const signedCounts = copyHeap(Module, Int16Array, metadata.signedCountsPtr, metadata.edgeCount);
    const weights = new Int32Array(signedCounts.length);
    for (let index = 0; index < weights.length; index += 1) weights[index] = signedCounts[index];
    this.weights = this.makeBuffer(weights, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "CSR weights");
    this.arrivals = this.makeBuffer(new Int32Array(this.neuronCount), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "synaptic arrivals");
    this.targets = copyHeap(Module, Uint32Array, metadata.externalTargetsPtr, metadata.externalTargetCount);
    this.events = null;
    this.eventCapacity = 0;
    this.uniform = null;
    this.uniformCapacity = 0;
    this.mainBind = null;
    this.summaryUniform = this.makeBuffer(new Uint32Array([this.neuronCount, 0, 0, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "summary parameters");
    this.probeBuffer = null;
    this.probeCapacity = 0;
    this.summary = null;
    this.summaryReadback = null;
    this.summaryCapacity = 0;
    this.summaryReadbackCapacity = 0;
    this.summaryBind = null;
    this.ensureEvents(this.targets.length);
    this.ensureUniforms(1);
  }

  async run(steps, offsets, lanes, counts, probes) {
    this.validateWindow(steps, offsets, lanes, counts, probes);
    this.ensureEvents(this.targets.length + lanes.length);
    this.ensureUniforms(steps);
    this.ensureSummary(probes);
    const needsBefore = probes.some((probe) => !this.probeCounts.has(probe));
    const packedEvents = new Uint32Array(this.targets.length + lanes.length);
    packedEvents.set(this.targets);
    for (let index = 0; index < lanes.length; index += 1) {
      packedEvents[this.targets.length + index] = lanes[index] | (counts[index] << 24);
    }
    if (packedEvents.byteLength) this.device.queue.writeBuffer(this.events, 0, packedEvents);
    const uniforms = new ArrayBuffer(steps * TICK_STRIDE);
    const view = new DataView(uniforms);
    for (let step = 0; step < steps; step += 1) {
      const offset = step * TICK_STRIDE;
      const ringSlot = (this.stepIndex + step) % this.ringSize;
      const eventOffset = offsets[step];
      const eventCount = offsets[step + 1] - eventOffset;
      view.setUint32(offset, this.neuronCount, true);
      view.setUint32(offset + 4, this.ringSize, true);
      view.setUint32(offset + 8, ringSlot, true);
      view.setUint32(offset + 12, eventOffset, true);
      view.setUint32(offset + 16, eventCount, true);
      view.setUint32(offset + 20, this.metadata.delaySteps, true);
      view.setUint32(offset + 24, this.targets.length, true);
      view.setFloat32(offset + 32, this.metadata.restingMv, true);
      view.setFloat32(offset + 36, this.metadata.resetMv, true);
      view.setFloat32(offset + 40, this.metadata.thresholdMv, true);
      view.setFloat32(offset + 44, this.metadata.membraneDecay, true);
      view.setFloat32(offset + 48, this.metadata.synapseDecay, true);
      view.setFloat32(offset + 52, this.metadata.coupling, true);
      view.setFloat32(offset + 56, this.metadata.synapseWeightMv, true);
      view.setFloat32(offset + 60, this.metadata.externalWeightMv, true);
    }
    this.device.queue.writeBuffer(this.uniform, 0, uniforms);
    if (probes.byteLength) this.device.queue.writeBuffer(this.probeBuffer, 0, probes);
    this.device.queue.writeBuffer(this.summaryUniform, 0, new Uint32Array([this.neuronCount, probes.length, 0, 0]));
    const encoder = this.device.createCommandEncoder();
    const telemetryBytes = this.summaryByteLength(probes.length);
    if (needsBefore && probes.length) {
      const beforePass = encoder.beginComputePass();
      beforePass.setPipeline(this.pipelines.probes);
      beforePass.setBindGroup(0, this.emptyBind);
      beforePass.setBindGroup(1, this.summaryBind);
      beforePass.dispatchWorkgroups(Math.ceil(probes.length / 256));
      beforePass.end();
      encoder.copyBufferToBuffer(this.summary, 0, this.summaryReadback, telemetryBytes, probes.length * Uint32Array.BYTES_PER_ELEMENT);
    }
    const pass = encoder.beginComputePass();
    for (let step = 0; step < steps; step += 1) {
      this.dispatch(pass, step === 0 ? this.pipelines.initialDecay : this.pipelines.resetDecay, this.groups, step);
      this.dispatch(pass, this.pipelines.propagate, this.groups, step);
      const eventCount = offsets[step + 1] - offsets[step];
      if (eventCount) this.dispatch(pass, this.pipelines.external, Math.ceil(eventCount / 256), step);
      if (step + 1 === steps) this.dispatch(pass, this.pipelines.resetStore, this.groups, step);
    }
    if (probes.length) {
      pass.setPipeline(this.pipelines.probes);
      pass.setBindGroup(0, this.emptyBind);
      pass.setBindGroup(1, this.summaryBind);
      pass.dispatchWorkgroups(Math.ceil(probes.length / 256));
    }
    pass.setPipeline(this.pipelines.reduce);
    pass.setBindGroup(0, this.emptyBind);
    pass.setBindGroup(1, this.summaryBind);
    pass.dispatchWorkgroups(this.groups);
    pass.end();
    encoder.copyBufferToBuffer(this.summary, 0, this.summaryReadback, 0, telemetryBytes);
    this.device.queue.submit([encoder.finish()]);
    const readbackBytes = telemetryBytes + (needsBefore ? probes.length * Uint32Array.BYTES_PER_ELEMENT : 0);
    await this.summaryReadback.mapAsync(GPUMapMode.READ, 0, readbackBytes);
    const data = new Uint32Array(this.summaryReadback.getMappedRange(0, readbackBytes).slice(0));
    this.summaryReadback.unmap();
    this.stepIndex += steps;
    const telemetryWords = telemetryBytes / Uint32Array.BYTES_PER_ELEMENT;
    return this.decodeTelemetry(data.subarray(0, telemetryWords), probes, needsBefore ? data.subarray(telemetryWords) : null);
  }

  async debugState() {
    const size = this.neuronCount * STATE_BYTES;
    const readback = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.state, 0, readback, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();
    return data;
  }

  decodeTelemetry(telemetry, probes, beforeCounts) {
    const deltas = new Uint32Array(probes.length);
    for (let index = 0; index < probes.length; index += 1) {
      const current = telemetry[index];
      const previous = this.probeCounts.get(probes[index]) ?? beforeCounts?.[index];
      if (previous === undefined) throw new Error("browser neural probe baseline is unavailable");
      if (current < previous) throw new Error("browser neural probe spike count moved backwards");
      deltas[index] = current - previous;
      this.probeCounts.set(probes[index], current);
    }
    let low = 0;
    let high = 0;
    let active = 0;
    let voltage = 0;
    for (let group = 0; group < this.groups; group += 1) {
      const index = probes.length + group * 4;
      low += telemetry[index];
      high += telemetry[index + 1];
      active += telemetry[index + 2];
      voltage += bitsFloat(telemetry[index + 3]);
    }
    const total = low + high * 0x10000;
    return { total, active, meanVoltage: voltage / this.neuronCount, probeDeltas: deltas };
  }

  dispatch(pass, pipeline, groups, uniformSlot) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.mainBind, [uniformSlot * TICK_STRIDE]);
    pass.dispatchWorkgroups(groups);
  }

  ensureEvents(words) {
    if (this.events && words <= this.eventCapacity) return;
    this.events?.destroy();
    this.eventCapacity = nextCapacity(Math.max(words, 1));
    this.events = this.makeEmptyBuffer(this.eventCapacity * Uint32Array.BYTES_PER_ELEMENT, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "sparse events");
    this.rebuildMainBind();
  }

  ensureUniforms(steps) {
    if (steps <= this.uniformCapacity) return;
    this.uniform?.destroy();
    this.uniformCapacity = nextCapacity(steps);
    this.uniform = this.makeEmptyBuffer(this.uniformCapacity * TICK_STRIDE, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "tick parameters");
    this.rebuildMainBind();
  }

  ensureSummary(probes) {
    let bindingsChanged = false;
    if (!this.probeBuffer || probes.length > this.probeCapacity) {
      this.probeBuffer?.destroy();
      this.probeCapacity = nextCapacity(probes.length || 1);
      this.probeBuffer = this.makeEmptyBuffer(this.probeCapacity * Uint32Array.BYTES_PER_ELEMENT, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "probe indices");
      bindingsChanged = true;
    }
    const words = probes.length + this.groups * 4;
    if (words > this.summaryCapacity) {
      this.summary?.destroy();
      this.summaryCapacity = nextCapacity(words);
      const bytes = this.summaryCapacity * Uint32Array.BYTES_PER_ELEMENT;
      this.summary = this.makeEmptyBuffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "window telemetry");
      bindingsChanged = true;
    }
    const readbackWords = words + probes.length;
    if (readbackWords > this.summaryReadbackCapacity) {
      this.summaryReadback?.destroy();
      this.summaryReadbackCapacity = nextCapacity(readbackWords);
      this.summaryReadback = this.makeEmptyBuffer(this.summaryReadbackCapacity * Uint32Array.BYTES_PER_ELEMENT, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, "window telemetry readback");
    }
    if (bindingsChanged || !this.summaryBind) this.rebuildSummaryBind();
  }

  rebuildSummaryBind() {
    this.summaryBind = this.device.createBindGroup({
      layout: this.summaryLayout,
      entries: [
        { binding: 0, resource: { buffer: this.state } },
        { binding: 1, resource: { buffer: this.probeBuffer } },
        { binding: 2, resource: { buffer: this.summary } },
        { binding: 3, resource: { buffer: this.summaryUniform } },
      ],
    });
  }

  rebuildMainBind() {
    if (!this.events || !this.uniform) return;
    this.mainBind = this.device.createBindGroup({
      layout: this.mainLayout,
      entries: [
        { binding: 0, resource: { buffer: this.state } },
        { binding: 1, resource: { buffer: this.ring } },
        { binding: 2, resource: { buffer: this.row } },
        { binding: 3, resource: { buffer: this.destinations } },
        { binding: 4, resource: { buffer: this.weights } },
        { binding: 5, resource: { buffer: this.arrivals } },
        { binding: 6, resource: { buffer: this.events } },
        { binding: 7, resource: { buffer: this.uniform, size: TICK_BYTES } },
      ],
    });
  }

  makeStateBuffer() {
    const words = new Uint32Array(this.neuronCount * STATE_WORDS);
    const floats = new Float32Array(words.buffer);
    const voltage = heapView(this.Module, Float32Array, this.metadata.voltagePtr, this.neuronCount);
    const conductance = heapView(this.Module, Float32Array, this.metadata.conductancePtr, this.neuronCount);
    const refractory = heapView(this.Module, Int32Array, this.metadata.refractoryRemainingPtr, this.neuronCount);
    const refractoryLengths = heapView(this.Module, Int32Array, this.metadata.refractoryLengthsPtr, this.neuronCount);
    const spikes = heapView(this.Module, Uint32Array, this.metadata.spikesPtr, this.neuronCount);
    const spikeCounts = heapView(this.Module, Uint32Array, this.metadata.spikeCountsPtr, this.neuronCount);
    const silenced = heapView(this.Module, Uint32Array, this.metadata.silencedSourcesPtr, this.neuronCount);
    for (let neuron = 0; neuron < this.neuronCount; neuron += 1) {
      const base = neuron * STATE_WORDS;
      floats[base] = voltage[neuron];
      floats[base + 1] = conductance[neuron];
      words[base + 2] = refractory[neuron];
      words[base + 3] = refractoryLengths[neuron];
      words[base + 4] = spikes[neuron];
      words[base + 5] = spikeCounts[neuron];
      words[base + 6] = silenced[neuron];
    }
    return this.makeBuffer(words, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, "neuron state");
  }

  makeBuffer(data, usage, label) {
    const buffer = this.makeEmptyBuffer(Math.max(data.byteLength, 4), usage, label);
    if (data.byteLength) this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    return buffer;
  }

  makeEmptyBuffer(size, usage, label) {
    const limit = this.device.limits.maxStorageBufferBindingSize;
    if ((usage & GPUBufferUsage.STORAGE) && size > limit) {
      throw new Error(`${label} requires ${size} bytes, above this adapter's storage-buffer limit of ${limit}`);
    }
    if (size > this.device.limits.maxBufferSize) {
      throw new Error(`${label} requires ${size} bytes, above this adapter's maximum buffer size of ${this.device.limits.maxBufferSize}`);
    }
    return this.device.createBuffer({ size, usage, label: `flybrain ${label}` });
  }

  validateWindow(steps, offsets, lanes, counts, probes) {
    if (offsets.length !== steps + 1 || offsets[0] !== 0 || offsets[steps] !== lanes.length || lanes.length !== counts.length) {
      throw new Error("invalid sparse stimulus window dimensions");
    }
    for (let index = 0; index < lanes.length; index += 1) {
      if (lanes[index] >= this.targets.length || lanes[index] >= 0x1000000 || counts[index] === 0) throw new Error("invalid sparse stimulus lane or count");
    }
    for (let index = 0; index < steps; index += 1) {
      if (offsets[index] > offsets[index + 1]) throw new Error("sparse stimulus offsets are not monotonic");
    }
    for (const probe of probes) {
      if (probe >= this.neuronCount) throw new Error("probe neuron outside browser neural state");
    }
  }

  summaryByteLength(probeCount) {
    return (probeCount + this.groups * 4) * Uint32Array.BYTES_PER_ELEMENT;
  }

  destroy() {
    for (const buffer of [this.state, this.ring, this.row, this.destinations, this.weights, this.arrivals, this.events, this.uniform, this.summaryUniform, this.probeBuffer, this.summary, this.summaryReadback]) {
      buffer?.destroy();
    }
  }
}

function storageBinding(binding, type) {
  return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } };
}

function heapView(Module, Type, pointer, length) {
  return new Type(Module.HEAPU8.buffer, pointer, length);
}

function copyHeap(Module, Type, pointer, length) {
  return new Type(heapView(Module, Type, pointer, length));
}

function nextCapacity(required) {
  let capacity = 1;
  while (capacity < required) capacity *= 2;
  return capacity;
}

function floatBits(value) {
  const bytes = new ArrayBuffer(4);
  new Float32Array(bytes)[0] = value;
  return new Uint32Array(bytes)[0];
}

function bitsFloat(bits) {
  const bytes = new ArrayBuffer(4);
  new Uint32Array(bytes)[0] = bits;
  return new Float32Array(bytes)[0];
}

function gpuErrorText(error) {
  return String(error?.message || error || "unknown error");
}

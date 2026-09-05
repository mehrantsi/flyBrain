mergeInto(LibraryManager.library, {
  fly_gpu_create__deps: ['$Asyncify'],
  fly_gpu_create__async: true,
  fly_gpu_create: (jsonPtr, jsonLen) => Asyncify.handleAsync(async () => {
    try {
      if (typeof Module.gpuCreate !== 'function') {
        throw new Error('browser neural WebGPU backend was not installed');
      }
      const json = new TextDecoder().decode(HEAPU8.subarray(jsonPtr, jsonPtr + jsonLen));
      return await Module.gpuCreate(JSON.parse(json));
    } catch (error) {
      Module.gpuLastError = String(error?.stack || error);
      Module.printErr(Module.gpuLastError);
      return -1;
    }
  }),

  fly_gpu_window__deps: ['$Asyncify'],
  fly_gpu_window__async: true,
  fly_gpu_window: (handle, steps, offsetsPtr, lanesPtr, countsPtr, eventLen, probesPtr, probeLen, resultPtr) => Asyncify.handleAsync(async () => {
    try {
      if (typeof Module.gpuWindow !== 'function') {
        throw new Error('browser neural WebGPU backend was not installed');
      }
      return await Module.gpuWindow(handle, steps, offsetsPtr, lanesPtr, countsPtr, eventLen, probesPtr, probeLen, resultPtr);
    } catch (error) {
      Module.gpuLastError = String(error?.stack || error);
      Module.printErr(Module.gpuLastError);
      return -2;
    }
  }),

  fly_gpu_destroy: (handle) => {
    try {
      Module.gpuDestroy?.(handle);
    } catch (error) {
      Module.gpuLastError = String(error?.stack || error);
      Module.printErr(Module.gpuLastError);
    }
  },
});

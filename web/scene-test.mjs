import assert from 'node:assert/strict';
import * as THREE from 'three';
import { FlySceneRenderer, makeMeshGeometry, retinaRgba, RETINA_WIDTH, RETINA_HEIGHT, RETINA_PAIR_BYTES, RETINA_EYE_BYTES } from './scene.js';

const mesh = makeMeshGeometry({
  vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
  faces: [[0, 1, 2]],
  normals: [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
  faceNormals: [[1, 1, 1]],
});
assert.deepEqual([...mesh.getAttribute('normal').array], [0, 0, 1, 0, 0, 1, 0, 0, 1]);
assert.deepEqual([...mesh.getAttribute('position').array], [0, 0, 0, 1, 0, 0, 0, 1, 0]);
mesh.dispose();

const eyes = new Uint8Array(RETINA_PAIR_BYTES);
eyes.fill(40, 0, RETINA_EYE_BYTES);
eyes.fill(190, RETINA_EYE_BYTES);
const rgba = retinaRgba(eyes);
for (let y = 0; y < RETINA_HEIGHT; y += 1) {
  for (let x = 0; x < RETINA_WIDTH * 2; x += 1) {
    const offset = (y * RETINA_WIDTH * 2 + x) * 4;
    const value = x < RETINA_WIDTH ? 40 : 190;
    assert.deepEqual([...rgba.subarray(offset, offset + 4)], [value, value, value, 255]);
  }
}
const rig = Object.create(FlySceneRenderer.prototype);
Object.assign(rig, {
  camera: new THREE.PerspectiveCamera(45, 1, 0.05, 1400),
  controls: { target: new THREE.Vector3(), update() {} },
  tempTarget: new THREE.Vector3(), followPosition: new THREE.Vector3(), followDelta: new THREE.Vector3(),
  cameraMode: 'chase', cameraPresetPending: true, cameras: [],
  snapshot: { root_position: [0, 0, 2] },
});
rig.updateObserverCamera();
assert.ok(Math.abs(rig.camera.position.distanceTo(rig.controls.target) - 36) < 1e-10);
rig.camera.position.sub(rig.controls.target).multiplyScalar(2).add(rig.controls.target);
const zoomedPosition = rig.camera.position.clone();
rig.snapshot = { root_position: [10, 4, 5] };
rig.updateObserverCamera();
assert.ok(rig.camera.position.distanceTo(zoomedPosition.add(new THREE.Vector3(10, 4, 3))) < 1e-10);
assert.ok(Math.abs(rig.camera.position.distanceTo(rig.controls.target) - 72) < 1e-10);
rig.cameraMode = 'orbit';
const freePosition = rig.camera.position.clone();
rig.snapshot = { root_position: [100, 40, 50] };
rig.updateObserverCamera();
assert.deepEqual(rig.camera.position.toArray(), freePosition.toArray());
const capture = Object.create(FlySceneRenderer.prototype);
const fly = { visible: true, userData: { isFly: true } };
const completions = [];
let binocular;
let target;
Object.assign(capture, {
  sceneReady: true, frameReady: true, disposed: false, pendingVision: true,
  cameras: [{ name: 'left_eye', fovy: 157 }, { name: 'right_eye', fovy: 157 }],
  cameraPoses: [0, 1].map(() => ({ position: new THREE.Vector3(), quaternion: new THREE.Quaternion() })),
  eyeCameras: [new THREE.PerspectiveCamera(), new THREE.PerspectiveCamera()],
  eyeRenderTargets: [0, 1],
  eyeReadbacks: [new Uint8Array(RETINA_WIDTH * RETINA_HEIGHT * 4), new Uint8Array(RETINA_WIDTH * RETINA_HEIGHT * 4)],
  visionScratch: new Uint8Array(RETINA_PAIR_BYTES), objects: [fly],
  applyWallCutaway() {}, onVision(pixels) { binocular = new Uint8Array(pixels); },
  renderer: {
    shadowMap: { autoUpdate: true }, setRenderTarget(value) { target = value; }, clear() {}, render() {},
    readRenderTargetPixelsAsync(eye, x, y, width, height, bytes) {
      return new Promise(resolve => completions.push(() => {
        for (let row = 0; row < height; row++) {
          bytes.fill(eye * 100 + row % 100, row * width * 4, (row + 1) * width * 4);
        }
        resolve();
      }));
    },
  },
});
const reading = capture.captureVision();
assert.equal(capture.visionInFlight, true);
assert.equal(capture.pendingVision, false);
assert.equal(fly.visible, true);
assert.equal(target, null);
assert.equal(capture.renderer.shadowMap.autoUpdate, true);
capture.pendingVision = true;
completions[1]();
completions[0]();
await reading;
assert.equal(capture.visionInFlight, false);
assert.equal(capture.pendingVision, true);
assert.equal(binocular[0], (RETINA_HEIGHT - 1) % 100);
assert.equal(binocular[RETINA_EYE_BYTES], 100 + (RETINA_HEIGHT - 1) % 100);
assert.equal(binocular[RETINA_EYE_BYTES - 1], 0);
assert.equal(binocular[RETINA_PAIR_BYTES - 1], 100);
binocular = undefined;
let disposals = 0;
capture.disposeResources = () => { disposals++; };
const lastReading = capture.captureVision();
capture.dispose();
assert.equal(disposals, 0);
completions[2]();
completions[3]();
await lastReading;
assert.equal(disposals, 1);
assert.equal(binocular, undefined);
console.log('PASS: mesh normals, camera zoom, binocular row layout, asynchronous eye pairing, pending frames and in-flight disposal');

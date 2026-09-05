import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export const RETINA_WIDTH = 450;
export const RETINA_HEIGHT = 512;
export const RETINA_CHANNELS = 3;
export const RETINA_EYE_BYTES = RETINA_WIDTH * RETINA_HEIGHT * RETINA_CHANNELS;
export const RETINA_PAIR_BYTES = RETINA_EYE_BYTES * 2;
export const BODY_POSE_STRIDE = 7;
export const CAMERA_POSE_STRIDE = 12;
export const PRIMITIVE_TYPES = Object.freeze({
  plane: 0,
  sphere: 2,
  capsule: 3,
  ellipsoid: 4,
  cylinder: 5,
  box: 6,
  mesh: 7,
});

const MM_PER_UNIT = 1;
const MAX_PIXEL_RATIO = 2;
const VISION_RATE_HZ = 15;
const VISION_PERIOD_MS = 1000 / VISION_RATE_HZ;
const CAMERA_NEAR_MM = 0.05;
const CAMERA_FAR_MM = 1400;
const OAK_TEXTURE_URL = "/assets/neuromechfly/textures/oak-v1.png";
const DEFAULT_FOV_DEG = 45;
const EYE_FOV_DEG = 157;
const CHASE_DISTANCE_MM = 36;

const WALL_RULES = Object.freeze([
  { key: "left", axis: 0, sign: -1, exact: ["room_wall_left"], prefixes: ["detail_baseboard_left", "detail_wall_left_"] },
  { key: "right", axis: 0, sign: 1, exact: ["room_wall_right"], prefixes: ["window_", "detail_baseboard_right"] },
  { key: "back", axis: 1, sign: 1, exact: ["room_wall_back"], prefixes: ["detail_baseboard_back", "detail_wall_panel_"] },
  { key: "front", axis: 1, sign: -1, exact: ["room_wall_front_left", "room_wall_front_right", "room_wall_front_window"], prefixes: ["detail_baseboard_front", "detail_wall_front_"] },
  { key: "ceiling", axis: 2, sign: 1, exact: ["room_wall_ceiling"], prefixes: [] },
]);

function finiteNumber(value, fallback = 0) {
  return Number.isFinite(Number(value)) ? Number(value) : fallback;
}

function vector3(value) {
  return [
    finiteNumber(value?.[0]) * MM_PER_UNIT,
    finiteNumber(value?.[1]) * MM_PER_UNIT,
    finiteNumber(value?.[2]) * MM_PER_UNIT,
  ];
}

function size3(value) {
  return [
    Math.max(0, finiteNumber(value?.[0])),
    Math.max(0, finiteNumber(value?.[1])),
    Math.max(0, finiteNumber(value?.[2])),
  ];
}

function rgbaColor(value, fallback = [0.6, 0.4, 0.22, 1]) {
  const values = [0, 1, 2, 3].map((index) => finiteNumber(value?.[index], fallback[index]));
  const scale = values.slice(0, 3).some((channel) => channel > 1) ? 255 : 1;
  return {
    color: new THREE.Color(values[0] / scale, values[1] / scale, values[2] / scale),
    opacity: Math.min(1, Math.max(0, values[3] / (values[3] > 1 ? 255 : 1))),
  };
}

function normalizeQuaternion(value) {
  const quaternion = new THREE.Quaternion(
    finiteNumber(value?.[1]),
    finiteNumber(value?.[2]),
    finiteNumber(value?.[3]),
    finiteNumber(value?.[0], 1),
  );
  if (quaternion.lengthSq() < 1e-12) quaternion.identity();
  return quaternion.normalize();
}

function quaternionFromRotationRows(rows) {
  const matrix = new THREE.Matrix4().set(
    finiteNumber(rows?.[0], 1), finiteNumber(rows?.[1]), finiteNumber(rows?.[2]), 0,
    finiteNumber(rows?.[3]), finiteNumber(rows?.[4], 1), finiteNumber(rows?.[5]), 0,
    finiteNumber(rows?.[6]), finiteNumber(rows?.[7]), finiteNumber(rows?.[8], 1), 0,
    0, 0, 0, 1,
  );
  return new THREE.Quaternion().setFromRotationMatrix(matrix).normalize();
}

function primitiveKey(type, size) {
  return `${type}:${size.map((value) => value.toPrecision(8)).join(",")}`;
}

function normalizeFaces(faces) {
  if (!Array.isArray(faces)) return [];
  if (faces.length === 0) return [];
  if (Array.isArray(faces[0])) return faces;
  const result = [];
  for (let index = 0; index + 2 < faces.length; index += 3) {
    result.push([faces[index], faces[index + 1], faces[index + 2]]);
  }
  return result;
}

export function makeMeshGeometry(mesh) {
  const vertices = Array.isArray(mesh?.vertices) ? mesh.vertices : [];
  const faces = normalizeFaces(mesh?.faces);
  const positions = [];
  const normals = [];
  for (let faceIndex = 0; faceIndex < faces.length; faceIndex += 1) {
    for (let corner = 0; corner < 3; corner += 1) {
      const vertex = vertices[faces[faceIndex][corner]];
      const normal = mesh.normals[mesh.faceNormals[faceIndex][corner]];
      if (!vertex || !normal) throw new Error("Invalid MuJoCo mesh vertex/normal index");
      positions.push(...vertex);
      normals.push(...normal);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
  geometry.computeBoundingSphere();
  return geometry;
}

function matchesWallRule(name, rule) {
  return rule.exact.includes(name) || rule.prefixes.some((prefix) => name.startsWith(prefix));
}

function wallRuleFor(name) {
  return WALL_RULES.find((rule) => matchesWallRule(name, rule)) ?? null;
}

function isFlyName(name) {
  return typeof name === "string" && name.startsWith("fly/");
}

function materialKind(name, materialName) {
  const value = `${name ?? ""} ${materialName ?? ""}`.toLowerCase();
  if (value.includes("wing")) return "wing";
  if (value.includes("eye")) return "eye";
  if (value.includes("oak") || value.includes("wood") || value.includes("floor") || value.includes("grid")) return "oak";
  if (value.includes("food") || value.includes("banana") || value.includes("sugar")) return "food";
  if (value.includes("band")) return "band";
  if (value.includes("bristle") || value.includes("chitin") || isFlyName(name)) return "chitin";
  return "default";
}

function colorForKind(kind, descriptorColor) {
  if (kind === "eye") return rgbaColor([0.43, 0.07, 0.045, 1]);
  if (kind === "band") return rgbaColor([0.16, 0.055, 0.025, 1]);
  if (kind === "wing") return rgbaColor([0.53, 0.56, 0.52, 0.31]);
  if (kind === "chitin") return rgbaColor(descriptorColor, [0.59, 0.31, 0.12, 1]);
  if (kind === "oak") return rgbaColor(descriptorColor, [0.40, 0.20, 0.09, 1]);
  if (kind === "food") return rgbaColor(descriptorColor, [0.72, 0.42, 0.06, 1]);
  return rgbaColor(descriptorColor);
}

function disposeMaterial(material) {
  if (!material) return;
  material.map?.dispose();
  material.dispose();
}

export function retinaRgba(pixels) {
  const bytes = pixels instanceof Uint8Array ? pixels : new Uint8Array(pixels);
  if (bytes.length !== RETINA_PAIR_BYTES) throw new Error("Expected two complete retina images");
  const rgba = new Uint8ClampedArray(RETINA_WIDTH * 2 * RETINA_HEIGHT * 4);
  for (let y = 0; y < RETINA_HEIGHT; y += 1) {
    for (let x = 0; x < RETINA_WIDTH * 2; x += 1) {
      const eye = x >= RETINA_WIDTH ? 1 : 0;
      const source = eye * RETINA_EYE_BYTES + (y * RETINA_WIDTH + x % RETINA_WIDTH) * 3;
      const target = (y * RETINA_WIDTH * 2 + x) * 4;
      rgba[target] = bytes[source];
      rgba[target + 1] = bytes[source + 1];
      rgba[target + 2] = bytes[source + 2];
      rgba[target + 3] = 255;
    }
  }
  return rgba;
}

export function drawRetinaPixels(canvas, pixels) {
  if (!canvas || !pixels) return;
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) return;
  canvas.width = RETINA_WIDTH * 2;
  canvas.height = RETINA_HEIGHT;
  context.putImageData(new ImageData(retinaRgba(pixels), canvas.width, canvas.height), 0, 0);
}

export class FlySceneRenderer {
  constructor(canvas, { onVision = null, onError = console.error } = {}) {
    if (!canvas) throw new Error("scene canvas is required");
    this.canvas = canvas;
    this.onVision = onVision;
    this.onError = onError;
    this.visionInFlight = false;
    this.disposed = false;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.toneMappingExposure = 1;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.setClearColor(0x0d121a, 1);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0d121a);
    this.camera = new THREE.PerspectiveCamera(DEFAULT_FOV_DEG, 1, CAMERA_NEAR_MM, CAMERA_FAR_MM);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 1000;
    this.controls.enabled = true;
    this.controls.zoomSpeed = 0.65;
    this.needsRender = true;
    this.controls.addEventListener('change', () => { this.needsRender = true; });

    this.worldRoot = new THREE.Group();
    this.scene.add(this.worldRoot);
    this.bodyGroups = [];
    this.objects = [];
    this.meshGeometries = [];
    this.primitiveGeometries = new Map();
    this.materials = [];
    this.materialCache = new Map();
    this.cameras = [];
    this.cameraPoses = [];
    this.cameraMode = "chase";
    this.cameraPresetPending = true;
    this.followPosition = new THREE.Vector3();
    this.followDelta = new THREE.Vector3();
    this.sceneReady = false;
    this.frameReady = false;
    this.pendingVision = false;
    this.lastVisionAt = -Infinity;
    this.visionScratch = new Uint8Array(RETINA_PAIR_BYTES);
    this.eyeRenderTargets = [
      this.makeEyeRenderTarget(),
      this.makeEyeRenderTarget(),
    ];
    this.eyeReadbacks = [
      new Uint8Array(RETINA_WIDTH * RETINA_HEIGHT * 4),
      new Uint8Array(RETINA_WIDTH * RETINA_HEIGHT * 4),
    ];
    this.eyeCameras = [
      this.makeEyeCamera(),
      this.makeEyeCamera(),
    ];
    this.tempPosition = new THREE.Vector3();
    this.tempTarget = new THREE.Vector3();
    this.tempQuaternion = new THREE.Quaternion();
    this.wallPlanes = new Map();
    this.snapshot = null;
    this.foodObjects = [];
    this.addLights();
    this.resize();
  }

  addLights() {
    const hemisphere = new THREE.HemisphereLight(0xd9e7f3, 0x4e2d1d, 1.1);
    hemisphere.position.set(0, 0, 1);
    this.scene.add(hemisphere);
    const key = new THREE.DirectionalLight(0xffe8c7, 2.0);
    key.position.set(-140, -100, 205);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 0.05;
    key.shadow.camera.far = 1800;
    key.shadow.camera.left = -500;
    key.shadow.camera.right = 500;
    key.shadow.camera.top = 500;
    key.shadow.camera.bottom = -500;
    key.shadow.bias = -0.0002;
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x8ca9d1, 0.35);
    fill.position.set(-240, 180, 180);
    this.scene.add(fill);
  }

  makeEyeRenderTarget() {
    return new THREE.WebGLRenderTarget(RETINA_WIDTH, RETINA_HEIGHT, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
      stencilBuffer: false,
      generateMipmaps: false,
    });
  }

  makeEyeCamera() {
    const camera = new THREE.PerspectiveCamera(EYE_FOV_DEG, RETINA_WIDTH / RETINA_HEIGHT, CAMERA_NEAR_MM, CAMERA_FAR_MM);
    camera.up.set(0, 0, 1);
    return camera;
  }

  setScene(descriptor) {
    if (!descriptor || !Array.isArray(descriptor.geoms) || !Array.isArray(descriptor.meshes) || !Array.isArray(descriptor.cameras)) {
      throw new Error("scene descriptor must contain geoms, meshes, and cameras arrays");
    }
    this.disposeSceneObjects();
    this.bodyGroups = Array.from({ length: Math.max(0, Number(descriptor?.bodyCount) || 0) }, () => new THREE.Group());
    for (const body of this.bodyGroups) this.worldRoot.add(body);
    this.cameras = Array.isArray(descriptor?.cameras) ? descriptor.cameras : [];
    this.cameraPoses = this.cameras.map(() => ({ position: new THREE.Vector3(), quaternion: new THREE.Quaternion() }));
    this.meshGeometries = (Array.isArray(descriptor?.meshes) ? descriptor.meshes : []).map(makeMeshGeometry);
    this.wallPlanes.clear();
    this.foodObjects = [];

    const geoms = Array.isArray(descriptor?.geoms) ? descriptor.geoms : [];
    for (const geomDescriptor of geoms) {
      const object = this.makeGeom(geomDescriptor);
      if (!object) continue;
      const bodyIndex = Number.isInteger(geomDescriptor?.body) ? geomDescriptor.body : Number(geomDescriptor?.body);
      const parent = this.bodyGroups[bodyIndex] ?? this.worldRoot;
      parent.add(object);
      this.objects.push(object);
      const name = String(geomDescriptor?.name ?? "");
      if (name === "food_patch") this.foodObjects.push(object);
      const rule = wallRuleFor(name);
      if (rule) object.userData.wallRule = rule;
      if (rule && rule.exact.includes(name)) this.wallPlanes.set(rule.key, this.wallPlane(object, rule));
    }

    for (const object of this.objects) {
      const rule = object.userData.wallRule;
      if (rule && !this.wallPlanes.has(rule.key)) this.wallPlanes.set(rule.key, this.wallPlane(object, rule));
    }
    this.sceneReady = true;
    this.needsRender = true;
    this.frameReady = false;
    this.pendingVision = false;
    this.snapshot = null;
    this.setCameraMode(this.cameraMode);
  }

  disposeSceneObjects() {
    const geometries = new Set();
    for (const object of this.objects) {
      object.parent?.remove(object);
      if (object.geometry && !geometries.has(object.geometry)) {
        geometries.add(object.geometry);
        object.geometry.dispose();
      }
    }
    for (const material of this.materials) disposeMaterial(material);
    for (const body of this.bodyGroups) this.worldRoot.remove(body);
    this.objects = [];
    this.meshGeometries = [];
    this.primitiveGeometries.clear();
    this.materials = [];
    this.materialCache.clear();
  }

  wallPlane(object, rule) {
    object.updateWorldMatrix(true, false);
    object.getWorldPosition(this.tempPosition);
    const size = object.userData.geomSize ?? [0, 0, 0];
    return this.tempPosition.getComponent(rule.axis) * rule.sign - size[rule.axis];
  }

  makeGeom(descriptor) {
    const type = Number(descriptor?.type);
    const size = size3(descriptor?.size);
    const geometry = type === PRIMITIVE_TYPES.mesh
      ? this.meshGeometries[Number(descriptor?.mesh)]
      : this.makePrimitiveGeometry(type, size);
    if (!geometry) return null;
    const name = String(descriptor?.name ?? "geom");
    const kind = materialKind(name, descriptor?.material);
    const object = new THREE.Mesh(geometry, this.makeMaterial(kind, descriptor));
    object.name = name;
    object.position.fromArray(vector3(descriptor?.pos));
    object.quaternion.copy(normalizeQuaternion(descriptor?.quat));
    object.userData.baseVisible = Number(descriptor?.group ?? 0) < 3;
    object.userData.geomSize = size;
    object.userData.isFly = isFlyName(name);
    object.userData.isWing = kind === "wing";
    object.userData.isWall = Boolean(wallRuleFor(name));
    object.userData.kind = kind;
    object.visible = object.userData.baseVisible;
    object.castShadow = object.userData.baseVisible && object.material.opacity >= 0.999;
    object.receiveShadow = object.userData.baseVisible && kind !== "wing";
    if (kind === "oak") object.receiveShadow = true;
    return object;
  }

  makePrimitiveGeometry(type, size) {
    const key = primitiveKey(type, size);
    const cached = this.primitiveGeometries.get(key);
    if (cached) return cached;
    let geometry;
    if (type === PRIMITIVE_TYPES.plane) {
      geometry = new THREE.PlaneGeometry(Math.max(size[0] * 2, 1), Math.max(size[1] * 2, 1));
    } else if (type === PRIMITIVE_TYPES.sphere) {
      geometry = new THREE.SphereGeometry(Math.max(size[0], 0.001), 24, 16);
    } else if (type === PRIMITIVE_TYPES.capsule) {
      geometry = new THREE.CapsuleGeometry(Math.max(size[0], 0.001), Math.max(size[1] * 2, 0), 8, 16);
      geometry.rotateX(Math.PI / 2);
    } else if (type === PRIMITIVE_TYPES.ellipsoid) {
      geometry = new THREE.SphereGeometry(1, 24, 16);
      geometry.scale(Math.max(size[0], 0.001), Math.max(size[1], 0.001), Math.max(size[2], 0.001));
    } else if (type === PRIMITIVE_TYPES.cylinder) {
      geometry = new THREE.CylinderGeometry(Math.max(size[0], 0.001), Math.max(size[0], 0.001), Math.max(size[1] * 2, 0.001), 20);
      geometry.rotateX(Math.PI / 2);
    } else if (type === PRIMITIVE_TYPES.box) {
      geometry = new THREE.BoxGeometry(Math.max(size[0] * 2, 0.001), Math.max(size[1] * 2, 0.001), Math.max(size[2] * 2, 0.001));
    } else {
      return null;
    }
    geometry.computeBoundingSphere();
    this.primitiveGeometries.set(key, geometry);
    return geometry;
  }

  makeMaterial(kind, descriptor) {
    const key = JSON.stringify([kind, descriptor?.rgba, descriptor?.texrepeat]);
    const cached = this.materialCache.get(key);
    if (cached) return cached;
    const { color, opacity } = colorForKind(kind, descriptor?.rgba);
    const transparent = opacity < 0.999;
    const options = {
      color,
      opacity,
      transparent,
      side: kind === "wing" || kind === "oak" ? THREE.DoubleSide : THREE.FrontSide,
      roughness: kind === "eye" ? 0.28 : kind === "wing" ? 0.72 : kind === "oak" ? 0.68 : 0.5,
      metalness: kind === "eye" ? 0.08 : 0,
    };
    const material = kind === "eye" ? new THREE.MeshPhysicalMaterial({ ...options, clearcoat: 0.18, clearcoatRoughness: 0.32 }) : new THREE.MeshStandardMaterial(options);
    if (kind === "oak") {
      const texture = new THREE.TextureLoader().load(OAK_TEXTURE_URL, () => {
        this.needsRender = true;
        this.pendingVision = this.frameReady;
      });
      texture.wrapS = THREE.RepeatWrapping;
      texture.wrapT = THREE.RepeatWrapping;
      texture.colorSpace = THREE.SRGBColorSpace;
      const repeat = descriptor?.texrepeat;
      texture.repeat.set(Math.max(0.01, finiteNumber(repeat?.[0], 1)), Math.max(0.01, finiteNumber(repeat?.[1], 1)));
      texture.anisotropy = Math.min(this.renderer.capabilities.getMaxAnisotropy(), 8);
      material.map = texture;
      material.needsUpdate = true;
    }
    this.materials.push(material);
    this.materialCache.set(key, material);
    return material;
  }

  updateFrame(poses, snapshot = null) {
    if (!this.sceneReady) return;
    const values = poses instanceof Float32Array ? poses : new Float32Array(poses ?? new ArrayBuffer(0));
    const bodyWords = this.bodyGroups.length * BODY_POSE_STRIDE;
    const cameraWords = this.cameras.length * CAMERA_POSE_STRIDE;
    if (values.length < bodyWords + cameraWords) return;
    for (let index = 0; index < this.bodyGroups.length; index += 1) {
      const offset = index * BODY_POSE_STRIDE;
      const body = this.bodyGroups[index];
      body.position.set(values[offset] * MM_PER_UNIT, values[offset + 1] * MM_PER_UNIT, values[offset + 2] * MM_PER_UNIT);
      this.tempQuaternion.set(values[offset + 4], values[offset + 5], values[offset + 6], values[offset + 3]);
      if (this.tempQuaternion.lengthSq() < 1e-12) this.tempQuaternion.identity();
      else this.tempQuaternion.normalize();
      body.quaternion.copy(this.tempQuaternion);
    }
    const cameraOffset = bodyWords;
    for (let index = 0; index < this.cameras.length; index += 1) {
      const offset = cameraOffset + index * CAMERA_POSE_STRIDE;
      const pose = this.cameraPoses[index];
      pose.position.set(values[offset] * MM_PER_UNIT, values[offset + 1] * MM_PER_UNIT, values[offset + 2] * MM_PER_UNIT);
      pose.quaternion.copy(quaternionFromRotationRows(values.subarray(offset + 3, offset + 12)));
    }
    this.snapshot = snapshot ?? this.snapshot;
    this.updateFood(snapshot);
    this.updateObserverCamera();
    this.frameReady = true;
    this.needsRender = true;
    this.pendingVision = true;
  }

  updateFood(snapshot) {
    if (!snapshot || !Array.isArray(snapshot.food_center)) return;
    const center = vector3(snapshot.food_center);
    const enabled = snapshot.food_enabled !== false;
    for (const object of this.foodObjects) {
      object.position.set(center[0], center[1], center[2]);
      object.visible = object.userData.baseVisible && enabled;
    }
  }

  findCameraIndex(kind) {
    const names = this.cameras.map((camera) => String(camera?.name ?? "").toLowerCase());
    if (kind === "chase") {
      return names.findIndex((name) => name === "chase" || name.includes("chase") || name.includes("trackingcam"));
    }
    if (kind === "room") {
      return names.findIndex((name) => name === "room" || name.includes("room_camera") || name === "room");
    }
    return -1;
  }

  setCameraMode(mode) {
    const next = mode === "room" || mode === "orbit" ? mode : "chase";
    this.cameraMode = next;
    this.cameraPresetPending = next !== "orbit" || !this.frameReady;
    if (this.frameReady) this.updateObserverCamera();
  }

  updateObserverCamera() {
    const target = this.tempTarget.fromArray(this.snapshot?.root_position ?? [0, 0, 0]);
    if (this.cameraPresetPending) {
      const roomIndex = this.findCameraIndex("room");
      if (this.cameraMode === "room" && roomIndex >= 0) {
        this.camera.position.copy(this.cameraPoses[roomIndex].position);
        this.camera.quaternion.copy(this.cameraPoses[roomIndex].quaternion);
        this.camera.getWorldDirection(this.followDelta);
        this.controls.target.copy(this.camera.position).addScaledVector(this.followDelta, 650);
        this.camera.fov = this.cameras[roomIndex].fovy;
      } else {
        this.controls.target.copy(target);
        this.followDelta.set(-1, -1, 0.57).normalize().multiplyScalar(CHASE_DISTANCE_MM);
        this.camera.position.copy(target).add(this.followDelta);
        this.camera.fov = DEFAULT_FOV_DEG;
      }
      this.camera.up.set(0, 0, 1);
      this.camera.lookAt(this.controls.target);
      this.camera.updateProjectionMatrix();
      this.cameraPresetPending = false;
    } else if (this.cameraMode === "chase") {
      this.followDelta.copy(target).sub(this.followPosition);
      this.camera.position.add(this.followDelta);
      this.controls.target.add(this.followDelta);
    }
    this.followPosition.copy(target);
    this.controls.update();
  }

  applyWallCutaway(enabled) {
    const observer = this.camera.position;
    for (const object of this.objects) {
      const baseVisible = object.userData.baseVisible;
      if (!baseVisible) {
        object.visible = false;
        continue;
      }
      if (!enabled || !object.userData.wallRule) {
        if (object.userData.kind !== "food" || this.snapshot?.food_enabled !== false) object.visible = true;
        continue;
      }
      const rule = object.userData.wallRule;
      const plane = this.wallPlanes.get(rule.key);
      if (!Number.isFinite(plane)) {
        object.visible = true;
        continue;
      }
      object.updateWorldMatrix(true, false);
      object.getWorldPosition(this.tempPosition);
      const outside = observer.getComponent(rule.axis) * rule.sign > plane;
      object.visible = !outside;
    }
  }

  render() {
    this.controls.update();
    if (!this.needsRender && !this.pendingVision) return;
    if (this.sceneReady) {
      this.applyWallCutaway(true);
      this.tempTarget.fromArray(this.snapshot?.root_position ?? [0, 0, 0]);
      this.camera.near = THREE.MathUtils.clamp(this.camera.position.distanceTo(this.tempTarget) * 0.02, CAMERA_NEAR_MM, 2);
      this.camera.far = Math.max(CAMERA_FAR_MM, this.camera.position.distanceTo(this.controls.target) + 900);
      this.camera.updateProjectionMatrix();
    }
    if (this.needsRender) {
      this.renderer.setRenderTarget(null);
      this.renderer.render(this.scene, this.camera);
      this.needsRender = false;
    }
    if (this.pendingVision && !this.visionInFlight && performance.now() - this.lastVisionAt >= VISION_PERIOD_MS) {
      this.captureVision().catch(this.onError);
    }
  }

  async captureVision() {
    if (!this.sceneReady || !this.frameReady || this.cameras.length < 2 || typeof this.onVision !== "function") return;
    const eyeIndices = [
      this.cameras.findIndex((camera) => {
        const name = String(camera?.name ?? "").toLowerCase();
        return name.includes("l_eye") || name.includes("left_eye");
      }),
      this.cameras.findIndex((camera) => {
        const name = String(camera?.name ?? "").toLowerCase();
        return name.includes("r_eye") || name.includes("right_eye");
      }),
    ];
    if (eyeIndices.some((index) => index < 0)) return;
    this.visionInFlight = true;
    this.pendingVision = false;
    this.lastVisionAt = performance.now();

    this.applyWallCutaway(false);
    const hidden = [];
    for (const object of this.objects) {
      if (object.userData.isFly) {
        hidden.push([object, object.visible]);
        object.visible = false;
      }
    }
    const readbacks = [];
    try {
      try {
        for (let eye = 0; eye < 2; eye += 1) {
          const cameraIndex = eyeIndices[eye];
          const pose = this.cameraPoses[cameraIndex];
          const descriptor = this.cameras[cameraIndex];
          const camera = this.eyeCameras[eye];
          camera.position.copy(pose.position);
          camera.quaternion.copy(pose.quaternion);
          camera.fov = Math.min(170, Math.max(5, finiteNumber(descriptor?.fovy, EYE_FOV_DEG)));
          camera.near = 0.035;
          camera.far = CAMERA_FAR_MM;
          camera.updateProjectionMatrix();
          this.renderer.setRenderTarget(this.eyeRenderTargets[eye]);
          this.renderer.clear(true, true, true);
          this.renderer.render(this.scene, camera);
          this.renderer.shadowMap.autoUpdate = false;
          readbacks.push(this.renderer.readRenderTargetPixelsAsync(
            this.eyeRenderTargets[eye], 0, 0, RETINA_WIDTH, RETINA_HEIGHT, this.eyeReadbacks[eye],
          ));
        }
      } finally {
        for (const [object, visible] of hidden) object.visible = visible;
        this.applyWallCutaway(true);
        this.renderer.setRenderTarget(null);
        this.renderer.shadowMap.autoUpdate = true;
      }
      await Promise.all(readbacks);
      if (this.disposed) return;
      for (let eye = 0; eye < 2; eye += 1) {
        const rgba = this.eyeReadbacks[eye];
        const eyeOffset = eye * RETINA_EYE_BYTES;
        for (let y = 0; y < RETINA_HEIGHT; y += 1) {
          const sourceRow = RETINA_HEIGHT - 1 - y;
          for (let x = 0; x < RETINA_WIDTH; x += 1) {
            const source = (sourceRow * RETINA_WIDTH + x) * 4;
            const target = eyeOffset + (y * RETINA_WIDTH + x) * 3;
            this.visionScratch[target] = rgba[source];
            this.visionScratch[target + 1] = rgba[source + 1];
            this.visionScratch[target + 2] = rgba[source + 2];
          }
        }
      }
      this.onVision(this.visionScratch.slice().buffer);
    } catch (error) {
      await Promise.allSettled(readbacks);
      throw error;
    } finally {
      this.visionInFlight = false;
      if (this.disposed) this.disposeResources();
    }
  }

  setRetinaPixels(pixels, canvas) {
    drawRetinaPixels(canvas, pixels);
  }

  resize() {
    this.needsRender = true;
    const width = Math.max(1, this.canvas.clientWidth || this.canvas.width || window.innerWidth);
    const height = Math.max(1, this.canvas.clientHeight || this.canvas.height || window.innerHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  dispose() {
    this.disposed = true;
    if (!this.visionInFlight) this.disposeResources();
  }

  disposeResources() {
    this.disposeSceneObjects();
    for (const target of this.eyeRenderTargets) target.dispose();
    this.controls.dispose();
    this.renderer.dispose();
  }
}

export function createSceneRenderer(canvas, options) {
  return new FlySceneRenderer(canvas, options);
}

export function sceneDescriptorSummary(descriptor) {
  return {
    bodies: Number(descriptor?.bodyCount) || 0,
    geoms: Array.isArray(descriptor?.geoms) ? descriptor.geoms.length : 0,
    meshes: Array.isArray(descriptor?.meshes) ? descriptor.meshes.length : 0,
    cameras: Array.isArray(descriptor?.cameras) ? descriptor.cameras.length : 0,
    brain: descriptor?.brain ?? null,
  };
}

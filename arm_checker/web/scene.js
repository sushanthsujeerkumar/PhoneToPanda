import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// The server owns calibration and kinematics. This module only draws its result.
// Server coordinates are right-handed, metres, Z-up; Three uses Y-up.
const worldToView = (p) => new THREE.Vector3(p[0], p[2], -p[1]);
const basis = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(1, 0, 0),
  -Math.PI / 2,
);
const yAxis = new THREE.Vector3(0, 1, 0);
const COLORS = { upper: 0xa4e9ce, forearm: 0xf4aa97, steel: 0x7793a3 };

export function createArmScene({ canvas, container, labels }) {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x14202b, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.35;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(33, 1, 0.02, 20);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.09;
  controls.minDistance = 0.7;
  controls.maxDistance = 2.7;
  controls.maxPolarAngle = Math.PI * 0.96;
  controls.enablePan = false;
  controls.target.set(0.025, -0.25, 0);
  scene.add(new THREE.HemisphereLight(0xe3f0ff, 0x21323b, 2.2));
  const key = new THREE.DirectionalLight(0xf1fbff, 3.2);
  key.position.set(-2, 3, 2);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xa4d0e8, 2.3);
  rim.position.set(2, 1, -2);
  scene.add(rim);
  const warm = new THREE.DirectionalLight(0xffc2ab, 1.0);
  warm.position.set(1, -0.4, 2);
  scene.add(warm);

  const grid = new THREE.GridHelper(2.4, 24, 0x435a68, 0x314653);
  grid.position.y = -0.71;
  grid.material.transparent = true;
  grid.material.opacity = 0.35;
  scene.add(grid);
  const groundRing = new THREE.Mesh(
    new THREE.RingGeometry(0.38, 0.382, 96),
    new THREE.MeshBasicMaterial({
      color: 0x738b98,
      opacity: 0.18,
      transparent: true,
      side: THREE.DoubleSide,
    }),
  );
  groundRing.rotation.x = -Math.PI / 2;
  groundRing.position.y = -0.709;
  scene.add(groundRing);

  const shadowCanvas = document.createElement("canvas");
  shadowCanvas.width = shadowCanvas.height = 128;
  const context = shadowCanvas.getContext("2d");
  const gradient = context.createRadialGradient(64, 64, 4, 64, 64, 62);
  gradient.addColorStop(0, "#030a1377");
  gradient.addColorStop(1, "#030a1300");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 128, 128);
  const shadowTexture = new THREE.CanvasTexture(shadowCanvas);
  const shadow = new THREE.Mesh(
    new THREE.PlaneGeometry(0.9, 0.65),
    new THREE.MeshBasicMaterial({
      map: shadowTexture,
      transparent: true,
      depthWrite: false,
    }),
  );
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.set(0.02, -0.707, 0);
  scene.add(shadow);

  const rig = new THREE.Group();
  scene.add(rig);
  function material(color, extra = {}) {
    return new THREE.MeshStandardMaterial({
      color,
      roughness: 0.38,
      metalness: 0.25,
      ...extra,
    });
  }
  function segment(color, width) {
    const group = new THREE.Group();
    // A tapered open frame makes bending and sensor twist easy to distinguish.
    const geom = new THREE.CylinderGeometry(0.58, 0.8, 1, 6, 1, false);
    const shell = new THREE.Mesh(
      geom,
      material(color, { transparent: true, opacity: 0.58, flatShading: true }),
    );
    shell.position.y = 0.5;
    const edge = new THREE.LineSegments(
      new THREE.EdgesGeometry(geom),
      new THREE.LineBasicMaterial({ color, opacity: 0.65, transparent: true }),
    );
    edge.position.y = 0.5;
    const core = new THREE.Mesh(
      new THREE.CylinderGeometry(0.2, 0.26, 0.91, 12),
      material(0x536b77),
    );
    core.position.y = 0.5;
    group.add(shell, edge, core);
    group.userData.width = width;
    rig.add(group);
    return group;
  }
  const segments = {
    upper: segment(COLORS.upper, 0.063),
    forearm: segment(COLORS.forearm, 0.052),
  };
  function joint(radius, color) {
    const group = new THREE.Group();
    group.add(
      new THREE.Mesh(
        new THREE.SphereGeometry(radius, 24, 16),
        material(0x263d48),
      ),
    );
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(radius * 1.08, 0.0026, 10, 40),
      material(color, { emissive: color, emissiveIntensity: 0.13 }),
    );
    group.add(ring);
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(radius * 0.27, 12, 8),
      material(color),
    );
    dot.position.z = radius;
    group.add(dot);
    rig.add(group);
    return group;
  }
  const joints = {
    shoulder: joint(0.039, COLORS.steel),
    elbow: joint(0.031, COLORS.upper),
    wrist: joint(0.022, COLORS.forearm),
  };
  const anchor = new THREE.Group();
  const anchorRing = new THREE.Mesh(
    new THREE.TorusGeometry(0.07, 0.001, 6, 64),
    new THREE.MeshBasicMaterial({
      color: 0x6b8492,
      transparent: true,
      opacity: 0.55,
    }),
  );
  anchor.add(anchorRing);
  const anchorLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.16, 0, 0),
      new THREE.Vector3(0.16, 0, 0),
    ]),
    new THREE.LineBasicMaterial({
      color: 0x526b7b,
      transparent: true,
      opacity: 0.5,
    }),
  );
  anchor.add(anchorLine);
  scene.add(anchor);

  function sensor(color) {
    const group = new THREE.Group();
    const chip = new THREE.Mesh(
      new THREE.BoxGeometry(0.041, 0.018, 0.052),
      material(0x253d48),
    );
    const face = new THREE.Mesh(
      new THREE.BoxGeometry(0.035, 0.003, 0.046),
      material(color, { roughness: 0.55 }),
    );
    face.position.y = 0.011;
    const direction = new THREE.Mesh(
      new THREE.ConeGeometry(0.005, 0.012, 3),
      material(0x193229),
    );
    direction.rotation.x = Math.PI / 2;
    direction.position.set(0, 0.014, -0.006);
    group.add(chip, face, direction);
    rig.add(group);
    return group;
  }
  const sensors = {
    upper: sensor(COLORS.upper),
    forearm: sensor(COLORS.forearm),
  };
  const axes = new THREE.Group();
  for (const [vector, color, text] of [
    [[1, 0, 0], 0xeea292, "X"],
    [[0, 0, -1], 0xb9d7b9, "Y"],
    [[0, 1, 0], 0x9dbcf2, "Z"],
  ]) {
    const dir = new THREE.Vector3(...vector);
    axes.add(
      new THREE.ArrowHelper(
        dir,
        new THREE.Vector3(),
        0.115,
        color,
        0.018,
        0.008,
      ),
    );
    const textureCanvas = document.createElement("canvas");
    textureCanvas.width = textureCanvas.height = 64;
    const ctx = textureCanvas.getContext("2d");
    ctx.font = "500 32px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = `#${color.toString(16)}`;
    ctx.fillText(text, 32, 32);
    const texture = new THREE.CanvasTexture(textureCanvas);
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthTest: false,
      }),
    );
    sprite.scale.set(0.033, 0.033, 1);
    sprite.position.copy(dir).multiplyScalar(0.139);
    axes.add(sprite);
  }
  axes.position.set(-0.26, -0.7, 0.17);
  scene.add(axes);

  let pose = null,
    valid = false,
    lengths = [0.3, 0.26],
    showTrail = false,
    lastTrailTime = 0;
  const pointTargets = {
    shoulder: new THREE.Vector3(),
    elbow: new THREE.Vector3(),
    wrist: new THREE.Vector3(),
  };
  const points = Object.fromEntries(
    Object.entries(pointTargets).map(([k, p]) => [k, p.clone()]),
  );
  const trailPoints = [];
  const trailGeometry = new THREE.BufferGeometry();
  trailGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(150 * 3), 3),
  );
  trailGeometry.setDrawRange(0, 0);
  const trail = new THREE.Line(
    trailGeometry,
    new THREE.LineBasicMaterial({
      color: COLORS.forearm,
      transparent: true,
      opacity: 0.35,
    }),
  );
  trail.frustumCulled = false;
  scene.add(trail);
  let wasValid = false,
    lastFrame = performance.now(),
    currentView = "perspective",
    framingLength = 0.56;
  function setView(name) {
    currentView = name;
    const target = controls.target;
    const offsets = {
      perspective: [0.82, 0.32, 1.28],
      front: [1.5, 0.02, 0],
      side: [0, 0.02, 1.5],
      top: [0, 1.65, 0.001],
    };
    camera.position
      .copy(target)
      .add(
        new THREE.Vector3(
          ...(offsets[name] || offsets.perspective),
        ).multiplyScalar(framingLength / 0.56),
      );
    camera.up.set(0, 1, 0);
    controls.update();
  }
  setView("perspective");
  function update(state) {
    pose = state.pose || null;
    valid = Boolean(
      pose?.valid &&
        [pose.shoulder, pose.elbow, pose.wrist].every(
          (p) => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite),
        ),
    );
    const cfg = state.config || {};
    lengths = [
      Number(cfg.upper_length) || 0.3,
      Number(cfg.forearm_length) || 0.26,
    ];
    const total = lengths[0] + lengths[1];
    if (Math.abs(total - framingLength) > 0.001) {
      framingLength = total;
      controls.target.y = -total * 0.43;
      controls.minDistance = (0.7 * total) / 0.56;
      controls.maxDistance = (2.7 * total) / 0.56;
      grid.position.y = -total - 0.15;
      groundRing.position.y = grid.position.y + 0.001;
      shadow.position.y = grid.position.y + 0.003;
      axes.position.y = grid.position.y + 0.01;
      setView(currentView);
    }
    if (valid) {
      for (const key of Object.keys(pointTargets))
        pointTargets[key].copy(worldToView(pose[key]));
    } else if (!state.calibration?.neutral_ready) {
      pointTargets.shoulder.set(0, 0, 0);
      pointTargets.elbow.set(0.018, -lengths[0], 0);
      pointTargets.wrist
        .copy(pointTargets.elbow)
        .add(new THREE.Vector3(lengths[1] * 0.66, -lengths[1] * 0.75, 0.025));
    }
    if (!wasValid && valid) {
      for (const key of Object.keys(points))
        points[key].copy(pointTargets[key]);
    }
    if (wasValid !== valid) clearTrail();
    wasValid = valid;
  }
  update({});
  for (const key of Object.keys(points)) points[key].copy(pointTargets[key]);
  function clearTrail() {
    trailPoints.length = 0;
    trailGeometry.setDrawRange(0, 0);
  }
  clearTrail();
  function orientBone(mesh, a, b) {
    const delta = b.clone().sub(a);
    mesh.position.copy(a);
    mesh.quaternion.setFromUnitVectors(yAxis, delta.clone().normalize());
    mesh.scale.set(
      mesh.userData.width,
      Math.max(delta.length(), 0.001),
      mesh.userData.width,
    );
  }
  function orientSensor(mesh, a, b, q) {
    mesh.position.copy(a).lerp(b, 0.48);
    if (Array.isArray(q) && q.length === 4 && q.every(Number.isFinite)) {
      mesh.quaternion
        .copy(basis)
        .multiply(new THREE.Quaternion(q[1], q[2], q[3], q[0]).normalize());
    } else
      mesh.quaternion.setFromUnitVectors(yAxis, b.clone().sub(a).normalize());
    // Lift the sensor plate from the bone's surface in its local normal.
    mesh.position.add(
      new THREE.Vector3(0, 0.035, 0).applyQuaternion(mesh.quaternion),
    );
  }
  const resize = new ResizeObserver(() => {
    const width = container.clientWidth,
      height = container.clientHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
  });
  resize.observe(container);
  let disposed = false;
  function frame(now) {
    if (disposed) return;
    requestAnimationFrame(frame);
    const dt = Math.min((now - lastFrame) / 1000, 0.1);
    lastFrame = now;
    const alpha = window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ? 1
      : 1 - Math.exp(-dt * 24);
    for (const key of Object.keys(points)) {
      points[key].lerp(pointTargets[key], alpha);
      joints[key].position.copy(points[key]);
    }
    anchor.position.copy(points.shoulder);
    orientBone(segments.upper, points.shoulder, points.elbow);
    orientBone(segments.forearm, points.elbow, points.wrist);
    orientSensor(
      sensors.upper,
      points.shoulder,
      points.elbow,
      valid ? pose.upper_q : null,
    );
    orientSensor(
      sensors.forearm,
      points.elbow,
      points.wrist,
      valid ? pose.forearm_q : null,
    );
    rig.traverse((object) => {
      if (object.material && object.material.isMeshStandardMaterial)
        object.material.emissiveIntensity = valid ? 0.13 : 0;
    });
    if (valid && showTrail && now - lastTrailTime > 65) {
      const p = points.wrist.clone();
      if (!trailPoints.length || trailPoints.at(-1).distanceTo(p) > 0.002) {
        trailPoints.push(p);
        if (trailPoints.length > 150) trailPoints.shift();
        const position = trailGeometry.attributes.position;
        trailPoints.forEach((point, index) =>
          position.setXYZ(index, point.x, point.y, point.z),
        );
        position.needsUpdate = true;
        trailGeometry.setDrawRange(0, trailPoints.length);
      }
      lastTrailTime = now;
    }
    trail.visible = showTrail && valid && trailPoints.length > 1;
    controls.update();
    for (const [key, element] of Object.entries(labels)) {
      const projected = points[key].clone().project(camera);
      const x =
        (projected.x * 0.5 + 0.5) * container.clientWidth +
        (key === "elbow" ? -90 : 18);
      const y = (-projected.y * 0.5 + 0.5) * container.clientHeight - 5;
      element.style.transform = `translate(${Math.max(8, Math.min(container.clientWidth - 110, x))}px,${y}px)`;
      element.style.opacity =
        projected.z < 1 && y > 70 && y < container.clientHeight - 65
          ? valid
            ? "1"
            : ".6"
          : "0";
    }
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
  return {
    update,
    setView,
    clearTrail,
    setAxes(value) {
      axes.visible = value;
    },
    setTrail(value) {
      showTrail = value;
      if (!value) clearTrail();
    },
    dispose() {
      disposed = true;
      resize.disconnect();
      controls.dispose();
      renderer.dispose();
    },
    getView() {
      return currentView;
    },
  };
}

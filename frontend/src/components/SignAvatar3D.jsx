import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  Camera,
  Pause,
  Play,
  Repeat,
  RotateCcw,
  Sparkles,
  Maximize2,
  Minimize2,
} from "lucide-react";
import {
  getTotalPlanDurationMs,
  sampleSignPlanPose,
} from "../utils/signAnimations";

const CAMERA_PRESETS = {
  studio: {
    label: "Studio",
    pos: [0.26, 1.43, 1.64],
    target: [0, 1.24, 0.08],
  },
  front: {
    label: "Front",
    pos: [0, 1.38, 1.52],
    target: [0, 1.25, 0.08],
  },
  hands: {
    label: "Hands",
    pos: [0.18, 1.32, 0.96],
    target: [-0.04, 1.24, 0.2],
  },
};

const UPPER_ARM_LEN = 0.27;
const FOREARM_LEN = 0.25;

// Reusable scratch vectors for 60fps IK & orientation with zero GC churn.
const _vAB = new THREE.Vector3();
const _vPole = new THREE.Vector3();
const _vPerp = new THREE.Vector3();
const _vElbow = new THREE.Vector3();
const _vDir = new THREE.Vector3();
const _vSide = new THREE.Vector3();
const _vFinger = new THREE.Vector3();
const _vPalm = new THREE.Vector3();
const _vUp = new THREE.Vector3(0, 1, 0);
const _mBasis = new THREE.Matrix4();

function solveArmIK(shoulderWorld, wristWorld, upperLen, foreLen, poleHint, outElbow) {
  _vAB.subVectors(wristWorld, shoulderWorld);
  const rawDist = _vAB.length();
  const minDist = Math.abs(upperLen - foreLen) + 0.025;
  const maxDist = upperLen + foreLen - 0.012;
  const dist = Math.max(minDist, Math.min(maxDist, rawDist));

  if (rawDist < 1e-5) {
    _vAB.set(0, -1, 0);
  } else {
    _vAB.divideScalar(rawDist);
  }

  const cosAngle = Math.max(
    -1,
    Math.min(1, (upperLen * upperLen + dist * dist - foreLen * foreLen) / (2 * upperLen * dist))
  );
  const sinAngle = Math.sqrt(Math.max(0, 1 - cosAngle * cosAngle));

  _vPole.copy(poleHint);
  const dot = _vPole.dot(_vAB);
  _vPerp.copy(_vPole).addScaledVector(_vAB, -dot);
  if (_vPerp.lengthSq() < 1e-5) {
    _vPerp.set(0, 0, -1);
  }
  _vPerp.normalize();

  outElbow
    .copy(shoulderWorld)
    .addScaledVector(_vAB, upperLen * cosAngle)
    .addScaledVector(_vPerp, upperLen * sinAngle);
}

function orientLimbSegment(mesh, startPos, endPos) {
  _vDir.subVectors(endPos, startPos);
  const len = _vDir.length();
  if (len < 1e-5) return;
  mesh.position.copy(startPos).addScaledVector(_vDir, 0.5);
  _vDir.divideScalar(len);
  mesh.quaternion.setFromUnitVectors(_vUp, _vDir);
}

function orientHandFromAxes(handGroup, fingerArr, palmArr) {
  _vFinger.set(fingerArr[0], fingerArr[1], fingerArr[2]).normalize();
  _vPalm.set(palmArr[0], palmArr[1], palmArr[2]).normalize();
  _vSide.crossVectors(_vFinger, _vPalm);
  if (_vSide.lengthSq() < 1e-5) {
    _vSide.set(1, 0, 0);
  } else {
    _vSide.normalize();
  }
  _vPalm.crossVectors(_vSide, _vFinger).normalize();
  _mBasis.makeBasis(_vSide, _vFinger, _vPalm);
  handGroup.quaternion.setFromRotationMatrix(_mBasis);
}

function buildArticulatedHand(side, skinMat, jointMat) {
  // side: -1 = right hand (thumb on -X when palm faces +Z), +1 = left hand (thumb on +X)
  const root = new THREE.Group();

  const wristCap = new THREE.Mesh(
    new THREE.SphereGeometry(0.028, 16, 12),
    jointMat
  );
  root.add(wristCap);

  const palmGeo = new THREE.BoxGeometry(0.072, 0.082, 0.024);
  const palmMesh = new THREE.Mesh(palmGeo, skinMat);
  palmMesh.position.set(0, 0.044, 0);
  palmMesh.castShadow = true;
  root.add(palmMesh);

  // Build 4 main fingers: index, middle, ring, pinky
  const fingerSpecs = [
    { key: "index", x: side * -0.025, len: [0.036, 0.024, 0.018], r: 0.0085 },
    { key: "middle", x: side * -0.008, len: [0.04, 0.027, 0.019], r: 0.0088 },
    { key: "ring", x: side * 0.009, len: [0.036, 0.024, 0.018], r: 0.0082 },
    { key: "pinky", x: side * 0.025, len: [0.028, 0.02, 0.015], r: 0.0074 },
  ];

  const fingers = {};
  for (const spec of fingerSpecs) {
    const mcp = new THREE.Group();
    mcp.position.set(spec.x, 0.084, 0);
    root.add(mcp);

    const makeSeg = (length, radius) => {
      const seg = new THREE.Mesh(
        new THREE.CapsuleGeometry(radius, Math.max(0.004, length - radius * 1.4), 6, 10),
        skinMat
      );
      seg.position.set(0, length * 0.5, 0);
      seg.castShadow = true;
      return seg;
    };

    mcp.add(makeSeg(spec.len[0], spec.r));

    const pip = new THREE.Group();
    pip.position.set(0, spec.len[0], 0);
    mcp.add(pip);
    pip.add(makeSeg(spec.len[1], spec.r * 0.92));

    const dip = new THREE.Group();
    dip.position.set(0, spec.len[1], 0);
    pip.add(dip);
    dip.add(makeSeg(spec.len[2], spec.r * 0.85));

    fingers[spec.key] = { mcp, pip, dip };
  }

  // Articulated thumb on side * -0.036
  const thumbBase = new THREE.Group();
  thumbBase.position.set(side * -0.034, 0.026, 0.006);
  root.add(thumbBase);

  const thumbSeg1 = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.0095, 0.022, 6, 10),
    skinMat
  );
  thumbSeg1.position.set(0, 0.018, 0);
  thumbSeg1.castShadow = true;
  thumbBase.add(thumbSeg1);

  const thumbTip = new THREE.Group();
  thumbTip.position.set(0, 0.035, 0);
  thumbBase.add(thumbTip);

  const thumbSeg2 = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.0085, 0.018, 6, 10),
    skinMat
  );
  thumbSeg2.position.set(0, 0.015, 0);
  thumbSeg2.castShadow = true;
  thumbTip.add(thumbSeg2);

  fingers.thumb = { base: thumbBase, tip: thumbTip };
  return { root, fingers };
}

function applyHandJoints(handRig, joints, side) {
  if (!joints) return;
  for (const name of ["index", "middle", "ring", "pinky"]) {
    const f = handRig.fingers[name];
    const angles = joints[name];
    if (!f || !angles) continue;
    f.mcp.rotation.set(angles[0], 0, -side * angles[3]);
    f.pip.rotation.set(angles[1], 0, 0);
    f.dip.rotation.set(angles[2], 0, 0);
  }
  const tAngles = joints.thumb;
  if (tAngles && handRig.fingers.thumb) {
    // [baseFlex, tipFlex, spreadOut, opposeAcrossPalm]
    handRig.fingers.thumb.base.rotation.set(
      tAngles[0] + tAngles[3] * 0.45,
      -side * tAngles[3],
      side * tAngles[2]
    );
    handRig.fingers.thumb.tip.rotation.set(tAngles[1], 0, -side * tAngles[3] * 0.25);
  }
}

function buildSignerRig(scene) {
  const avatarRoot = new THREE.Group();
  scene.add(avatarRoot);

  const skinMat = new THREE.MeshStandardMaterial({
    color: 0xdba881,
    roughness: 0.74,
    metalness: 0,
  });
  const jacketMat = new THREE.MeshStandardMaterial({
    color: 0x247d78,
    roughness: 0.9,
    metalness: 0,
  });
  const trimMat = new THREE.MeshStandardMaterial({
    color: 0xd4edce,
    roughness: 0.35,
    metalness: 0.35,
    emissive: 0x629d83,
    emissiveIntensity: 0.08,
  });
  const jointMat = new THREE.MeshStandardMaterial({
    color: 0xdba881,
    roughness: 0.45,
    metalness: 0.3,
  });
  const hairMat = new THREE.MeshStandardMaterial({
    color: 0x30231f,
    roughness: 0.92,
    metalness: 0,
  });
  const browMat = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    roughness: 0.5,
  });
  const eyeWhiteMat = new THREE.MeshStandardMaterial({
    color: 0xf8fafc,
    roughness: 0.25,
  });
  const pupilMat = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    roughness: 0.2,
  });

  // Lower pelvis base
  const pelvis = new THREE.Mesh(
    new THREE.CylinderGeometry(0.145, 0.155, 0.22, 24),
    jacketMat
  );
  pelvis.position.set(0, 0.88, 0);
  avatarRoot.add(pelvis);

  // Upper torso pivots around hip center for natural role-shifts
  const torsoGroup = new THREE.Group();
  torsoGroup.position.set(0, 0.98, 0);
  avatarRoot.add(torsoGroup);

  const chest = new THREE.Mesh(
    new THREE.CylinderGeometry(0.185, 0.145, 0.42, 24),
    jacketMat
  );
  chest.position.set(0, 0.21, 0);
  chest.scale.set(1.12, 1, 0.72);
  chest.castShadow = true;
  torsoGroup.add(chest);

  const collarRing = new THREE.Mesh(
    new THREE.TorusGeometry(0.088, 0.012, 12, 28),
    trimMat
  );
  collarRing.position.set(0, 0.42, 0.005);
  collarRing.rotation.x = Math.PI * 0.46;
  torsoGroup.add(collarRing);

  const chestStripe = new THREE.Mesh(
    new THREE.BoxGeometry(0.008, 0.08, 0.012),
    trimMat
  );
  chestStripe.position.set(0, 0.36, 0.133);
  torsoGroup.add(chestStripe);

  // Small embroidered sun emblem gives Ray a recognizable studio uniform.
  const emblem = new THREE.Mesh(new THREE.CircleGeometry(0.024, 32), trimMat);
  emblem.position.set(0.094, 0.3, 0.127);
  torsoGroup.add(emblem);

  // Neck + Head
  const neck = new THREE.Mesh(
    new THREE.CylinderGeometry(0.048, 0.055, 0.1, 16),
    skinMat
  );
  neck.position.set(0, 0.45, 0);
  torsoGroup.add(neck);

  const headGroup = new THREE.Group();
  headGroup.position.set(0, 0.56, 0.01);
  torsoGroup.add(headGroup);

  const headMesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.115, 28, 22),
    skinMat
  );
  headMesh.scale.set(0.92, 1.08, 0.96);
  headMesh.castShadow = true;
  headGroup.add(headMesh);

  // Stylized hair cap
  const hairCap = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 24, 18, 0, Math.PI * 2, 0, Math.PI * 0.54),
    hairMat
  );
  hairCap.scale.set(0.95, 1.06, 1.02);
  hairCap.position.set(0, 0.015, -0.012);
  hairCap.rotation.x = -0.18;
  headGroup.add(hairCap);

  for (const [x, y, z, size] of [[-0.055, 0.103, 0.045, 0.05], [0, 0.118, 0.027, 0.055], [0.047, 0.108, 0.008, 0.046]]) {
    const lock = new THREE.Mesh(new THREE.SphereGeometry(size, 20, 16), hairMat);
    lock.position.set(x, y, z);
    lock.scale.set(1, 0.65, 0.8);
    headGroup.add(lock);
  }
  for (const side of [-1, 1]) {
    const ear = new THREE.Mesh(new THREE.SphereGeometry(0.023, 16, 12), skinMat);
    ear.scale.set(0.5, 1, 0.7);
    ear.position.set(side * 0.105, -0.003, 0);
    headGroup.add(ear);
  }
  const nose = new THREE.Mesh(new THREE.SphereGeometry(0.017, 16, 12), skinMat);
  nose.scale.set(0.72, 1, 0.9);
  nose.position.set(0, -0.01, 0.108);
  headGroup.add(nose);

  // Eyes + Pupils + Eyelid blink scale
  const makeEye = (x) => {
    const eyeGroup = new THREE.Group();
    eyeGroup.position.set(x, 0.018, 0.092);
    const sclera = new THREE.Mesh(new THREE.SphereGeometry(0.018, 16, 12), eyeWhiteMat);
    sclera.scale.set(1.05, 0.85, 0.6);
    eyeGroup.add(sclera);

    const pupil = new THREE.Mesh(new THREE.SphereGeometry(0.0095, 14, 10), pupilMat);
    pupil.position.set(0, 0, 0.0085);
    eyeGroup.add(pupil);
    headGroup.add(eyeGroup);
    return { eyeGroup, pupil };
  };
  const rightEye = makeEye(-0.036);
  const leftEye = makeEye(0.036);

  // Eyebrows for NMM question mood
  const browGeo = new THREE.CapsuleGeometry(0.0055, 0.024, 4, 8);
  const rightBrow = new THREE.Mesh(browGeo, browMat);
  rightBrow.rotation.z = Math.PI / 2;
  rightBrow.position.set(-0.036, 0.046, 0.1);
  headGroup.add(rightBrow);

  const leftBrow = new THREE.Mesh(browGeo, browMat);
  leftBrow.rotation.z = Math.PI / 2;
  leftBrow.position.set(0.036, 0.046, 0.1);
  headGroup.add(leftBrow);

  // Subtle smile arc
  const mouthMat = new THREE.MeshStandardMaterial({
    color: 0x9f1239,
    roughness: 0.4,
  });
  const mouth = new THREE.Mesh(
    new THREE.TorusGeometry(0.022, 0.0045, 8, 18, Math.PI * 0.72),
    mouthMat
  );
  mouth.position.set(0, -0.044, 0.098);
  mouth.rotation.z = Math.PI * 1.14;
  headGroup.add(mouth);

  // Shoulder anchors (attached to torso so torsoYaw turns shoulders!)
  const rightShoulderAnchor = new THREE.Object3D();
  rightShoulderAnchor.position.set(-0.2, 0.4, 0);
  torsoGroup.add(rightShoulderAnchor);

  const leftShoulderAnchor = new THREE.Object3D();
  leftShoulderAnchor.position.set(0.2, 0.4, 0);
  torsoGroup.add(leftShoulderAnchor);

  const buildArm = (side) => {
    const shoulderSphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.048, 18, 14),
      jacketMat
    );
    const elbowSphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.036, 16, 12),
      jointMat
    );
    const upperArm = new THREE.Mesh(
      new THREE.CylinderGeometry(0.042, 0.036, UPPER_ARM_LEN, 16),
      jacketMat
    );
    upperArm.castShadow = true;

    const forearm = new THREE.Mesh(
      new THREE.CylinderGeometry(0.034, 0.027, FOREARM_LEN, 16),
      skinMat
    );
    forearm.castShadow = true;

    const hand = buildArticulatedHand(side, skinMat, trimMat);

    avatarRoot.add(shoulderSphere, elbowSphere, upperArm, forearm, hand.root);
    return { shoulderSphere, elbowSphere, upperArm, forearm, hand };
  };

  const rightArm = buildArm(-1);
  const leftArm = buildArm(1);

  // Holographic spatial Locus A & Locus B rings in signing space
  const makeLocusRing = (x, colorHex) => {
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.075, 0.006, 12, 32),
      new THREE.MeshBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: 0.24,
      })
    );
    ring.position.set(x, 1.18, 0.28);
    avatarRoot.add(ring);
    return ring;
  };
  const locusARing = makeLocusRing(-0.34, 0x38bdf8);
  const locusBRing = makeLocusRing(0.22, 0x10b981);

  return {
    avatarRoot,
    torsoGroup,
    headGroup,
    rightEye,
    leftEye,
    rightBrow,
    leftBrow,
    mouth,
    rightShoulderAnchor,
    leftShoulderAnchor,
    rightArm,
    leftArm,
    locusARing,
    locusBRing,
  };
}

function formatNmmStatus(plan) {
  if (!plan?.steps?.length) return "Ready · Idle";
  const st = plan.sentence_type;
  const brow = plan.nmm?.brow_state;
  if (st === "wh_question") return "WH-Question · Brows Furrowed";
  if (st === "yes_no_question") return "Y/N Question · Brows Raised";
  if (st === "negation") return "Negation · Head Shake";
  if (st === "affirmation") return "Affirmation · Head Nod";
  if (brow === "raised") return "Brows Raised";
  if (brow === "furrowed") return "Brows Furrowed";
  return "Statement";
}

export default function SignAvatar3D({ signPlan, playSignal }) {
  const mountRef = useRef(null);
  const planRef = useRef(signPlan);
  const playingRef = useRef(true);
  const loopRef = useRef(false);
  const speedRef = useRef(1);
  const elapsedRef = useRef(0);
  const cameraPresetRef = useRef("studio");
  const cameraLerpRef = useRef(null);

  const [playing, setPlaying] = useState(true);
  const [looping, setLooping] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [cameraPreset, setCameraPreset] = useState("studio");
  const [expanded, setExpanded] = useState(false);
  const [renderError, setRenderError] = useState(false);
  const [hudState, setHudState] = useState({
    stepIndex: -1,
    activeLetter: null,
    activeLetterIndex: -1,
    progressPct: 0,
  });

  const totalDurationMs = useMemo(
    () => getTotalPlanDurationMs(signPlan),
    [signPlan]
  );

  useEffect(() => {
    if (!expanded) return undefined;
    const close = (event) => { if (event.key === "Escape") setExpanded(false); };
    document.addEventListener("keydown", close);
    return () => document.removeEventListener("keydown", close);
  }, [expanded]);

  useEffect(() => {
    planRef.current = signPlan;
    elapsedRef.current = 0;
    const hasSteps = Boolean(signPlan?.steps?.length);
    playingRef.current = hasSteps;
    setPlaying(hasSteps);
    setHudState({
      stepIndex: hasSteps ? 0 : -1,
      activeLetter: null,
      activeLetterIndex: -1,
      progressPct: 0,
    });
  }, [signPlan, playSignal]);

  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);

  useEffect(() => {
    loopRef.current = looping;
  }, [looping]);

  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  const handleSelectCameraPreset = (presetKey) => {
    setCameraPreset(presetKey);
    cameraPresetRef.current = presetKey;
    const spec = CAMERA_PRESETS[presetKey] || CAMERA_PRESETS.studio;
    cameraLerpRef.current = {
      pos: new THREE.Vector3(...spec.pos),
      target: new THREE.Vector3(...spec.target),
    };
  };

  const handleReplay = () => {
    elapsedRef.current = 0;
    playingRef.current = true;
    setPlaying(true);
  };

  const handleJumpToStep = (targetIdx) => {
    const steps = signPlan?.steps || [];
    let offset = 0;
    for (let i = 0; i < targetIdx && i < steps.length; i += 1) {
      offset += steps[i].duration_ms || 1100;
    }
    elapsedRef.current = offset + 20;
    playingRef.current = true;
    setPlaying(true);
  };

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return undefined;

    const width = container.clientWidth || 640;
    const height = container.clientHeight || 420;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x172c32);
    scene.fog = new THREE.FogExp2(0x172c32, 0.12);

    const camera = new THREE.PerspectiveCamera(36, width / height, 0.1, 20);
    const initCam = CAMERA_PRESETS.studio;
    camera.position.set(...initCam.pos);

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    } catch {
      setRenderError(true);
      return undefined;
    }
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.35;
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(...initCam.target);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.9;
    controls.maxDistance = 2.6;
    controls.maxPolarAngle = Math.PI * 0.62;
    controls.update();

    // Studio 3-point lighting (key + cyan left rim + emerald right rim)
    scene.add(new THREE.HemisphereLight(0xe8f5ee, 0x33434e, 2));
    const keyLight = new THREE.DirectionalLight(0xffe5cb, 2.5);
    keyLight.position.set(1.2, 2.4, 2.2);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    scene.add(keyLight);

    const cyanRim = new THREE.PointLight(0xb0e9dd, 2, 6);
    cyanRim.position.set(-1.3, 1.65, 0.8);
    scene.add(cyanRim);

    const greenRim = new THREE.PointLight(0xffd3a8, 1.5, 6);
    greenRim.position.set(1.3, 1.55, 0.75);
    scene.add(greenRim);

    // Subtle signing pedestal & grid
    const pedestal = new THREE.Mesh(
      new THREE.CylinderGeometry(0.62, 0.72, 0.04, 48),
      new THREE.MeshStandardMaterial({ color: 0x29444a, roughness: 0.85, metalness: 0.05 })
    );
    pedestal.position.set(0, 0.75, 0);
    pedestal.receiveShadow = true;
    scene.add(pedestal);

    const grid = new THREE.PolarGridHelper(0.9, 12, 4, 64, 0x8fc7b7, 0x36555b);
    grid.position.set(0, 0.772, 0);
    grid.material.opacity = 0.28;
    grid.material.transparent = true;
    scene.add(grid);

    const rig = buildSignerRig(scene);

    const rShoulderWorld = new THREE.Vector3();
    const lShoulderWorld = new THREE.Vector3();
    const rWristWorld = new THREE.Vector3();
    const lWristWorld = new THREE.Vector3();
    const rPole = new THREE.Vector3(-0.65, -0.35, -0.55).normalize();
    const lPole = new THREE.Vector3(0.65, -0.35, -0.55).normalize();

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        const h = entry.contentRect.height;
        if (w > 10 && h > 10) {
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        }
      }
    });
    resizeObserver.observe(container);

    let rafId = 0;
    let lastTime = performance.now();
    let lastHudSync = 0;
    let lastStepSent = -99;
    let lastLetterIdxSent = -99;

    const animate = (now) => {
      const dtMs = Math.min(60, now - lastTime);
      lastTime = now;

      const activePlan = planRef.current;
      const totalMs = getTotalPlanDurationMs(activePlan);

      if (playingRef.current && totalMs > 0) {
        elapsedRef.current += dtMs * speedRef.current;
        if (elapsedRef.current >= totalMs) {
          if (loopRef.current) {
            elapsedRef.current = 0;
          } else {
            elapsedRef.current = totalMs;
            playingRef.current = false;
            setPlaying(false);
          }
        }
      }

      const sampled = sampleSignPlanPose(
        activePlan,
        elapsedRef.current,
        now / 1000
      );
      const { pose } = sampled;

      // Apply torso role-shift first so shoulder world positions update accurately.
      rig.torsoGroup.rotation.y = pose.torsoYaw;
      rig.torsoGroup.updateMatrixWorld(true);

      rig.headGroup.rotation.set(pose.head.pitch, pose.head.yaw, pose.head.roll);

      // Eyebrows (raise for Y/N questions, furrow & tilt inward for WH-questions)
      const browY = 0.046 + pose.browOffset * 0.011 - pose.browFurrow * 0.008;
      rig.rightBrow.position.y = browY;
      rig.leftBrow.position.y = browY;
      rig.rightBrow.rotation.z = Math.PI / 2 - pose.browFurrow * 0.24;
      rig.leftBrow.rotation.z = Math.PI / 2 + pose.browFurrow * 0.24;

      // Eyes: gaze tracking + natural periodic blink
      const blinkCycle = (now / 1000) % 4.1;
      const eyeScaleY = blinkCycle > 3.96 ? 0.15 : 1.0;
      rig.rightEye.eyeGroup.scale.y = eyeScaleY;
      rig.leftEye.eyeGroup.scale.y = eyeScaleY;
      rig.rightEye.pupil.position.x = pose.gazeX * 0.0045;
      rig.leftEye.pupil.position.x = pose.gazeX * 0.0045;

      // Mouth smile inflection
      const smileScale = 0.75 + pose.smile * 0.45;
      rig.mouth.scale.set(smileScale, 0.7 + pose.smile * 0.65, 1);

      // Solve 2-bone arm IK for both arms
      rig.rightShoulderAnchor.getWorldPosition(rShoulderWorld);
      rig.leftShoulderAnchor.getWorldPosition(lShoulderWorld);

      rWristWorld.set(pose.right.wrist[0], pose.right.wrist[1], pose.right.wrist[2]);
      lWristWorld.set(pose.left.wrist[0], pose.left.wrist[1], pose.left.wrist[2]);

      solveArmIK(rShoulderWorld, rWristWorld, UPPER_ARM_LEN, FOREARM_LEN, rPole, _vElbow);
      rig.rightArm.shoulderSphere.position.copy(rShoulderWorld);
      rig.rightArm.elbowSphere.position.copy(_vElbow);
      orientLimbSegment(rig.rightArm.upperArm, rShoulderWorld, _vElbow);
      orientLimbSegment(rig.rightArm.forearm, _vElbow, rWristWorld);
      rig.rightArm.hand.root.position.copy(rWristWorld);
      orientHandFromAxes(rig.rightArm.hand.root, pose.right.finger, pose.right.palm);
      applyHandJoints(rig.rightArm.hand, pose.right.joints, -1);

      solveArmIK(lShoulderWorld, lWristWorld, UPPER_ARM_LEN, FOREARM_LEN, lPole, _vElbow);
      rig.leftArm.shoulderSphere.position.copy(lShoulderWorld);
      rig.leftArm.elbowSphere.position.copy(_vElbow);
      orientLimbSegment(rig.leftArm.upperArm, lShoulderWorld, _vElbow);
      orientLimbSegment(rig.leftArm.forearm, _vElbow, lWristWorld);
      rig.leftArm.hand.root.position.copy(lWristWorld);
      orientHandFromAxes(rig.leftArm.hand.root, pose.left.finger, pose.left.palm);
      applyHandJoints(rig.leftArm.hand, pose.left.joints, 1);

      // Highlight spatial Locus A / Locus B rings
      rig.locusARing.material.opacity = pose.activeLoci?.left ? 0.92 : 0.16;
      rig.locusBRing.material.opacity = pose.activeLoci?.right ? 0.92 : 0.16;
      const pulse = 1 + Math.sin(now * 0.008) * 0.08;
      rig.locusARing.scale.setScalar(pose.activeLoci?.left ? pulse : 1);
      rig.locusBRing.scale.setScalar(pose.activeLoci?.right ? pulse : 1);

      // Smooth camera preset transition if triggered
      if (cameraLerpRef.current) {
        camera.position.lerp(cameraLerpRef.current.pos, 0.14);
        controls.target.lerp(cameraLerpRef.current.target, 0.14);
        if (camera.position.distanceTo(cameraLerpRef.current.pos) < 0.01) {
          cameraLerpRef.current = null;
        }
      }

      controls.update();
      renderer.render(scene, camera);

      const pct = Math.round(sampled.progress * 100);
      if (
        sampled.stepIndex !== lastStepSent ||
        sampled.activeLetterIndex !== lastLetterIdxSent ||
        now - lastHudSync > 85
      ) {
        lastStepSent = sampled.stepIndex;
        lastLetterIdxSent = sampled.activeLetterIndex;
        lastHudSync = now;
        setHudState({
          stepIndex: sampled.stepIndex,
          activeLetter: sampled.activeLetter,
          activeLetterIndex: sampled.activeLetterIndex,
          progressPct: pct,
        });
      }

      rafId = window.requestAnimationFrame(animate);
    };

    rafId = window.requestAnimationFrame(animate);

    return () => {
      window.cancelAnimationFrame(rafId);
      resizeObserver.disconnect();
      controls.dispose();
      scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m) => m.dispose());
          } else {
            obj.material.dispose();
          }
        }
      });
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  const steps = signPlan?.steps || [];
  const activeStep =
    hudState.stepIndex >= 0 && hudState.stepIndex < steps.length
      ? steps[hudState.stepIndex]
      : null;

  return (
    <div className={`avatar-stage-shell ${expanded ? "expanded" : ""}`}>
      <div className="ray-stage-title"><div className="ray-avatar-mark">R</div><div><strong>Ray</strong><span>Your signing companion</span></div><span className={`ray-status ${playing ? "is-signing" : ""}`}><i />{renderError ? "3D unavailable" : playing ? "Signing for you" : "Ready when you are"}</span><button type="button" className="stage-expand" onClick={() => setExpanded(v => !v)} aria-label={expanded ? "Minimize Ray’s stage" : "Expand Ray’s stage"} aria-pressed={expanded}>{expanded ? <Minimize2 size={17} /> : <Maximize2 size={17} />}</button></div>
      <div className="avatar-viewport-wrap">
        <div ref={mountRef} className="avatar-canvas-mount" role="img" aria-label="Ray, an animated 3D signing companion. Drag to rotate; use the camera buttons for preset views." />
        {renderError && <div className="avatar-render-fallback" role="status"><Sparkles size={32} /><h3>Ray needs a 3D-capable browser</h3><p>Enable hardware acceleration or use the Video / Cue Card view to follow along.</p></div>}

        <div className="avatar-hud-top">
          <div className="avatar-gloss-badge">
            <Sparkles size={14} />
            <span>
              {activeStep
                ? activeStep.type === "fingerspell" && hudState.activeLetter
                  ? `FINGERSPELL · ${activeStep.gloss} [${hudState.activeLetter}]`
                  : activeStep.gloss
                : "HELLO, I’M RAY"}
            </span>
          </div>
          <span className="avatar-nmm-pill">{formatNmmStatus(signPlan)}</span>
        </div>

        {activeStep && (
          <div className="avatar-cue-caption">
            <p>{activeStep.cue}</p>
            {activeStep.type === "fingerspell" && activeStep.letters?.length > 0 && (
              <div className="avatar-letter-row">
                {activeStep.letters.map((ch, idx) => (
                  <span
                    key={`${ch}-${idx}`}
                    className={`avatar-letter-chip ${
                      hudState.activeLetterIndex === idx ? "active" : ""
                    }`}
                  >
                    {ch}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        <div className="avatar-orbit-hint">Drag to rotate · Scroll to zoom</div>
      </div>

      <div className="avatar-controls-bar">
        <div className="avatar-play-group">
          <button
            type="button"
            className="primary-btn avatar-play-btn"
            onClick={() => {
              if (!playing && totalDurationMs > 0 && elapsedRef.current >= totalDurationMs) {
                elapsedRef.current = 0;
              }
              setPlaying((p) => !p);
            }}
            disabled={steps.length === 0}
          >
            {playing ? <Pause size={15} /> : <Play size={15} />}
            <span>{playing ? "Pause" : "Play"}</span>
          </button>

          <button
            type="button"
            className="icon-btn"
            onClick={handleReplay}
            disabled={steps.length === 0}
            title="Replay sign sequence"
          >
            <RotateCcw size={15} />
            <span>Replay</span>
          </button>

          <button
            type="button"
            className={`icon-btn ${looping ? "active" : ""}`}
            onClick={() => setLooping((l) => !l)}
            title="Loop playback"
            aria-pressed={looping}
          >
            <Repeat size={15} />
            <span>Loop</span>
          </button>

          <label className="select-pill" title="Signing Speed">
            <span>Speed</span>
            <select
              aria-label="Ray’s signing speed"
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
            >
              <option value={0.75}>0.75×</option>
              <option value={1}>1.0×</option>
              <option value={1.25}>1.25×</option>
              <option value={1.5}>1.5×</option>
            </select>
          </label>
        </div>

        <div className="avatar-camera-group">
          <Camera size={14} className="muted-icon" />
          {Object.entries(CAMERA_PRESETS).map(([key, item]) => (
            <button
              key={key}
              type="button"
              className={`chip-btn ${cameraPreset === key ? "selected" : ""}`}
              onClick={() => handleSelectCameraPreset(key)}
              aria-pressed={cameraPreset === key}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {steps.length > 0 && (
        <div className="avatar-timeline-wrap">
          <div className="timeline-label"><span>FOLLOW ALONG</span><span>{Math.round(totalDurationMs * hudState.progressPct / 100000)}s / {(totalDurationMs / 1000).toFixed(1)}s</span></div>
          <input className="avatar-seek" type="range" min="0" max="100" value={hudState.progressPct} aria-label="Ray’s signing progress" style={{ "--progress": `${hudState.progressPct}%` }} onChange={(event) => { const pct = Number(event.target.value); elapsedRef.current = totalDurationMs * pct / 100; setHudState(current => ({ ...current, progressPct: pct })); }} />
          <div className="sign-sequence-strip">
            <small>Signs</small>
            {steps.map((step, idx) => (
              <button
                key={`${step.id}-${idx}`}
                type="button"
                className={`seq-pill ${
                  hudState.stepIndex === idx ? "active" : ""
                }`}
                onClick={() => handleJumpToStep(idx)}
              >
                {idx + 1}. {step.gloss}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

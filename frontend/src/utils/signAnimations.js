/**
 * Articulated 3D handshapes, keyframed sign trajectories, NMM expressions,
 * and smooth co-articulation interpolation for SignAvatar3D.
 */

// Handshape joint angles (radians):
// thumb: [baseFlex, tipFlex, spreadOut, opposeAcrossPalm]
// fingers (index/middle/ring/pinky): [mcpFlex, pipFlex, dipFlex, lateralSpread]
export const HANDSHAPES = {
  relaxed: {
    thumb: [0.25, 0.22, 0.38, 0.45],
    index: [0.3, 0.4, 0.25, 0.04],
    middle: [0.35, 0.45, 0.28, 0.01],
    ring: [0.42, 0.5, 0.32, -0.02],
    pinky: [0.48, 0.55, 0.35, -0.06],
  },
  open_flat: {
    thumb: [0.15, 0.1, 0.22, 0.35],
    index: [0.04, 0.05, 0.04, 0.02],
    middle: [0.04, 0.05, 0.04, 0.0],
    ring: [0.05, 0.06, 0.04, -0.01],
    pinky: [0.06, 0.07, 0.05, -0.03],
  },
  open_spread: {
    thumb: [0.05, 0.05, 0.62, 0.2],
    index: [0.04, 0.05, 0.03, 0.18],
    middle: [0.04, 0.05, 0.03, 0.04],
    ring: [0.04, 0.05, 0.03, -0.07],
    pinky: [0.05, 0.06, 0.04, -0.2],
  },
  fist: {
    thumb: [0.75, 0.85, 0.12, 1.05],
    index: [1.35, 1.5, 1.1, 0.02],
    middle: [1.38, 1.52, 1.12, 0.0],
    ring: [1.4, 1.55, 1.15, -0.02],
    pinky: [1.42, 1.55, 1.15, -0.04],
  },
  fist_thumb_up: {
    thumb: [0.04, 0.04, 0.68, 0.12],
    index: [1.35, 1.5, 1.1, 0.02],
    middle: [1.38, 1.52, 1.12, 0.0],
    ring: [1.4, 1.55, 1.15, -0.02],
    pinky: [1.42, 1.55, 1.15, -0.04],
  },
  point: {
    thumb: [0.65, 0.72, 0.16, 0.92],
    index: [0.03, 0.04, 0.03, 0.02],
    middle: [1.35, 1.5, 1.1, 0.0],
    ring: [1.4, 1.52, 1.12, -0.02],
    pinky: [1.42, 1.55, 1.15, -0.04],
  },
  v_hand: {
    thumb: [0.68, 0.78, 0.16, 0.95],
    index: [0.04, 0.05, 0.04, 0.18],
    middle: [0.04, 0.05, 0.04, -0.14],
    ring: [1.38, 1.52, 1.12, -0.02],
    pinky: [1.42, 1.55, 1.15, -0.04],
  },
  h_hand: {
    thumb: [0.68, 0.78, 0.16, 0.95],
    index: [0.04, 0.05, 0.04, 0.02],
    middle: [0.04, 0.05, 0.04, -0.01],
    ring: [1.38, 1.52, 1.12, -0.02],
    pinky: [1.42, 1.55, 1.15, -0.04],
  },
  w_hand: {
    thumb: [0.58, 0.65, 0.22, 1.05],
    index: [0.04, 0.05, 0.04, 0.12],
    middle: [0.04, 0.05, 0.04, 0.0],
    ring: [0.05, 0.06, 0.04, -0.12],
    pinky: [1.35, 1.48, 1.1, -0.04],
  },
  ily: {
    thumb: [0.04, 0.04, 0.65, 0.18],
    index: [0.03, 0.04, 0.03, 0.08],
    middle: [1.38, 1.52, 1.12, 0.0],
    ring: [1.4, 1.55, 1.15, -0.02],
    pinky: [0.04, 0.05, 0.04, -0.14],
  },
  y_hand: {
    thumb: [0.04, 0.04, 0.65, 0.15],
    index: [1.35, 1.5, 1.1, 0.02],
    middle: [1.38, 1.52, 1.12, 0.0],
    ring: [1.4, 1.55, 1.15, -0.02],
    pinky: [0.04, 0.05, 0.04, -0.15],
  },
  pinky_up: {
    thumb: [0.7, 0.8, 0.16, 0.95],
    index: [1.35, 1.5, 1.1, 0.02],
    middle: [1.38, 1.52, 1.12, 0.0],
    ring: [1.4, 1.55, 1.15, -0.02],
    pinky: [0.04, 0.05, 0.04, -0.08],
  },
  cupped: {
    thumb: [0.42, 0.52, 0.35, 0.78],
    index: [0.62, 0.72, 0.52, 0.04],
    middle: [0.65, 0.75, 0.55, 0.0],
    ring: [0.68, 0.78, 0.58, -0.03],
    pinky: [0.72, 0.82, 0.6, -0.06],
  },
  claw: {
    thumb: [0.45, 0.65, 0.45, 0.65],
    index: [0.55, 1.05, 0.85, 0.1],
    middle: [0.58, 1.08, 0.88, 0.02],
    ring: [0.6, 1.1, 0.9, -0.05],
    pinky: [0.65, 1.12, 0.92, -0.12],
  },
  no_open: {
    thumb: [0.22, 0.18, 0.42, 0.68],
    index: [0.18, 0.15, 0.1, 0.04],
    middle: [0.18, 0.15, 0.1, -0.02],
    ring: [1.38, 1.52, 1.12, -0.02],
    pinky: [1.42, 1.55, 1.15, -0.04],
  },
  no_closed: {
    thumb: [0.45, 0.45, 0.25, 0.88],
    index: [0.95, 0.52, 0.25, 0.03],
    middle: [0.95, 0.52, 0.25, -0.02],
    ring: [1.4, 1.54, 1.15, -0.02],
    pinky: [1.44, 1.56, 1.16, -0.04],
  },
  o_hand: {
    thumb: [0.52, 0.65, 0.3, 0.95],
    index: [0.85, 0.98, 0.75, 0.02],
    middle: [0.88, 1.0, 0.78, 0.0],
    ring: [0.9, 1.02, 0.8, -0.02],
    pinky: [0.92, 1.04, 0.82, -0.04],
  },
  f_hand: {
    thumb: [0.55, 0.65, 0.28, 0.95],
    index: [0.92, 1.02, 0.72, 0.02],
    middle: [0.05, 0.06, 0.04, 0.08],
    ring: [0.05, 0.06, 0.04, -0.04],
    pinky: [0.06, 0.07, 0.05, -0.14],
  },
  l_hand: {
    thumb: [0.04, 0.04, 0.78, 0.1],
    index: [0.03, 0.04, 0.03, 0.04],
    middle: [1.38, 1.52, 1.12, 0.0],
    ring: [1.4, 1.55, 1.15, -0.02],
    pinky: [1.42, 1.55, 1.15, -0.04],
  },
  k_hand: {
    thumb: [0.35, 0.25, 0.38, 0.78],
    index: [0.05, 0.06, 0.04, 0.12],
    middle: [0.38, 0.18, 0.08, -0.1],
    ring: [1.38, 1.52, 1.12, -0.02],
    pinky: [1.42, 1.55, 1.15, -0.04],
  },
  three_hand: {
    thumb: [0.04, 0.04, 0.66, 0.15],
    index: [0.04, 0.05, 0.04, 0.12],
    middle: [0.04, 0.05, 0.04, -0.08],
    ring: [1.38, 1.52, 1.12, -0.02],
    pinky: [1.42, 1.55, 1.15, -0.04],
  },
  four_hand: {
    thumb: [0.72, 0.82, 0.12, 1.05],
    index: [0.04, 0.05, 0.04, 0.14],
    middle: [0.04, 0.05, 0.04, 0.03],
    ring: [0.04, 0.05, 0.04, -0.06],
    pinky: [0.05, 0.06, 0.04, -0.16],
  },
  r_hand: {
    thumb: [0.68, 0.75, 0.18, 0.95],
    index: [0.06, 0.08, 0.05, -0.06],
    middle: [0.12, 0.12, 0.08, 0.08],
    ring: [1.38, 1.52, 1.12, -0.02],
    pinky: [1.42, 1.55, 1.15, -0.04],
  },
  x_hand: {
    thumb: [0.7, 0.8, 0.15, 0.98],
    index: [0.55, 1.15, 0.9, 0.02],
    middle: [1.38, 1.52, 1.12, 0.0],
    ring: [1.4, 1.55, 1.15, -0.02],
    pinky: [1.42, 1.55, 1.15, -0.04],
  },
};

const LETTER_HANDSHAPE = {
  A: { shape: "fist_thumb_up", palm: [0, 0, 1], finger: [0, 1, 0] },
  B: { shape: "open_flat", palm: [0, 0, 1], finger: [0, 1, 0] },
  C: { shape: "cupped", palm: [0.95, 0, 0.25], finger: [0, 1, 0] },
  D: { shape: "point", palm: [0, 0, 1], finger: [0, 1, 0] },
  E: { shape: "claw", palm: [0, 0, 1], finger: [0, 1, 0] },
  F: { shape: "f_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
  G: { shape: "point", palm: [0.95, 0, 0.15], finger: [0.95, 0.1, 0.2] },
  H: { shape: "h_hand", palm: [0.95, 0, 0.15], finger: [0.95, 0.05, 0.2] },
  I: { shape: "pinky_up", palm: [0, 0, 1], finger: [0, 1, 0] },
  J: { shape: "pinky_up", palm: [0.5, 0, 0.85], finger: [0, 1, 0] },
  K: { shape: "k_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
  L: { shape: "l_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
  M: { shape: "fist", palm: [0, -0.2, 0.98], finger: [0, 0.8, 0.55] },
  N: { shape: "fist", palm: [0, -0.1, 0.99], finger: [0, 0.9, 0.4] },
  O: { shape: "o_hand", palm: [0.9, 0, 0.35], finger: [0, 1, 0] },
  P: { shape: "k_hand", palm: [0, -0.85, 0.5], finger: [0.3, -0.5, 0.8] },
  Q: { shape: "point", palm: [0, -0.85, 0.5], finger: [0.2, -0.75, 0.65] },
  R: { shape: "r_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
  S: { shape: "fist", palm: [0, 0, 1], finger: [0, 1, 0] },
  T: { shape: "fist", palm: [0, 0.1, 0.99], finger: [0, 0.95, 0.25] },
  U: { shape: "h_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
  V: { shape: "v_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
  W: { shape: "w_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
  X: { shape: "x_hand", palm: [0.85, 0, 0.5], finger: [0, 1, 0] },
  Y: { shape: "y_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
  Z: { shape: "point", palm: [0, 0, 1], finger: [-0.25, 0.9, 0.35] },
  "0": { shape: "o_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
  "1": { shape: "point", palm: [0, 0, 1], finger: [0, 1, 0] },
  "2": { shape: "v_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
  "3": { shape: "three_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
  "4": { shape: "four_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
  "5": { shape: "open_spread", palm: [0, 0, 1], finger: [0, 1, 0] },
  "6": { shape: "w_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
  "7": { shape: "w_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
  "8": { shape: "w_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
  "9": { shape: "f_hand", palm: [0, 0, 1], finger: [0, 1, 0] },
};

export const IDLE_POSE = {
  right: {
    wrist: [-0.25, 1.04, 0.16],
    palm: [0.7, 0.1, 0.7],
    finger: [0.12, -0.95, 0.25],
    shape: "relaxed",
  },
  left: {
    wrist: [0.25, 1.04, 0.16],
    palm: [-0.7, 0.1, 0.7],
    finger: [-0.12, -0.95, 0.25],
    shape: "relaxed",
  },
  head: { pitch: 0, yaw: 0, roll: 0 },
  torsoYaw: 0,
  gazeX: 0,
  browOffset: 0,
  browFurrow: 0,
  smile: 0.25,
  activeLoci: { left: false, right: false },
};

// Keyframes per sign: each keyframe has normalized t in [0..1].
// Coordinates are in avatar world meters (avatar stands at origin facing +Z;
// avatar's right side is -X, avatar's left side is +X).
export const SIGN_TRACKS = {
  hello: [
    {
      t: 0.0,
      right: { wrist: [-0.2, 1.38, 0.18], palm: [0.1, 0, 0.98], finger: [0.15, 0.95, 0.22], shape: "open_flat" },
      head: { pitch: -0.04, yaw: -0.05, roll: 0.03 },
      smile: 0.75,
    },
    {
      t: 0.35,
      right: { wrist: [-0.14, 1.56, 0.2], palm: [0.15, 0, 0.98], finger: [-0.08, 0.96, 0.24], shape: "open_spread" },
      head: { pitch: -0.05, yaw: 0.04, roll: 0.06 },
      smile: 0.9,
    },
    {
      t: 0.72,
      right: { wrist: [-0.34, 1.54, 0.25], palm: [0.25, 0, 0.95], finger: [-0.28, 0.92, 0.25], shape: "open_spread" },
      head: { pitch: -0.03, yaw: -0.04, roll: 0.04 },
      smile: 0.85,
    },
    {
      t: 1.0,
      right: { wrist: [-0.28, 1.46, 0.24], palm: [0.15, 0, 0.98], finger: [-0.12, 0.95, 0.24], shape: "open_flat" },
      head: { pitch: 0, yaw: 0, roll: 0 },
      smile: 0.65,
    },
  ],
  thank_you: [
    {
      t: 0.0,
      right: { wrist: [-0.04, 1.43, 0.2], palm: [0, 0.15, -0.98], finger: [-0.08, 0.95, -0.25], shape: "open_flat" },
      head: { pitch: 0.02, yaw: 0, roll: 0 },
      smile: 0.8,
    },
    {
      t: 0.35,
      right: { wrist: [-0.04, 1.45, 0.22], palm: [0, 0.55, -0.82], finger: [0, 0.82, 0.55], shape: "open_flat" },
      head: { pitch: 0.08, yaw: 0, roll: 0 },
      smile: 0.95,
    },
    {
      t: 0.85,
      right: { wrist: [-0.06, 1.32, 0.36], palm: [0, 0.95, 0.28], finger: [0, 0.25, 0.96], shape: "open_flat" },
      head: { pitch: 0.1, yaw: 0, roll: 0 },
      smile: 0.85,
    },
    {
      t: 1.0,
      right: { wrist: [-0.1, 1.28, 0.34], palm: [0, 0.92, 0.35], finger: [0, 0.3, 0.95], shape: "open_flat" },
      head: { pitch: 0.03, yaw: 0, roll: 0 },
      smile: 0.65,
    },
  ],
  help: [
    {
      t: 0.0,
      left: { wrist: [0.06, 1.15, 0.26], palm: [0, 0.98, 0.2], finger: [-0.35, 0.2, 0.9], shape: "open_flat" },
      right: { wrist: [-0.04, 1.24, 0.27], palm: [0.85, 0.15, 0.5], finger: [0.25, 0.15, 0.95], shape: "fist_thumb_up" },
      smile: 0.35,
    },
    {
      t: 0.45,
      left: { wrist: [0.05, 1.21, 0.28], palm: [0, 0.98, 0.2], finger: [-0.35, 0.2, 0.9], shape: "open_flat" },
      right: { wrist: [0.01, 1.27, 0.28], palm: [0.85, 0.15, 0.5], finger: [0.25, 0.15, 0.95], shape: "fist_thumb_up" },
      head: { pitch: -0.03, yaw: 0, roll: 0 },
    },
    {
      t: 0.85,
      left: { wrist: [0.04, 1.34, 0.3], palm: [0, 0.96, 0.25], finger: [-0.35, 0.2, 0.9], shape: "open_flat" },
      right: { wrist: [0.01, 1.4, 0.3], palm: [0.85, 0.15, 0.5], finger: [0.25, 0.15, 0.95], shape: "fist_thumb_up" },
      head: { pitch: -0.05, yaw: 0, roll: 0 },
    },
    {
      t: 1.0,
      left: { wrist: [0.05, 1.31, 0.29], palm: [0, 0.96, 0.25], finger: [-0.35, 0.2, 0.9], shape: "open_flat" },
      right: { wrist: [0.0, 1.37, 0.29], palm: [0.85, 0.15, 0.5], finger: [0.25, 0.15, 0.95], shape: "fist_thumb_up" },
    },
  ],
  yes: [
    {
      t: 0.0,
      right: { wrist: [-0.2, 1.34, 0.26], palm: [0.1, 0.1, 0.98], finger: [0, 0.98, 0.15], shape: "fist" },
      head: { pitch: -0.02, yaw: 0, roll: 0 },
      smile: 0.65,
    },
    {
      t: 0.28,
      right: { wrist: [-0.2, 1.28, 0.3], palm: [0.1, -0.55, 0.82], finger: [0, 0.65, 0.75], shape: "fist" },
      head: { pitch: 0.12, yaw: 0, roll: 0 },
      smile: 0.75,
    },
    {
      t: 0.54,
      right: { wrist: [-0.2, 1.35, 0.25], palm: [0.1, 0.1, 0.98], finger: [0, 0.98, 0.15], shape: "fist" },
      head: { pitch: -0.02, yaw: 0, roll: 0 },
    },
    {
      t: 0.82,
      right: { wrist: [-0.2, 1.27, 0.3], palm: [0.1, -0.55, 0.82], finger: [0, 0.65, 0.75], shape: "fist" },
      head: { pitch: 0.12, yaw: 0, roll: 0 },
    },
    {
      t: 1.0,
      right: { wrist: [-0.2, 1.33, 0.26], palm: [0.1, 0.05, 0.98], finger: [0, 0.96, 0.2], shape: "fist" },
      head: { pitch: 0.02, yaw: 0, roll: 0 },
    },
  ],
  no: [
    {
      t: 0.0,
      right: { wrist: [-0.16, 1.32, 0.28], palm: [0.65, 0, 0.75], finger: [0.2, 0.85, 0.45], shape: "no_open" },
      head: { pitch: 0, yaw: -0.1, roll: 0 },
      smile: 0.05,
    },
    {
      t: 0.32,
      right: { wrist: [-0.15, 1.31, 0.29], palm: [0.65, 0, 0.75], finger: [0.2, 0.82, 0.5], shape: "no_closed" },
      head: { pitch: 0, yaw: 0.12, roll: 0 },
    },
    {
      t: 0.6,
      right: { wrist: [-0.16, 1.32, 0.28], palm: [0.65, 0, 0.75], finger: [0.2, 0.85, 0.45], shape: "no_open" },
      head: { pitch: 0, yaw: -0.12, roll: 0 },
    },
    {
      t: 0.88,
      right: { wrist: [-0.15, 1.31, 0.29], palm: [0.65, 0, 0.75], finger: [0.2, 0.82, 0.5], shape: "no_closed" },
      head: { pitch: 0, yaw: 0.08, roll: 0 },
    },
    {
      t: 1.0,
      right: { wrist: [-0.15, 1.31, 0.29], palm: [0.65, 0, 0.75], finger: [0.2, 0.82, 0.5], shape: "no_closed" },
      head: { pitch: 0, yaw: 0, roll: 0 },
    },
  ],
  water: [
    {
      t: 0.0,
      right: { wrist: [-0.07, 1.4, 0.22], palm: [0.95, 0, 0.25], finger: [0, 0.98, 0.1], shape: "w_hand" },
      smile: 0.35,
    },
    {
      t: 0.32,
      right: { wrist: [-0.03, 1.44, 0.19], palm: [0.95, 0, 0.25], finger: [0, 0.98, 0.1], shape: "w_hand" },
    },
    {
      t: 0.58,
      right: { wrist: [-0.08, 1.41, 0.23], palm: [0.95, 0, 0.25], finger: [0, 0.98, 0.1], shape: "w_hand" },
    },
    {
      t: 0.85,
      right: { wrist: [-0.03, 1.44, 0.19], palm: [0.95, 0, 0.25], finger: [0, 0.98, 0.1], shape: "w_hand" },
    },
    {
      t: 1.0,
      right: { wrist: [-0.07, 1.41, 0.22], palm: [0.95, 0, 0.25], finger: [0, 0.98, 0.1], shape: "w_hand" },
    },
  ],
  i_love_you: [
    {
      t: 0.0,
      right: { wrist: [-0.18, 1.24, 0.24], palm: [0, 0, 1], finger: [0, 0.98, 0.15], shape: "ily" },
      head: { pitch: -0.02, yaw: 0, roll: 0.04 },
      smile: 0.95,
    },
    {
      t: 0.45,
      right: { wrist: [-0.16, 1.34, 0.33], palm: [0, 0.08, 0.99], finger: [0, 0.98, 0.18], shape: "ily" },
      head: { pitch: -0.04, yaw: 0, roll: 0.07 },
      smile: 1.0,
    },
    {
      t: 0.8,
      right: { wrist: [-0.17, 1.33, 0.31], palm: [0, 0.08, 0.99], finger: [0, 0.98, 0.18], shape: "ily" },
      head: { pitch: -0.03, yaw: 0, roll: 0.05 },
    },
    {
      t: 1.0,
      right: { wrist: [-0.18, 1.3, 0.28], palm: [0, 0.05, 0.99], finger: [0, 0.98, 0.16], shape: "ily" },
    },
  ],
  give_a_to_b: [
    {
      t: 0.0,
      right: { wrist: [-0.32, 1.2, 0.26], palm: [0.2, 0.95, 0.22], finger: [-0.25, 0.25, 0.93], shape: "cupped" },
      head: { pitch: 0.02, yaw: -0.25, roll: 0 },
      torsoYaw: -0.14,
      gazeX: -0.65,
      activeLoci: { left: true, right: false },
    },
    {
      t: 0.35,
      right: { wrist: [-0.18, 1.28, 0.34], palm: [0.1, 0.96, 0.25], finger: [0, 0.32, 0.94], shape: "cupped" },
      head: { pitch: 0, yaw: -0.05, roll: 0 },
      torsoYaw: -0.03,
      gazeX: -0.15,
      activeLoci: { left: true, right: true },
    },
    {
      t: 0.78,
      right: { wrist: [0.16, 1.21, 0.29], palm: [-0.2, 0.95, 0.22], finger: [0.35, 0.2, 0.91], shape: "open_flat" },
      head: { pitch: 0.02, yaw: 0.26, roll: 0 },
      torsoYaw: 0.16,
      gazeX: 0.65,
      activeLoci: { left: false, right: true },
    },
    {
      t: 1.0,
      right: { wrist: [0.18, 1.18, 0.28], palm: [-0.2, 0.95, 0.22], finger: [0.35, 0.2, 0.91], shape: "open_flat" },
      head: { pitch: 0.02, yaw: 0.22, roll: 0 },
      torsoYaw: 0.12,
      gazeX: 0.5,
      activeLoci: { left: true, right: true },
    },
  ],
  please: [
    {
      t: 0.0,
      right: { wrist: [-0.03, 1.24, 0.2], palm: [0, 0, -1], finger: [0.7, 0.7, 0], shape: "open_flat" },
      smile: 0.7,
    },
    {
      t: 0.33,
      right: { wrist: [0.02, 1.27, 0.2], palm: [0, 0, -1], finger: [0.7, 0.7, 0], shape: "open_flat" },
    },
    {
      t: 0.66,
      right: { wrist: [-0.03, 1.2, 0.2], palm: [0, 0, -1], finger: [0.7, 0.7, 0], shape: "open_flat" },
    },
    {
      t: 1.0,
      right: { wrist: [-0.01, 1.24, 0.2], palm: [0, 0, -1], finger: [0.7, 0.7, 0], shape: "open_flat" },
    },
  ],
  sorry: [
    {
      t: 0.0,
      right: { wrist: [-0.02, 1.23, 0.21], palm: [0, 0, -1], finger: [0.55, 0.82, 0], shape: "fist" },
      head: { pitch: 0.06, yaw: 0, roll: 0.09 },
      smile: 0.05,
    },
    {
      t: 0.35,
      right: { wrist: [0.03, 1.27, 0.21], palm: [0, 0, -1], finger: [0.55, 0.82, 0], shape: "fist" },
      head: { pitch: 0.07, yaw: 0, roll: 0.1 },
    },
    {
      t: 0.7,
      right: { wrist: [-0.04, 1.2, 0.21], palm: [0, 0, -1], finger: [0.55, 0.82, 0], shape: "fist" },
      head: { pitch: 0.06, yaw: 0, roll: 0.08 },
    },
    {
      t: 1.0,
      right: { wrist: [-0.01, 1.24, 0.21], palm: [0, 0, -1], finger: [0.55, 0.82, 0], shape: "fist" },
    },
  ],
  good_morning: [
    {
      t: 0.0,
      right: { wrist: [-0.04, 1.44, 0.2], palm: [0, 0.2, -0.98], finger: [0, 0.96, -0.2], shape: "open_flat" },
      left: { wrist: [0.1, 1.15, 0.24], palm: [0, -0.8, 0.6], finger: [-0.85, 0, 0.5], shape: "open_flat" },
      smile: 0.85,
    },
    {
      t: 0.42,
      right: { wrist: [-0.03, 1.18, 0.27], palm: [0, 0.95, 0.3], finger: [0.25, 0.2, 0.94], shape: "open_flat" },
      left: { wrist: [0.04, 1.14, 0.25], palm: [0, -0.85, 0.5], finger: [-0.95, 0, 0.25], shape: "open_flat" },
    },
    {
      t: 0.88,
      right: { wrist: [-0.14, 1.38, 0.26], palm: [0, 0.15, -0.98], finger: [0, 0.98, 0.1], shape: "open_flat" },
      left: { wrist: [0.05, 1.14, 0.25], palm: [0, -0.85, 0.5], finger: [-0.95, 0, 0.25], shape: "open_flat" },
    },
    {
      t: 1.0,
      right: { wrist: [-0.14, 1.36, 0.26], palm: [0, 0.15, -0.98], finger: [0, 0.98, 0.1], shape: "open_flat" },
    },
  ],
  good: [
    {
      t: 0.0,
      right: { wrist: [-0.03, 1.44, 0.2], palm: [0, 0.2, -0.98], finger: [0, 0.96, -0.2], shape: "open_flat" },
      left: { wrist: [0.08, 1.16, 0.25], palm: [0, 0.98, 0.2], finger: [-0.3, 0.2, 0.92], shape: "open_flat" },
      smile: 0.85,
    },
    {
      t: 0.65,
      right: { wrist: [-0.01, 1.21, 0.28], palm: [0, 0.95, 0.3], finger: [0.25, 0.18, 0.95], shape: "open_flat" },
      left: { wrist: [0.05, 1.16, 0.27], palm: [0, 0.98, 0.2], finger: [-0.3, 0.2, 0.92], shape: "open_flat" },
    },
    {
      t: 1.0,
      right: { wrist: [-0.01, 1.21, 0.28], palm: [0, 0.95, 0.3], finger: [0.25, 0.18, 0.95], shape: "open_flat" },
    },
  ],
  welcome: [
    {
      t: 0.0,
      right: { wrist: [-0.26, 1.24, 0.34], palm: [0.3, 0.92, 0.22], finger: [0.1, 0.2, 0.96], shape: "open_flat" },
      smile: 0.85,
    },
    {
      t: 0.7,
      right: { wrist: [-0.05, 1.22, 0.23], palm: [0.2, 0.9, -0.35], finger: [0.6, 0.2, 0.75], shape: "open_flat" },
      head: { pitch: 0.05, yaw: 0, roll: 0 },
    },
    {
      t: 1.0,
      right: { wrist: [-0.05, 1.22, 0.23], palm: [0.2, 0.9, -0.35], finger: [0.6, 0.2, 0.75], shape: "open_flat" },
    },
  ],
  name: [
    {
      t: 0.0,
      left: { wrist: [0.07, 1.23, 0.26], palm: [-0.6, 0.6, 0.5], finger: [-0.85, 0.35, 0.35], shape: "h_hand" },
      right: { wrist: [-0.09, 1.33, 0.27], palm: [0.6, -0.6, 0.5], finger: [0.85, 0.35, 0.35], shape: "h_hand" },
    },
    {
      t: 0.35,
      left: { wrist: [0.05, 1.23, 0.26], palm: [-0.6, 0.6, 0.5], finger: [-0.85, 0.35, 0.35], shape: "h_hand" },
      right: { wrist: [-0.03, 1.26, 0.27], palm: [0.6, -0.6, 0.5], finger: [0.85, 0.35, 0.35], shape: "h_hand" },
    },
    {
      t: 0.62,
      left: { wrist: [0.05, 1.23, 0.26], palm: [-0.6, 0.6, 0.5], finger: [-0.85, 0.35, 0.35], shape: "h_hand" },
      right: { wrist: [-0.08, 1.33, 0.27], palm: [0.6, -0.6, 0.5], finger: [0.85, 0.35, 0.35], shape: "h_hand" },
    },
    {
      t: 0.9,
      left: { wrist: [0.05, 1.23, 0.26], palm: [-0.6, 0.6, 0.5], finger: [-0.85, 0.35, 0.35], shape: "h_hand" },
      right: { wrist: [-0.03, 1.26, 0.27], palm: [0.6, -0.6, 0.5], finger: [0.85, 0.35, 0.35], shape: "h_hand" },
    },
  ],
  what: [
    {
      t: 0.0,
      right: { wrist: [-0.22, 1.24, 0.28], palm: [0, 0.95, 0.3], finger: [-0.1, 0.25, 0.95], shape: "open_spread" },
      left: { wrist: [0.22, 1.24, 0.28], palm: [0, 0.95, 0.3], finger: [0.1, 0.25, 0.95], shape: "open_spread" },
      browFurrow: 0.9,
      head: { pitch: 0.04, yaw: 0, roll: 0.08 },
      smile: 0.0,
    },
    {
      t: 0.35,
      right: { wrist: [-0.28, 1.24, 0.29], palm: [0.15, 0.94, 0.3], finger: [-0.25, 0.25, 0.93], shape: "open_spread" },
      left: { wrist: [0.28, 1.24, 0.29], palm: [-0.15, 0.94, 0.3], finger: [0.25, 0.25, 0.93], shape: "open_spread" },
      browFurrow: 1.0,
      head: { pitch: 0.05, yaw: 0.05, roll: 0.1 },
    },
    {
      t: 0.72,
      right: { wrist: [-0.2, 1.25, 0.29], palm: [-0.1, 0.94, 0.3], finger: [0.05, 0.25, 0.95], shape: "open_spread" },
      left: { wrist: [0.2, 1.25, 0.29], palm: [0.1, 0.94, 0.3], finger: [-0.05, 0.25, 0.95], shape: "open_spread" },
      browFurrow: 1.0,
      head: { pitch: 0.04, yaw: -0.05, roll: 0.08 },
    },
    {
      t: 1.0,
      right: { wrist: [-0.25, 1.24, 0.28], palm: [0, 0.95, 0.3], finger: [-0.1, 0.25, 0.95], shape: "open_spread" },
      left: { wrist: [0.25, 1.24, 0.28], palm: [0, 0.95, 0.3], finger: [0.1, 0.25, 0.95], shape: "open_spread" },
      browFurrow: 0.85,
    },
  ],
  where: [
    {
      t: 0.0,
      right: { wrist: [-0.2, 1.34, 0.28], palm: [0, 0, 1], finger: [0, 0.98, 0.15], shape: "point" },
      browFurrow: 0.9,
      head: { pitch: 0.04, yaw: 0, roll: 0.08 },
      smile: 0.0,
    },
    {
      t: 0.33,
      right: { wrist: [-0.15, 1.34, 0.28], palm: [0.15, 0, 0.98], finger: [0.28, 0.94, 0.15], shape: "point" },
      browFurrow: 1.0,
    },
    {
      t: 0.68,
      right: { wrist: [-0.25, 1.34, 0.28], palm: [-0.15, 0, 0.98], finger: [-0.28, 0.94, 0.15], shape: "point" },
      browFurrow: 1.0,
    },
    {
      t: 1.0,
      right: { wrist: [-0.2, 1.34, 0.28], palm: [0, 0, 1], finger: [0, 0.98, 0.15], shape: "point" },
      browFurrow: 0.85,
    },
  ],
  who: [
    {
      t: 0.0,
      right: { wrist: [-0.04, 1.38, 0.23], palm: [0, 0.1, -0.98], finger: [0, 0.98, 0.15], shape: "point" },
      browFurrow: 0.9,
      head: { pitch: 0.04, yaw: 0, roll: 0.07 },
      smile: 0.0,
    },
    {
      t: 0.5,
      right: { wrist: [-0.08, 1.4, 0.24], palm: [0.15, 0.1, -0.98], finger: [-0.1, 0.98, 0.15], shape: "point" },
      browFurrow: 1.0,
    },
    {
      t: 1.0,
      right: { wrist: [-0.04, 1.38, 0.23], palm: [0, 0.1, -0.98], finger: [0, 0.98, 0.15], shape: "point" },
    },
  ],
  want: [
    {
      t: 0.0,
      right: { wrist: [-0.2, 1.24, 0.34], palm: [0, 0.92, 0.38], finger: [0, 0.2, 0.96], shape: "claw" },
      left: { wrist: [0.2, 1.24, 0.34], palm: [0, 0.92, 0.38], finger: [0, 0.2, 0.96], shape: "claw" },
    },
    {
      t: 0.7,
      right: { wrist: [-0.16, 1.21, 0.22], palm: [0, 0.95, 0.3], finger: [0, 0.25, 0.95], shape: "claw" },
      left: { wrist: [0.16, 1.21, 0.22], palm: [0, 0.95, 0.3], finger: [0, 0.25, 0.95], shape: "claw" },
    },
    {
      t: 1.0,
      right: { wrist: [-0.16, 1.21, 0.22], palm: [0, 0.95, 0.3], finger: [0, 0.25, 0.95], shape: "claw" },
      left: { wrist: [0.16, 1.21, 0.22], palm: [0, 0.95, 0.3], finger: [0, 0.25, 0.95], shape: "claw" },
    },
  ],
  stop: [
    {
      t: 0.0,
      left: { wrist: [0.05, 1.19, 0.28], palm: [0, 0.98, 0.2], finger: [-0.3, 0.15, 0.94], shape: "open_flat" },
      right: { wrist: [-0.04, 1.4, 0.28], palm: [0.95, 0, 0.25], finger: [0.15, 0.3, 0.94], shape: "open_flat" },
      smile: 0.05,
    },
    {
      t: 0.55,
      left: { wrist: [0.05, 1.19, 0.28], palm: [0, 0.98, 0.2], finger: [-0.3, 0.15, 0.94], shape: "open_flat" },
      right: { wrist: [0.02, 1.24, 0.29], palm: [0.95, 0, 0.25], finger: [0.15, 0.15, 0.96], shape: "open_flat" },
    },
    {
      t: 1.0,
      left: { wrist: [0.05, 1.19, 0.28], palm: [0, 0.98, 0.2], finger: [-0.3, 0.15, 0.94], shape: "open_flat" },
      right: { wrist: [0.02, 1.24, 0.29], palm: [0.95, 0, 0.25], finger: [0.15, 0.15, 0.96], shape: "open_flat" },
    },
  ],
  your: [
    {
      t: 0.0,
      right: { wrist: [-0.1, 1.24, 0.24], palm: [0, 0.1, 0.99], finger: [0, 0.96, 0.25], shape: "open_flat" },
    },
    {
      t: 0.65,
      right: { wrist: [-0.08, 1.28, 0.37], palm: [0, 0.08, 0.99], finger: [0, 0.96, 0.25], shape: "open_flat" },
    },
    {
      t: 1.0,
      right: { wrist: [-0.08, 1.28, 0.35], palm: [0, 0.08, 0.99], finger: [0, 0.96, 0.25], shape: "open_flat" },
    },
  ],
  you: [
    {
      t: 0.0,
      right: { wrist: [-0.1, 1.24, 0.24], palm: [0.8, 0.2, 0.55], finger: [0, 0.25, 0.96], shape: "point" },
    },
    {
      t: 0.65,
      right: { wrist: [-0.08, 1.28, 0.37], palm: [0.8, 0.2, 0.55], finger: [0, 0.2, 0.98], shape: "point" },
    },
    {
      t: 1.0,
      right: { wrist: [-0.08, 1.28, 0.35], palm: [0.8, 0.2, 0.55], finger: [0, 0.2, 0.98], shape: "point" },
    },
  ],
  my: [
    {
      t: 0.0,
      right: { wrist: [-0.06, 1.24, 0.26], palm: [0, 0, -1], finger: [0.45, 0.88, 0], shape: "open_flat" },
    },
    {
      t: 0.6,
      right: { wrist: [-0.02, 1.24, 0.2], palm: [0, 0, -1], finger: [0.45, 0.88, 0], shape: "open_flat" },
    },
    {
      t: 1.0,
      right: { wrist: [-0.02, 1.24, 0.2], palm: [0, 0, -1], finger: [0.45, 0.88, 0], shape: "open_flat" },
    },
  ],
  me: [
    {
      t: 0.0,
      right: { wrist: [-0.08, 1.24, 0.28], palm: [0, 0.2, -0.98], finger: [0.15, 0.15, -0.97], shape: "point" },
    },
    {
      t: 0.6,
      right: { wrist: [-0.02, 1.24, 0.22], palm: [0, 0.2, -0.98], finger: [0.1, 0.1, -0.98], shape: "point" },
    },
    {
      t: 1.0,
      right: { wrist: [-0.02, 1.24, 0.22], palm: [0, 0.2, -0.98], finger: [0.1, 0.1, -0.98], shape: "point" },
    },
  ],
  change: [
    {
      t: 0.0,
      left: { wrist: [0.07, 1.21, 0.27], palm: [0, 0.3, 0.95], finger: [0, 0.9, 0.35], shape: "cupped" },
      right: { wrist: [-0.07, 1.24, 0.29], palm: [0, -0.3, 0.95], finger: [0, 0.9, 0.35], shape: "cupped" },
    },
    {
      t: 0.55,
      left: { wrist: [-0.05, 1.24, 0.29], palm: [0, 0.4, 0.9], finger: [0, 0.9, 0.35], shape: "cupped" },
      right: { wrist: [0.05, 1.21, 0.27], palm: [0, -0.4, 0.9], finger: [0, 0.9, 0.35], shape: "cupped" },
    },
    {
      t: 1.0,
      left: { wrist: [0.06, 1.22, 0.28], palm: [0, 0.3, 0.95], finger: [0, 0.9, 0.35], shape: "cupped" },
      right: { wrist: [-0.06, 1.23, 0.28], palm: [0, -0.3, 0.95], finger: [0, 0.9, 0.35], shape: "cupped" },
    },
  ],
  correct: [
    {
      t: 0.0,
      left: { wrist: [0.06, 1.18, 0.28], palm: [0, 1, 0.15], finger: [-0.2, 0.1, 0.96], shape: "open_flat" },
      right: { wrist: [-0.1, 1.32, 0.3], palm: [0, -0.95, 0.2], finger: [0.2, 0.1, 0.96], shape: "f_hand" },
    },
    {
      t: 0.6,
      left: { wrist: [0.05, 1.19, 0.29], palm: [0, 1, 0.15], finger: [-0.2, 0.1, 0.96], shape: "open_flat" },
      right: { wrist: [0.02, 1.22, 0.3], palm: [0, -0.95, 0.2], finger: [0.2, 0.1, 0.96], shape: "f_hand" },
    },
    {
      t: 1.0,
      left: { wrist: [0.05, 1.19, 0.29], palm: [0, 1, 0.15], finger: [-0.2, 0.1, 0.96], shape: "open_flat" },
      right: { wrist: [0.02, 1.24, 0.3], palm: [0, -0.95, 0.2], finger: [0.2, 0.1, 0.96], shape: "f_hand" },
    },
  ],
  goodbye: [
    {
      t: 0.0,
      right: { wrist: [-0.22, 1.44, 0.24], palm: [0, 0.15, 0.98], finger: [0, 0.98, 0.12], shape: "open_flat" },
      smile: 0.6,
    },
    {
      t: 0.35,
      right: { wrist: [-0.16, 1.46, 0.26], palm: [0.18, 0.15, 0.97], finger: [0.2, 0.96, 0.12], shape: "open_flat" },
      smile: 0.65,
    },
    {
      t: 0.7,
      right: { wrist: [-0.25, 1.45, 0.25], palm: [-0.18, 0.15, 0.97], finger: [-0.2, 0.96, 0.12], shape: "open_flat" },
      smile: 0.65,
    },
    {
      t: 1.0,
      right: { wrist: [-0.2, 1.44, 0.24], palm: [0, 0.15, 0.98], finger: [0, 0.98, 0.12], shape: "open_flat" },
      smile: 0.6,
    },
  ],
  home: [
    {
      t: 0.0,
      left: { wrist: [0.14, 1.22, 0.26], palm: [-0.85, 0.3, 0.42], finger: [-0.45, 0.88, 0.1], shape: "open_flat" },
      right: { wrist: [-0.14, 1.22, 0.26], palm: [0.85, 0.3, 0.42], finger: [0.45, 0.88, 0.1], shape: "open_flat" },
    },
    {
      t: 0.6,
      left: { wrist: [0.06, 1.28, 0.28], palm: [-0.92, 0.2, 0.32], finger: [-0.4, 0.91, 0.08], shape: "open_flat" },
      right: { wrist: [-0.06, 1.28, 0.28], palm: [0.92, 0.2, 0.32], finger: [0.4, 0.91, 0.08], shape: "open_flat" },
    },
    {
      t: 1.0,
      left: { wrist: [0.06, 1.28, 0.28], palm: [-0.92, 0.2, 0.32], finger: [-0.4, 0.91, 0.08], shape: "open_flat" },
      right: { wrist: [-0.06, 1.28, 0.28], palm: [0.92, 0.2, 0.32], finger: [0.4, 0.91, 0.08], shape: "open_flat" },
    },
  ],
  hug: [
    {
      t: 0.0,
      left: { wrist: [0.18, 1.22, 0.25], palm: [0, 0, -1], finger: [-0.3, 0.94, 0.1], shape: "fist" },
      right: { wrist: [-0.18, 1.22, 0.25], palm: [0, 0, -1], finger: [0.3, 0.94, 0.1], shape: "fist" },
      smile: 0.55,
    },
    {
      t: 0.65,
      left: { wrist: [-0.06, 1.26, 0.2], palm: [0, 0, -1], finger: [-0.6, 0.78, 0.1], shape: "fist" },
      right: { wrist: [0.06, 1.26, 0.22], palm: [0, 0, -1], finger: [0.6, 0.78, 0.1], shape: "fist" },
      smile: 0.7,
    },
    {
      t: 1.0,
      left: { wrist: [-0.05, 1.26, 0.2], palm: [0, 0, -1], finger: [-0.55, 0.82, 0.1], shape: "fist" },
      right: { wrist: [0.05, 1.26, 0.22], palm: [0, 0, -1], finger: [0.55, 0.82, 0.1], shape: "fist" },
      smile: 0.65,
    },
  ],
  meet: [
    {
      t: 0.0,
      left: { wrist: [0.16, 1.24, 0.28], palm: [-0.95, 0, 0.28], finger: [0, 0.96, 0.22], shape: "point" },
      right: { wrist: [-0.16, 1.24, 0.28], palm: [0.95, 0, 0.28], finger: [0, 0.96, 0.22], shape: "point" },
    },
    {
      t: 0.65,
      left: { wrist: [0.04, 1.26, 0.3], palm: [-0.95, 0, 0.25], finger: [0, 0.97, 0.2], shape: "point" },
      right: { wrist: [-0.04, 1.26, 0.3], palm: [0.95, 0, 0.25], finger: [0, 0.97, 0.2], shape: "point" },
    },
    {
      t: 1.0,
      left: { wrist: [0.04, 1.26, 0.3], palm: [-0.95, 0, 0.25], finger: [0, 0.97, 0.2], shape: "point" },
      right: { wrist: [-0.04, 1.26, 0.3], palm: [0.95, 0, 0.25], finger: [0, 0.97, 0.2], shape: "point" },
    },
  ],
  touch: [
    {
      t: 0.0,
      left: { wrist: [0.06, 1.19, 0.28], palm: [0, -0.98, 0.2], finger: [-0.25, 0.1, 0.96], shape: "open_flat" },
      right: { wrist: [-0.1, 1.32, 0.28], palm: [0, -0.96, 0.2], finger: [0.25, 0.1, 0.96], shape: "open_flat" },
    },
    {
      t: 0.6,
      left: { wrist: [0.05, 1.19, 0.28], palm: [0, -0.98, 0.2], finger: [-0.25, 0.1, 0.96], shape: "open_flat" },
      right: { wrist: [0.02, 1.23, 0.28], palm: [0, -0.96, 0.2], finger: [0.25, 0.1, 0.96], shape: "open_flat" },
    },
    {
      t: 1.0,
      left: { wrist: [0.05, 1.19, 0.28], palm: [0, -0.98, 0.2], finger: [-0.25, 0.1, 0.96], shape: "open_flat" },
      right: { wrist: [-0.02, 1.26, 0.28], palm: [0, -0.96, 0.2], finger: [0.25, 0.1, 0.96], shape: "open_flat" },
    },
  ],
  eat: [
    {
      t: 0.0,
      right: { wrist: [-0.1, 1.32, 0.26], palm: [0, 0.2, -0.96], finger: [0.05, 0.9, -0.35], shape: "o_hand" },
    },
    {
      t: 0.35,
      right: { wrist: [-0.03, 1.44, 0.2], palm: [0, 0.15, -0.98], finger: [0.05, 0.88, -0.42], shape: "o_hand" },
      head: { pitch: 0.06, yaw: 0, roll: 0 },
    },
    {
      t: 0.65,
      right: { wrist: [-0.08, 1.37, 0.24], palm: [0, 0.2, -0.96], finger: [0.05, 0.9, -0.35], shape: "o_hand" },
    },
    {
      t: 1.0,
      right: { wrist: [-0.03, 1.44, 0.2], palm: [0, 0.15, -0.98], finger: [0.05, 0.88, -0.42], shape: "o_hand" },
    },
  ],
  drink: [
    {
      t: 0.0,
      right: { wrist: [-0.12, 1.28, 0.28], palm: [0.92, 0, -0.35], finger: [0.25, 0.75, 0.6], shape: "cupped" },
    },
    {
      t: 0.55,
      right: { wrist: [-0.05, 1.41, 0.22], palm: [0.85, 0.35, -0.35], finger: [0.2, 0.4, 0.88], shape: "cupped" },
      head: { pitch: -0.08, yaw: 0, roll: 0 },
    },
    {
      t: 1.0,
      right: { wrist: [-0.09, 1.35, 0.26], palm: [0.9, 0.15, -0.35], finger: [0.25, 0.65, 0.7], shape: "cupped" },
    },
  ],
  more: [
    {
      t: 0.0,
      left: { wrist: [0.16, 1.25, 0.28], palm: [-0.95, 0, -0.25], finger: [-0.55, 0.55, 0.62], shape: "o_hand" },
      right: { wrist: [-0.16, 1.25, 0.28], palm: [0.95, 0, -0.25], finger: [0.55, 0.55, 0.62], shape: "o_hand" },
    },
    {
      t: 0.45,
      left: { wrist: [0.04, 1.26, 0.31], palm: [-0.96, 0, -0.2], finger: [-0.75, 0.4, 0.5], shape: "o_hand" },
      right: { wrist: [-0.04, 1.26, 0.31], palm: [0.96, 0, -0.2], finger: [0.75, 0.4, 0.5], shape: "o_hand" },
    },
    {
      t: 0.72,
      left: { wrist: [0.1, 1.25, 0.29], palm: [-0.95, 0, -0.25], finger: [-0.6, 0.5, 0.6], shape: "o_hand" },
      right: { wrist: [-0.1, 1.25, 0.29], palm: [0.95, 0, -0.25], finger: [0.6, 0.5, 0.6], shape: "o_hand" },
    },
    {
      t: 1.0,
      left: { wrist: [0.04, 1.26, 0.31], palm: [-0.96, 0, -0.2], finger: [-0.75, 0.4, 0.5], shape: "o_hand" },
      right: { wrist: [-0.04, 1.26, 0.31], palm: [0.96, 0, -0.2], finger: [0.75, 0.4, 0.5], shape: "o_hand" },
    },
  ],
  finish: [
    {
      t: 0.0,
      left: { wrist: [0.12, 1.26, 0.25], palm: [0, 0, -1], finger: [0, 0.96, 0.2], shape: "open_spread" },
      right: { wrist: [-0.12, 1.26, 0.25], palm: [0, 0, -1], finger: [0, 0.96, 0.2], shape: "open_spread" },
    },
    {
      t: 0.55,
      left: { wrist: [0.24, 1.24, 0.32], palm: [0.35, -0.1, 0.92], finger: [0.25, 0.9, 0.3], shape: "open_spread" },
      right: { wrist: [-0.24, 1.24, 0.32], palm: [-0.35, -0.1, 0.92], finger: [-0.25, 0.9, 0.3], shape: "open_spread" },
      head: { pitch: 0.06, yaw: 0, roll: 0 },
    },
    {
      t: 1.0,
      left: { wrist: [0.22, 1.24, 0.3], palm: [0.2, 0, 0.97], finger: [0.15, 0.92, 0.32], shape: "open_spread" },
      right: { wrist: [-0.22, 1.24, 0.3], palm: [-0.2, 0, 0.97], finger: [-0.15, 0.92, 0.32], shape: "open_spread" },
    },
  ],
  wait: [
    {
      t: 0.0,
      left: { wrist: [0.16, 1.22, 0.3], palm: [-0.15, 0.85, 0.48], finger: [0, 0.35, 0.93], shape: "open_spread" },
      right: { wrist: [-0.16, 1.22, 0.3], palm: [0.15, 0.85, 0.48], finger: [0, 0.35, 0.93], shape: "open_spread" },
    },
    {
      t: 0.5,
      left: { wrist: [0.18, 1.24, 0.32], palm: [-0.1, 0.88, 0.45], finger: [0, 0.42, 0.9], shape: "open_spread" },
      right: { wrist: [-0.18, 1.2, 0.32], palm: [0.1, 0.88, 0.45], finger: [0, 0.3, 0.95], shape: "open_spread" },
    },
    {
      t: 1.0,
      left: { wrist: [0.16, 1.22, 0.31], palm: [-0.15, 0.85, 0.48], finger: [0, 0.36, 0.93], shape: "open_spread" },
      right: { wrist: [-0.16, 1.23, 0.31], palm: [0.15, 0.85, 0.48], finger: [0, 0.4, 0.91], shape: "open_spread" },
    },
  ],
  sleep: [
    {
      t: 0.0,
      right: { wrist: [-0.04, 1.5, 0.22], palm: [0, 0, -1], finger: [0, 0.96, 0.2], shape: "open_spread" },
    },
    {
      t: 0.65,
      right: { wrist: [-0.03, 1.38, 0.22], palm: [0, 0.1, -0.99], finger: [0, 0.9, -0.35], shape: "cupped" },
      head: { pitch: 0.1, yaw: 0, roll: 0.12 },
      smile: 0.3,
    },
    {
      t: 1.0,
      right: { wrist: [-0.03, 1.36, 0.22], palm: [0, 0.15, -0.98], finger: [0, 0.88, -0.4], shape: "o_hand" },
      head: { pitch: 0.12, yaw: 0, roll: 0.15 },
    },
  ],
  pain: [
    {
      t: 0.0,
      left: { wrist: [0.18, 1.26, 0.28], palm: [0, 0, -1], finger: [-0.95, 0.1, 0.2], shape: "point" },
      right: { wrist: [-0.18, 1.26, 0.28], palm: [0, 0, -1], finger: [0.95, 0.1, 0.2], shape: "point" },
      browFurrow: 0.85,
      smile: 0.0,
    },
    {
      t: 0.45,
      left: { wrist: [0.09, 1.26, 0.3], palm: [0, -0.1, -0.98], finger: [-0.98, 0.05, 0.1], shape: "point" },
      right: { wrist: [-0.09, 1.26, 0.3], palm: [0, -0.1, -0.98], finger: [0.98, 0.05, 0.1], shape: "point" },
      browFurrow: 0.95,
      smile: 0.0,
    },
    {
      t: 0.75,
      left: { wrist: [0.16, 1.26, 0.28], palm: [0, 0, -1], finger: [-0.95, 0.1, 0.2], shape: "point" },
      right: { wrist: [-0.16, 1.26, 0.28], palm: [0, 0, -1], finger: [0.95, 0.1, 0.2], shape: "point" },
      browFurrow: 0.85,
      smile: 0.0,
    },
    {
      t: 1.0,
      left: { wrist: [0.09, 1.26, 0.3], palm: [0, -0.1, -0.98], finger: [-0.98, 0.05, 0.1], shape: "point" },
      right: { wrist: [-0.09, 1.26, 0.3], palm: [0, -0.1, -0.98], finger: [0.98, 0.05, 0.1], shape: "point" },
      browFurrow: 0.85,
      smile: 0.0,
    },
  ],
  bathroom: [
    {
      t: 0.0,
      right: { wrist: [-0.22, 1.36, 0.26], palm: [0.1, 0, 0.98], finger: [0, 1, 0], shape: "fist" },
    },
    {
      t: 0.35,
      right: { wrist: [-0.26, 1.36, 0.26], palm: [0.35, 0, 0.92], finger: [-0.2, 0.96, 0.15], shape: "fist" },
    },
    {
      t: 0.7,
      right: { wrist: [-0.19, 1.36, 0.26], palm: [-0.25, 0, 0.95], finger: [0.2, 0.96, 0.15], shape: "fist" },
    },
    {
      t: 1.0,
      right: { wrist: [-0.22, 1.36, 0.26], palm: [0.1, 0, 0.98], finger: [0, 1, 0], shape: "fist" },
    },
  ],
  book: [
    {
      t: 0.0,
      left: { wrist: [0.03, 1.22, 0.3], palm: [-0.98, 0.1, 0.1], finger: [0, 0.4, 0.91], shape: "open_flat" },
      right: { wrist: [-0.03, 1.22, 0.3], palm: [0.98, 0.1, 0.1], finger: [0, 0.4, 0.91], shape: "open_flat" },
    },
    {
      t: 0.65,
      left: { wrist: [0.1, 1.22, 0.32], palm: [-0.2, 0.92, 0.32], finger: [0.2, 0.25, 0.94], shape: "open_flat" },
      right: { wrist: [-0.1, 1.22, 0.32], palm: [0.2, 0.92, 0.32], finger: [-0.2, 0.25, 0.94], shape: "open_flat" },
    },
    {
      t: 1.0,
      left: { wrist: [0.1, 1.22, 0.32], palm: [-0.2, 0.92, 0.32], finger: [0.2, 0.25, 0.94], shape: "open_flat" },
      right: { wrist: [-0.1, 1.22, 0.32], palm: [0.2, 0.92, 0.32], finger: [-0.2, 0.25, 0.94], shape: "open_flat" },
    },
  ],
  mom: [
    {
      t: 0.0,
      right: { wrist: [-0.06, 1.38, 0.24], palm: [0.92, 0, 0.32], finger: [0, 0.98, 0.1], shape: "open_spread" },
      smile: 0.85,
    },
    {
      t: 0.5,
      right: { wrist: [-0.03, 1.43, 0.2], palm: [0.92, 0, 0.32], finger: [0, 0.98, 0.1], shape: "open_spread" },
      smile: 0.95,
    },
    {
      t: 1.0,
      right: { wrist: [-0.03, 1.43, 0.2], palm: [0.92, 0, 0.32], finger: [0, 0.98, 0.1], shape: "open_spread" },
      smile: 0.85,
    },
  ],
  dad: [
    {
      t: 0.0,
      right: { wrist: [-0.08, 1.5, 0.24], palm: [0.92, 0, 0.32], finger: [0, 0.98, 0.1], shape: "open_spread" },
      smile: 0.85,
    },
    {
      t: 0.5,
      right: { wrist: [-0.04, 1.55, 0.2], palm: [0.92, 0, 0.32], finger: [0, 0.98, 0.1], shape: "open_spread" },
      smile: 0.95,
    },
    {
      t: 1.0,
      right: { wrist: [-0.04, 1.55, 0.2], palm: [0.92, 0, 0.32], finger: [0, 0.98, 0.1], shape: "open_spread" },
      smile: 0.85,
    },
  ],
  milk: [
    {
      t: 0.0,
      right: { wrist: [-0.1, 1.24, 0.28], palm: [0.92, 0, 0.35], finger: [0, 1, 0], shape: "cupped" },
    },
    {
      t: 0.38,
      right: { wrist: [-0.1, 1.24, 0.28], palm: [0.92, 0, 0.35], finger: [0, 1, 0], shape: "fist" },
    },
    {
      t: 0.72,
      right: { wrist: [-0.1, 1.24, 0.28], palm: [0.92, 0, 0.35], finger: [0, 1, 0], shape: "cupped" },
    },
    {
      t: 1.0,
      right: { wrist: [-0.1, 1.24, 0.28], palm: [0.92, 0, 0.35], finger: [0, 1, 0], shape: "fist" },
    },
  ],
  come: [
    {
      t: 0.0,
      right: { wrist: [-0.18, 1.28, 0.38], palm: [0.2, 0.4, -0.88], finger: [0, 0.7, 0.7], shape: "point" },
    },
    {
      t: 0.65,
      right: { wrist: [-0.1, 1.28, 0.22], palm: [0.1, 0.2, -0.96], finger: [0, 0.92, -0.25], shape: "point" },
    },
    {
      t: 1.0,
      right: { wrist: [-0.1, 1.28, 0.22], palm: [0.1, 0.2, -0.96], finger: [0, 0.92, -0.25], shape: "point" },
    },
  ],
  go: [
    {
      t: 0.0,
      right: { wrist: [-0.12, 1.26, 0.24], palm: [0.4, 0, -0.9], finger: [0, 0.95, 0.2], shape: "point" },
    },
    {
      t: 0.65,
      right: { wrist: [-0.18, 1.26, 0.4], palm: [0.1, -0.5, 0.85], finger: [0, 0.2, 0.97], shape: "point" },
    },
    {
      t: 1.0,
      right: { wrist: [-0.18, 1.26, 0.4], palm: [0.1, -0.5, 0.85], finger: [0, 0.2, 0.97], shape: "point" },
    },
  ],
  when: [
    {
      t: 0.0,
      left: { wrist: [0.06, 1.24, 0.28], palm: [-0.85, 0, 0.5], finger: [0, 1, 0], shape: "point" },
      right: { wrist: [-0.12, 1.32, 0.28], palm: [0.7, 0, 0.7], finger: [0.25, 0.92, 0.25], shape: "point" },
      browFurrow: 0.85,
    },
    {
      t: 0.5,
      left: { wrist: [0.06, 1.24, 0.28], palm: [-0.85, 0, 0.5], finger: [0, 1, 0], shape: "point" },
      right: { wrist: [0.02, 1.36, 0.32], palm: [0.2, -0.3, 0.92], finger: [0.4, 0.85, 0.3], shape: "point" },
      browFurrow: 0.95,
    },
    {
      t: 1.0,
      left: { wrist: [0.06, 1.24, 0.28], palm: [-0.85, 0, 0.5], finger: [0, 1, 0], shape: "point" },
      right: { wrist: [0.04, 1.33, 0.28], palm: [0.6, -0.2, 0.75], finger: [0.55, 0.78, 0.25], shape: "point" },
      browFurrow: 0.9,
    },
  ],
  how: [
    {
      t: 0.0,
      left: { wrist: [0.05, 1.25, 0.26], palm: [-0.3, -0.35, -0.88], finger: [-0.5, -0.5, 0.7], shape: "cupped" },
      right: { wrist: [-0.05, 1.25, 0.26], palm: [0.3, -0.35, -0.88], finger: [0.5, -0.5, 0.7], shape: "cupped" },
      browFurrow: 0.85,
    },
    {
      t: 0.65,
      left: { wrist: [0.1, 1.23, 0.32], palm: [-0.15, 0.92, 0.32], finger: [-0.15, 0.35, 0.92], shape: "open_flat" },
      right: { wrist: [-0.1, 1.23, 0.32], palm: [0.15, 0.92, 0.32], finger: [0.15, 0.35, 0.92], shape: "open_flat" },
      browFurrow: 0.95,
    },
    {
      t: 1.0,
      left: { wrist: [0.1, 1.23, 0.32], palm: [-0.15, 0.92, 0.32], finger: [-0.15, 0.35, 0.92], shape: "open_flat" },
      right: { wrist: [-0.1, 1.23, 0.32], palm: [0.15, 0.92, 0.32], finger: [0.15, 0.35, 0.92], shape: "open_flat" },
      browFurrow: 0.9,
    },
  ],
  danger: [
    {
      t: 0.0,
      left: { wrist: [0.18, 1.24, 0.3], palm: [0, 0, 1], finger: [0, 0.96, 0.2], shape: "open_spread" },
      right: { wrist: [-0.16, 1.38, 0.24], palm: [0.3, 0, 0.95], finger: [0, 0.98, 0.1], shape: "point" },
      browFurrow: 0.9,
      smile: 0.0,
    },
    {
      t: 0.55,
      left: { wrist: [0.22, 1.28, 0.36], palm: [0, 0, 1], finger: [0, 0.96, 0.2], shape: "open_spread" },
      right: { wrist: [-0.12, 1.44, 0.22], palm: [0.3, 0, 0.95], finger: [0, 0.98, 0.1], shape: "point" },
      browFurrow: 0.95,
      head: { pitch: -0.05, yaw: 0.06, roll: 0 },
      smile: 0.0,
    },
    {
      t: 1.0,
      left: { wrist: [0.2, 1.26, 0.34], palm: [0, 0, 1], finger: [0, 0.96, 0.2], shape: "open_spread" },
      right: { wrist: [-0.14, 1.42, 0.24], palm: [0.3, 0, 0.95], finger: [0, 0.98, 0.1], shape: "point" },
      browFurrow: 0.9,
      smile: 0.0,
    },
  ],
  wash: [
    {
      t: 0.0,
      left: { wrist: [0.04, 1.22, 0.28], palm: [-0.2, 0.9, 0.35], finger: [-0.7, 0.2, 0.65], shape: "fist" },
      right: { wrist: [-0.04, 1.26, 0.28], palm: [0.2, -0.9, 0.35], finger: [0.7, 0.2, 0.65], shape: "fist" },
    },
    {
      t: 0.5,
      left: { wrist: [0.02, 1.2, 0.3], palm: [-0.2, 0.9, 0.35], finger: [-0.7, 0.2, 0.65], shape: "fist" },
      right: { wrist: [-0.02, 1.28, 0.26], palm: [0.2, -0.9, 0.35], finger: [0.7, 0.2, 0.65], shape: "fist" },
    },
    {
      t: 1.0,
      left: { wrist: [0.04, 1.22, 0.28], palm: [-0.2, 0.9, 0.35], finger: [-0.7, 0.2, 0.65], shape: "fist" },
      right: { wrist: [-0.04, 1.26, 0.28], palm: [0.2, -0.9, 0.35], finger: [0.7, 0.2, 0.65], shape: "fist" },
    },
  ],
  work: [
    {
      t: 0.0,
      left: { wrist: [0.05, 1.2, 0.28], palm: [-0.2, -0.85, 0.45], finger: [-0.75, 0.2, 0.6], shape: "fist" },
      right: { wrist: [-0.06, 1.32, 0.28], palm: [0.2, -0.85, 0.45], finger: [0.75, 0.2, 0.6], shape: "fist" },
    },
    {
      t: 0.45,
      left: { wrist: [0.05, 1.2, 0.28], palm: [-0.2, -0.85, 0.45], finger: [-0.75, 0.2, 0.6], shape: "fist" },
      right: { wrist: [0.03, 1.24, 0.28], palm: [0.2, -0.85, 0.45], finger: [0.75, 0.2, 0.6], shape: "fist" },
    },
    {
      t: 0.72,
      left: { wrist: [0.05, 1.2, 0.28], palm: [-0.2, -0.85, 0.45], finger: [-0.75, 0.2, 0.6], shape: "fist" },
      right: { wrist: [-0.04, 1.3, 0.28], palm: [0.2, -0.85, 0.45], finger: [0.75, 0.2, 0.6], shape: "fist" },
    },
    {
      t: 1.0,
      left: { wrist: [0.05, 1.2, 0.28], palm: [-0.2, -0.85, 0.45], finger: [-0.75, 0.2, 0.6], shape: "fist" },
      right: { wrist: [0.03, 1.24, 0.28], palm: [0.2, -0.85, 0.45], finger: [0.75, 0.2, 0.6], shape: "fist" },
    },
  ],
  unknown_gloss: [
    {
      t: 0.0,
      right: { wrist: [-0.18, 1.26, 0.28], palm: [0.2, 0.9, 0.38], finger: [0, 0.35, 0.93], shape: "open_flat" },
    },
    {
      t: 0.5,
      right: { wrist: [-0.15, 1.3, 0.32], palm: [0.1, 0.9, 0.42], finger: [0, 0.35, 0.93], shape: "open_spread" },
    },
    {
      t: 1.0,
      right: { wrist: [-0.16, 1.28, 0.3], palm: [0.15, 0.9, 0.4], finger: [0, 0.35, 0.93], shape: "open_flat" },
    },
  ],
};

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpArr(a, b, t) {
  return a.map((v, idx) => lerp(v, b[idx], t));
}

function smoothstep(t) {
  const c = Math.max(0, Math.min(1, t));
  return c * c * (3 - 2 * c);
}

function normalizeVec(v) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function resolveJoints(hand) {
  if (hand.joints) return hand.joints;
  const preset = HANDSHAPES[hand.shape || "relaxed"] || HANDSHAPES.relaxed;
  return {
    thumb: [...preset.thumb],
    index: [...preset.index],
    middle: [...preset.middle],
    ring: [...preset.ring],
    pinky: [...preset.pinky],
  };
}

function interpolateJointDict(jA, jB, t) {
  return {
    thumb: lerpArr(jA.thumb, jB.thumb, t),
    index: lerpArr(jA.index, jB.index, t),
    middle: lerpArr(jA.middle, jB.middle, t),
    ring: lerpArr(jA.ring, jB.ring, t),
    pinky: lerpArr(jA.pinky, jB.pinky, t),
  };
}

function resolveKeyframePose(kf) {
  const right = {
    wrist: kf.right?.wrist || IDLE_POSE.right.wrist,
    palm: normalizeVec(kf.right?.palm || IDLE_POSE.right.palm),
    finger: normalizeVec(kf.right?.finger || IDLE_POSE.right.finger),
    shape: kf.right?.shape || IDLE_POSE.right.shape,
  };
  const left = {
    wrist: kf.left?.wrist || IDLE_POSE.left.wrist,
    palm: normalizeVec(kf.left?.palm || IDLE_POSE.left.palm),
    finger: normalizeVec(kf.left?.finger || IDLE_POSE.left.finger),
    shape: kf.left?.shape || IDLE_POSE.left.shape,
  };
  return {
    right: { ...right, joints: resolveJoints(right) },
    left: { ...left, joints: resolveJoints(left) },
    head: {
      pitch: kf.head?.pitch ?? 0,
      yaw: kf.head?.yaw ?? 0,
      roll: kf.head?.roll ?? 0,
    },
    torsoYaw: kf.torsoYaw ?? 0,
    gazeX: kf.gazeX ?? 0,
    browOffset: kf.browOffset ?? 0,
    browFurrow: kf.browFurrow ?? 0,
    smile: kf.smile ?? IDLE_POSE.smile,
    activeLoci: kf.activeLoci || IDLE_POSE.activeLoci,
  };
}

function blendPoses(poseA, poseB, t) {
  const s = smoothstep(t);
  return {
    right: {
      wrist: lerpArr(poseA.right.wrist, poseB.right.wrist, s),
      palm: normalizeVec(lerpArr(poseA.right.palm, poseB.right.palm, s)),
      finger: normalizeVec(lerpArr(poseA.right.finger, poseB.right.finger, s)),
      joints: interpolateJointDict(
        resolveJoints(poseA.right),
        resolveJoints(poseB.right),
        s
      ),
    },
    left: {
      wrist: lerpArr(poseA.left.wrist, poseB.left.wrist, s),
      palm: normalizeVec(lerpArr(poseA.left.palm, poseB.left.palm, s)),
      finger: normalizeVec(lerpArr(poseA.left.finger, poseB.left.finger, s)),
      joints: interpolateJointDict(
        resolveJoints(poseA.left),
        resolveJoints(poseB.left),
        s
      ),
    },
    head: {
      pitch: lerp(poseA.head.pitch, poseB.head.pitch, s),
      yaw: lerp(poseA.head.yaw, poseB.head.yaw, s),
      roll: lerp(poseA.head.roll, poseB.head.roll, s),
    },
    torsoYaw: lerp(poseA.torsoYaw, poseB.torsoYaw, s),
    gazeX: lerp(poseA.gazeX, poseB.gazeX, s),
    browOffset: lerp(poseA.browOffset, poseB.browOffset, s),
    browFurrow: lerp(poseA.browFurrow, poseB.browFurrow, s),
    smile: lerp(poseA.smile, poseB.smile, s),
    activeLoci: s < 0.5 ? poseA.activeLoci : poseB.activeLoci,
  };
}

function getFingerspellTrack(letters) {
  if (!letters || letters.length === 0) return [IDLE_POSE];
  const count = letters.length;
  const track = [];
  for (let i = 0; i < count; i += 1) {
    const ch = String(letters[i]).toUpperCase();
    const spec = LETTER_HANDSHAPE[ch] || LETTER_HANDSHAPE.B;
    const tCenter = (i + 0.5) / count;
    // Slight lateral drift + gentle forward bounce for doubled letters.
    const xOffset = -0.2 + (i / Math.max(1, count - 1)) * 0.08;
    const zBounce = i % 2 === 1 ? 0.31 : 0.29;
    track.push({
      t: Math.max(0, tCenter - 0.12 / count),
      right: {
        wrist: [xOffset, 1.32, zBounce],
        palm: spec.palm,
        finger: spec.finger,
        shape: spec.shape,
      },
      head: { pitch: 0.02, yaw: 0, roll: 0 },
      smile: 0.4,
    });
    track.push({
      t: Math.min(1, tCenter + 0.16 / count),
      right: {
        wrist: [xOffset, 1.32, zBounce],
        palm: spec.palm,
        finger: spec.finger,
        shape: spec.shape,
      },
      head: { pitch: 0.02, yaw: 0, roll: 0 },
      smile: 0.4,
    });
  }
  return track;
}

function sampleTrackAt(track, t) {
  if (!track || track.length === 0) return resolveKeyframePose(IDLE_POSE);
  if (track.length === 1 || t <= track[0].t) {
    return resolveKeyframePose(track[0]);
  }
  if (t >= track[track.length - 1].t) {
    return resolveKeyframePose(track[track.length - 1]);
  }
  for (let i = 0; i < track.length - 1; i += 1) {
    const k0 = track[i];
    const k1 = track[i + 1];
    if (t >= k0.t && t <= k1.t) {
      const span = Math.max(0.0001, k1.t - k0.t);
      const localT = (t - k0.t) / span;
      return blendPoses(
        resolveKeyframePose(k0),
        resolveKeyframePose(k1),
        localT
      );
    }
  }
  return resolveKeyframePose(track[track.length - 1]);
}

function getStepTrack(step) {
  if (step.type === "fingerspell") {
    return getFingerspellTrack(step.letters || []);
  }
  return SIGN_TRACKS[step.id] || SIGN_TRACKS.unknown_gloss;
}

export function getTotalPlanDurationMs(plan) {
  if (!plan?.steps?.length) return 0;
  return plan.steps.reduce((acc, s) => acc + (s.duration_ms || 1100), 0);
}

/**
 * Sample the 3D avatar pose at `elapsedMs` along a SignPlan, applying smooth
 * co-articulation blending between signs and NMM facial/head inflections.
 */
export function sampleSignPlanPose(plan, elapsedMs, idleTimeSec = 0) {
  const idle = resolveKeyframePose(IDLE_POSE);
  // Subtle natural breathing + blinking when idle.
  const breath = Math.sin(idleTimeSec * 1.7) * 0.008;
  idle.right.wrist = [
    idle.right.wrist[0],
    idle.right.wrist[1] + breath,
    idle.right.wrist[2],
  ];
  idle.left.wrist = [
    idle.left.wrist[0],
    idle.left.wrist[1] + breath,
    idle.left.wrist[2],
  ];

  const steps = plan?.steps || [];
  if (steps.length === 0) {
    const base = blendPoses(idle, idle, 0);
    return {
      pose: base,
      stepIndex: -1,
      activeStep: null,
      activeLetter: null,
      activeLetterIndex: -1,
      progress: 0,
    };
  }

  const totalMs = getTotalPlanDurationMs(plan);
  const clampedMs = Math.max(0, Math.min(totalMs, elapsedMs));

  let cursor = 0;
  let stepIndex = steps.length - 1;
  let stepStartMs = 0;
  for (let i = 0; i < steps.length; i += 1) {
    const dur = steps[i].duration_ms || 1100;
    if (clampedMs <= cursor + dur || i === steps.length - 1) {
      stepIndex = i;
      stepStartMs = cursor;
      break;
    }
    cursor += dur;
  }

  const activeStep = steps[stepIndex];
  const stepDur = activeStep.duration_ms || 1100;
  const localMs = Math.max(0, clampedMs - stepStartMs);
  const localT = Math.min(1, localMs / stepDur);

  const currentTrack = getStepTrack(activeStep);
  const transitionMs = Math.min(230, stepDur * 0.24);

  let sampled;
  if (localMs < transitionMs) {
    const prevPose =
      stepIndex === 0
        ? idle
        : sampleTrackAt(getStepTrack(steps[stepIndex - 1]), 1.0);
    const enterPose = sampleTrackAt(currentTrack, transitionMs / stepDur);
    sampled = blendPoses(prevPose, enterPose, localMs / transitionMs);
  } else {
    sampled = sampleTrackAt(currentTrack, localT);
    // Smoothly return to ready stance at the end of the final step.
    if (stepIndex === steps.length - 1 && localT > 0.85) {
      const exitT = (localT - 0.85) / 0.15;
      const endPose = sampleTrackAt(currentTrack, 1.0);
      sampled = blendPoses(endPose, idle, exitT);
    }
  }

  // Layer clause-level Non-Manual Markers (NMM) from the grammar planner.
  const nmm = plan.nmm || {};
  const clauseProgress = totalMs > 0 ? clampedMs / totalMs : 0;
  if (nmm.brow_state === "raised") {
    sampled.browOffset = Math.max(sampled.browOffset, 0.9);
  } else if (nmm.brow_state === "furrowed") {
    sampled.browFurrow = Math.max(sampled.browFurrow, 0.9);
  }
  if (nmm.head_tilt) {
    sampled.head.roll += 0.08;
    sampled.head.pitch += 0.04;
  }
  if (nmm.head_shake) {
    sampled.head.yaw += Math.sin(clauseProgress * Math.PI * 4) * 0.14;
  }
  if (nmm.head_nod) {
    sampled.head.pitch += Math.sin(clauseProgress * Math.PI * 4) * 0.1;
  }

  let activeLetter = null;
  let activeLetterIndex = -1;
  if (activeStep.type === "fingerspell" && activeStep.letters?.length) {
    const idx = Math.min(
      activeStep.letters.length - 1,
      Math.floor(localT * activeStep.letters.length)
    );
    activeLetter = activeStep.letters[idx];
    activeLetterIndex = idx;
  }

  return {
    pose: sampled,
    stepIndex,
    activeStep,
    activeLetter,
    activeLetterIndex,
    progress: totalMs > 0 ? clampedMs / totalMs : 0,
  };
}

/**
 * Local fallback planner if backend /api/text-to-sign is temporarily offline.
 */
export function buildLocalSignPlan(text, phrases = []) {
  const cleaned = (text || "").trim();
  if (!cleaned) {
    return {
      input_text: "",
      sentence_type: "statement",
      nmm: { brow_state: "neutral", head_nod: false, head_shake: false, head_tilt: false },
      gloss_line: "",
      steps: [],
    };
  }
  const hasQ = cleaned.includes("?");
  const lower = cleaned.toLowerCase();
  const isWh = /\b(what|where|who|when|why|how)\b/i.test(lower);
  const isNeg = /\b(no|not|never|nope)\b/i.test(lower);

  const steps = [];
  const words = cleaned.replace(/[^\p{L}\p{N}\s]/gu, " ").split(/\s+/).filter(Boolean);
  const fillers = new Set(["is", "am", "are", "the", "a", "an", "to", "of", "in", "on", "do"]);
  const matchedIds = new Set();

  for (const p of phrases) {
    const trigMatch = (p.triggers || []).some((t) => lower.includes(t.toLowerCase()));
    if (trigMatch && !matchedIds.has(p.id)) {
      matchedIds.add(p.id);
      steps.push({
        type: "sign",
        id: p.id,
        gloss: p.gloss,
        cue: p.cue,
        duration_ms: 1200,
      });
    }
  }

  if (steps.length === 0) {
    for (const w of words) {
      if (fillers.has(w.toLowerCase())) continue;
      const letters = Array.from(w.toUpperCase()).filter((c) => /[A-Z0-9]/.test(c)).slice(0, 12);
      if (letters.length > 0) {
        steps.push({
          type: "fingerspell",
          id: "fingerspell",
          gloss: w.toUpperCase(),
          cue: `Fingerspell ${letters.join("-")}`,
          duration_ms: Math.max(900, letters.length * 450 + 200),
          letters,
        });
      }
    }
  }

  return {
    input_text: cleaned,
    sentence_type: isWh ? "wh_question" : hasQ ? "yes_no_question" : isNeg ? "negation" : "statement",
    nmm: {
      brow_state: isWh ? "furrowed" : hasQ ? "raised" : "neutral",
      head_nod: false,
      head_shake: isNeg,
      head_tilt: isWh || hasQ,
    },
    gloss_line: steps.map((s) => s.gloss).join(" "),
    steps,
  };
}

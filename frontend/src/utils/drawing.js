export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export const POSE_CONNECTIONS = [
  [11, 12], // Shoulders
  [11, 13], [13, 15], // Left arm
  [12, 14], [14, 16], // Right arm
  [11, 23], [12, 24], [23, 24], // Torso box
];

const FINGERTIP_INDICES = new Set([4, 8, 12, 16, 20]);

const HAND_PALETTE = {
  Left: {
    bone: "rgba(56, 189, 248, 0.92)",
    joint: "#e0f2fe",
    tip: "#0ea5e9",
    badgeBg: "rgba(14, 165, 233, 0.88)",
  },
  Right: {
    bone: "rgba(52, 211, 153, 0.92)",
    joint: "#d1fae5",
    tip: "#10b981",
    badgeBg: "rgba(16, 185, 129, 0.88)",
  },
  Unknown: {
    bone: "rgba(251, 191, 36, 0.92)",
    joint: "#fef3c7",
    tip: "#f59e0b",
    badgeBg: "rgba(245, 158, 11, 0.88)",
  },
};

function drawPolyline(ctx, pts, width, height, strokeStyle, lineWidth = 1.5) {
  if (!pts || pts.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(pts[0][0] * width, pts[0][1] * height);
  for (let i = 1; i < pts.length; i += 1) {
    ctx.lineTo(pts[i][0] * width, pts[i][1] * height);
  }
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}

function drawDirectionalArc(ctx, width, height, leftX, rightX, fromZone, toZone, label) {
  const zoneX = (z) => {
    if (z === "left") return leftX * 0.5;
    if (z === "right") return rightX + (width - rightX) * 0.5;
    return (leftX + rightX) * 0.5;
  };

  const x1 = zoneX(fromZone);
  const x2 = zoneX(toZone);
  const y = height * 0.56;
  const ctrlX = (x1 + x2) * 0.5;
  const ctrlY = y - Math.min(90, Math.abs(x2 - x1) * 0.28);

  ctx.save();
  ctx.strokeStyle = "rgba(56, 189, 248, 0.92)";
  ctx.lineWidth = 3.5;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x1, y);
  ctx.quadraticCurveTo(ctrlX, ctrlY, x2, y);
  ctx.stroke();

  // Arrowhead at target locus
  const angle = Math.atan2(y - ctrlY, x2 - ctrlX);
  const headLen = 12;
  ctx.fillStyle = "#38bdf8";
  ctx.beginPath();
  ctx.moveTo(x2, y);
  ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6), y - headLen * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6), y - headLen * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();

  if (label) {
    ctx.font = "700 12px system-ui, -apple-system, sans-serif";
    const textW = ctx.measureText(label).width;
    const bx = ctrlX - (textW + 18) / 2;
    const by = ctrlY - 28;
    ctx.fillStyle = "rgba(15, 23, 42, 0.9)";
    ctx.strokeStyle = "rgba(56, 189, 248, 0.85)";
    ctx.lineWidth = 1.5;
    roundRect(ctx, bx, by, textW + 18, 24, 999);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#e0f2fe";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, ctrlX, by + 12.5);
  }
  ctx.restore();
}

/**
 * Draw signing-space guides (body-anchored L/C/R lanes), holistic pose + face
 * contours (NMM indicators), directional verb arcs, and 21-point hand skeletons.
 */
export function drawOverlay(
  ctx,
  width,
  height,
  {
    hands = [],
    holistic = null,
    spatialGrammar = null,
    directionalFlash = null,
    showGuides = true,
    showSpatialLanes = true,
    showHolistic = true,
    leftZoneMaxX = 0.35,
    rightZoneMinX = 0.65,
    activeLoci = { left: null, right: null },
  }
) {
  ctx.clearRect(0, 0, width, height);

  const signingSpace = spatialGrammar?.signing_space;
  const nmm = spatialGrammar?.non_manual_markers;
  const referentFocus = spatialGrammar?.referent_focus || "center";
  const effectiveLeftX = signingSpace?.left_boundary ?? leftZoneMaxX;
  const effectiveRightX = signingSpace?.right_boundary ?? rightZoneMinX;
  const isBodyAnchored = signingSpace?.source === "holistic_pose";

  if (showSpatialLanes) {
    const lx = effectiveLeftX * width;
    const rx = effectiveRightX * width;

    ctx.save();
    ctx.strokeStyle = isBodyAnchored
      ? "rgba(99, 102, 241, 0.55)"
      : "rgba(148, 163, 184, 0.35)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);

    ctx.beginPath();
    ctx.moveTo(lx, 12);
    ctx.lineTo(lx, height - 12);
    ctx.moveTo(rx, 12);
    ctx.lineTo(rx, height - 12);
    ctx.stroke();

    if (isBodyAnchored && typeof signingSpace?.torso_center_x === "number") {
      const mx = signingSpace.torso_center_x * width;
      ctx.strokeStyle = "rgba(56, 189, 248, 0.25)";
      ctx.setLineDash([3, 6]);
      ctx.beginPath();
      ctx.moveTo(mx, 44);
      ctx.lineTo(mx, height - 44);
      ctx.stroke();
    }

    ctx.setLineDash([]);
    ctx.font = "600 12px system-ui, -apple-system, sans-serif";

    const drawZonePill = (text, cx, active, focused) => {
      const padX = 10;
      const metrics = ctx.measureText(text);
      const pillW = metrics.width + padX * 2;
      const pillH = 24;
      const x = Math.max(8, Math.min(width - pillW - 8, cx - pillW / 2));
      const y = 12;

      ctx.fillStyle = focused
        ? "rgba(14, 165, 233, 0.9)"
        : active
        ? "rgba(99, 102, 241, 0.85)"
        : "rgba(15, 23, 42, 0.68)";
      ctx.strokeStyle = focused
        ? "rgba(186, 230, 253, 0.98)"
        : active
        ? "rgba(165, 180, 252, 0.95)"
        : "rgba(148, 163, 184, 0.35)";
      ctx.lineWidth = focused ? 1.8 : 1;
      roundRect(ctx, x, y, pillW, pillH, 999);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#f8fafc";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, x + pillW / 2, y + pillH / 2 + 0.5);
    };

    const leftTag = activeLoci.left ? `LEFT · ${activeLoci.left}` : "LEFT · Locus A";
    const rightTag = activeLoci.right ? `RIGHT · ${activeLoci.right}` : "RIGHT · Locus B";
    const centerTag = isBodyAnchored ? "CENTER · Body Anchor" : "CENTER · Neutral";

    drawZonePill(
      referentFocus === "left" ? `◉ ${leftTag}` : leftTag,
      lx / 2,
      Boolean(activeLoci.left),
      referentFocus === "left"
    );
    drawZonePill(
      centerTag,
      (lx + rx) / 2,
      isBodyAnchored,
      false
    );
    drawZonePill(
      referentFocus === "right" ? `◉ ${rightTag}` : rightTag,
      rx + (width - rx) / 2,
      Boolean(activeLoci.right),
      referentFocus === "right"
    );
    ctx.restore();
  }

  if (showGuides) {
    const padX = width * 0.08;
    const padTop = height * 0.14;
    const padBottom = height * 0.08;
    const boxW = width - padX * 2;
    const boxH = height - padTop - padBottom;

    ctx.save();
    ctx.strokeStyle =
      hands.length > 0
        ? "rgba(52, 211, 153, 0.45)"
        : "rgba(226, 232, 240, 0.32)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([8, 6]);
    roundRect(ctx, padX, padTop, boxW, boxH, 16);
    ctx.stroke();
    ctx.restore();
  }

  // Holistic upper-body Pose + Face contours for spatial & NMM grammar
  if (showHolistic && holistic?.enabled) {
    const pose = holistic.pose_landmarks || [];
    if (pose.length > 16) {
      ctx.save();
      ctx.strokeStyle = "rgba(129, 140, 248, 0.65)";
      ctx.lineWidth = Math.max(2, width * 0.0032);
      ctx.lineCap = "round";
      for (const [aIdx, bIdx] of POSE_CONNECTIONS) {
        const a = pose[aIdx];
        const b = pose[bIdx];
        if (!a || !b) continue;
        if ((a.v ?? 1) < 0.25 || (b.v ?? 1) < 0.25) continue;
        ctx.beginPath();
        ctx.moveTo(a.x * width, a.y * height);
        ctx.lineTo(b.x * width, b.y * height);
        ctx.stroke();
      }
      for (const idx of [11, 12, 13, 14]) {
        const p = pose[idx];
        if (!p || (p.v ?? 1) < 0.25) continue;
        ctx.beginPath();
        ctx.arc(p.x * width, p.y * height, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = "#a5b4fc";
        ctx.fill();
      }
      ctx.restore();
    }

    const contours = holistic.face_contours || {};
    if (Object.keys(contours).length > 0) {
      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      const ovalStroke =
        nmm?.sentence_type === "negation"
          ? "rgba(248, 113, 113, 0.78)"
          : nmm?.sentence_type === "affirmation"
          ? "rgba(52, 211, 153, 0.78)"
          : "rgba(148, 163, 184, 0.36)";
      drawPolyline(ctx, contours.face_oval, width, height, ovalStroke, 1.3);
      drawPolyline(ctx, contours.nose_bridge, width, height, "rgba(148, 163, 184, 0.32)", 1.1);
      drawPolyline(ctx, contours.left_eye, width, height, "rgba(125, 211, 252, 0.65)", 1.3);
      drawPolyline(ctx, contours.right_eye, width, height, "rgba(125, 211, 252, 0.65)", 1.3);
      drawPolyline(ctx, contours.left_iris, width, height, "rgba(56, 189, 248, 0.9)", 1.5);
      drawPolyline(ctx, contours.right_iris, width, height, "rgba(56, 189, 248, 0.9)", 1.5);

      const mouthColor =
        nmm?.mouth_state === "open" || nmm?.mouth_state === "pucker"
          ? "rgba(244, 114, 182, 0.85)"
          : "rgba(226, 232, 240, 0.55)";
      drawPolyline(ctx, contours.lips_outer, width, height, mouthColor, 1.4);
      drawPolyline(ctx, contours.lips_inner, width, height, "rgba(226, 232, 240, 0.35)", 1);

      const browColor =
        nmm?.brow_state === "raised"
          ? "rgba(251, 191, 36, 0.95)"
          : nmm?.brow_state === "furrowed"
          ? "rgba(251, 113, 133, 0.95)"
          : "rgba(203, 213, 225, 0.72)";
      const browWidth = nmm?.brow_state !== "neutral" ? 2.6 : 1.7;
      drawPolyline(ctx, contours.left_eyebrow, width, height, browColor, browWidth);
      drawPolyline(ctx, contours.right_eyebrow, width, height, browColor, browWidth);
      ctx.restore();
    }
  }

  if (directionalFlash && showSpatialLanes) {
    drawDirectionalArc(
      ctx,
      width,
      height,
      effectiveLeftX * width,
      effectiveRightX * width,
      directionalFlash.from_zone,
      directionalFlash.to_zone,
      directionalFlash.phrases?.en || directionalFlash.action
    );
  }

  for (const hand of hands) {
    const landmarks = hand.landmarks || [];
    if (landmarks.length === 0) continue;

    const palette = HAND_PALETTE[hand.handedness] || HAND_PALETTE.Unknown;

    ctx.save();
    // Draw bone connections
    ctx.strokeStyle = palette.bone;
    ctx.lineWidth = Math.max(2, width * 0.004);
    ctx.lineCap = "round";

    for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
      const a = landmarks[startIdx];
      const b = landmarks[endIdx];
      if (!a || !b) continue;
      ctx.beginPath();
      ctx.moveTo(a.x * width, a.y * height);
      ctx.lineTo(b.x * width, b.y * height);
      ctx.stroke();
    }

    // Draw joints and track bounding box
    let minX = 1;
    let minY = 1;
    let maxX = 0;
    let maxY = 0;

    for (let idx = 0; idx < landmarks.length; idx += 1) {
      const lm = landmarks[idx];
      if (lm.x < minX) minX = lm.x;
      if (lm.y < minY) minY = lm.y;
      if (lm.x > maxX) maxX = lm.x;
      if (lm.y > maxY) maxY = lm.y;

      const cx = lm.x * width;
      const cy = lm.y * height;
      const isTip = FINGERTIP_INDICES.has(idx);
      const isWrist = idx === 0;
      const radius = isWrist ? 6 : isTip ? 5 : 3.5;

      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = isTip || isWrist ? palette.tip : palette.joint;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(15, 23, 42, 0.85)";
      ctx.stroke();
    }

    // Floating label pill above the hand bounding box
    const gestureLabel =
      hand.gesture && hand.gesture !== "None"
        ? `${hand.gesture} ${(hand.gesture_score * 100).toFixed(0)}%`
        : "Tracking";
    const zoneTag = (hand.spatial_zone || "center").toUpperCase();
    const heightTag = hand.height_level ? `/${hand.height_level}` : "";
    const labelText = `${hand.handedness} · ${gestureLabel} · ${zoneTag}${heightTag}`;

    ctx.font = "600 12px system-ui, -apple-system, sans-serif";
    const textW = ctx.measureText(labelText).width;
    const pillW = textW + 16;
    const pillH = 24;
    const bx = Math.max(8, Math.min(width - pillW - 8, minX * width));
    const by = Math.max(40, minY * height - pillH - 8);

    ctx.fillStyle = palette.badgeBg;
    roundRect(ctx, bx, by, pillW, pillH, 8);
    ctx.fill();

    ctx.fillStyle = "#0f172a";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(labelText, bx + 8, by + pillH / 2 + 0.5);
    ctx.restore();
  }
}

function roundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

/**
 * Evaluate capture clarity directly from hand landmarks & frame brightness.
 */
export function evaluateCaptureQuality(hands, brightness) {
  if (brightness !== null && brightness < 52) {
    return {
      status: "warn",
      message: "Low light detected — increase room light or boost Brightness slider",
    };
  }
  if (brightness !== null && brightness > 225) {
    return {
      status: "warn",
      message: "Overexposed frame — reduce glare or lower Brightness",
    };
  }
  if (!hands || hands.length === 0) {
    return {
      status: "idle",
      message: "Place your hand inside the dashed signing guide",
    };
  }

  for (const hand of hands) {
    const landmarks = hand.landmarks || [];
    if (landmarks.length === 0) continue;
    let minX = 1;
    let minY = 1;
    let maxX = 0;
    let maxY = 0;
    for (const lm of landmarks) {
      if (lm.x < minX) minX = lm.x;
      if (lm.y < minY) minY = lm.y;
      if (lm.x > maxX) maxX = lm.x;
      if (lm.y > maxY) maxY = lm.y;
    }

    if (minX < 0.03 || maxX > 0.97 || minY < 0.03 || maxY > 0.97) {
      return {
        status: "warn",
        message: "Hand is too close to the frame edge — center it inside the box",
      };
    }

    const area = (maxX - minX) * (maxY - minY);
    if (area < 0.015) {
      return {
        status: "warn",
        message: "Hand looks small — move slightly closer to the camera",
      };
    }
  }

  return {
    status: "ok",
    message: "Clear capture — landmarks & gesture tracking locked cleanly",
  };
}

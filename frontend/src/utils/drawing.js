export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
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

/**
 * Draw optional signing-space guides (optimal capture box + L/C/R spatial lanes)
 * and detected 21-point hand skeletons onto the aligned overlay canvas.
 */
export function drawOverlay(
  ctx,
  width,
  height,
  {
    hands = [],
    showGuides = true,
    showSpatialLanes = true,
    leftZoneMaxX = 0.35,
    rightZoneMinX = 0.65,
    activeLoci = { left: null, right: null },
  }
) {
  ctx.clearRect(0, 0, width, height);

  if (showSpatialLanes) {
    const lx = leftZoneMaxX * width;
    const rx = rightZoneMinX * width;

    ctx.save();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);

    ctx.beginPath();
    ctx.moveTo(lx, 12);
    ctx.lineTo(lx, height - 12);
    ctx.moveTo(rx, 12);
    ctx.lineTo(rx, height - 12);
    ctx.stroke();

    ctx.setLineDash([]);
    ctx.font = "600 12px system-ui, -apple-system, sans-serif";

    const drawZonePill = (text, cx, active) => {
      const padX = 10;
      const metrics = ctx.measureText(text);
      const pillW = metrics.width + padX * 2;
      const pillH = 24;
      const x = cx - pillW / 2;
      const y = 12;

      ctx.fillStyle = active
        ? "rgba(99, 102, 241, 0.85)"
        : "rgba(15, 23, 42, 0.68)";
      ctx.strokeStyle = active
        ? "rgba(165, 180, 252, 0.95)"
        : "rgba(148, 163, 184, 0.35)";
      ctx.lineWidth = 1;
      roundRect(ctx, x, y, pillW, pillH, 999);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#f8fafc";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, cx, y + pillH / 2 + 0.5);
    };

    drawZonePill(
      activeLoci.left ? `LEFT · ${activeLoci.left}` : "LEFT · Locus A",
      lx / 2,
      Boolean(activeLoci.left)
    );
    drawZonePill("CENTER · Neutral", (lx + rx) / 2, false);
    drawZonePill(
      activeLoci.right ? `RIGHT · ${activeLoci.right}` : "RIGHT · Locus B",
      rx + (width - rx) / 2,
      Boolean(activeLoci.right)
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
    const labelText = `${hand.handedness} · ${gestureLabel} · ${zoneTag}`;

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

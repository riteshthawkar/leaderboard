import { useEffect, useMemo, useRef, useState } from "react";

// Per-benchmark hero motifs — a cohesive "recursive geometry" system.
// Each page modulates the same precise parametric language (nested / offset
// forms) so they read as one designed family, and each encodes its dataset:
//   dysm      -> visual perception: moiré interference of two concentric-ring fields
//   minds_eye -> cognition: a flat net folding up into a 3D cube
//   spatial   -> spatial reasoning: nested rectangles in one-point perspective
// Monochrome + theme-aware via currentColor.

function Glow({ id, cx, cy, r }) {
  return (
    <>
      <defs>
        <radialGradient id={id} cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.15" />
          <stop offset="55%" stopColor="currentColor" stopOpacity="0.05" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
        </radialGradient>
      </defs>
      <circle cx={cx} cy={cy} r={r} fill={`url(#${id})`} stroke="none" />
    </>
  );
}

// --- dysm: "Do You See Me" — a single A4 frame on a technical grid. Each cycle a
// new image appears; a scanner sweeps top->bottom and everything it has passed is
// re-rendered as a coarse pixel mosaic with tiles dropped out — what the model
// actually "perceives": lower resolution, with parts missing. No skew: just the
// frame, the image, and the scan line.
const DW = 100; // local frame width (authoring units)
const DH = 141; // local frame height (A4 portrait ratio)
const DFRAME = { w: 360, h: 430, x: 765, y: 85 }; // viewBox placement; w/h independent so the frame can be wider without getting taller
const DSCX = DFRAME.w / DW; // local -> viewBox scale (x)
const DSCY = DFRAME.h / DH; // local -> viewBox scale (y), independent of x so width can grow alone

// each image is a back-to-front list of grayscale shapes in DW x DH space.
// kinds: rect | circle | ellipse | tri | poly. g = currentColor fill opacity;
// optional st = stroke opacity (outline), sw = stroke width. These mimic the
// "Do You See Me" benchmark samples: shape grids, scattered polygons, letters.
function dReg(cx, cy, r, n, rotDeg = 0) {
  const pts = [];
  const rot = (rotDeg * Math.PI) / 180;
  for (let i = 0; i < n; i += 1) {
    const a = rot - Math.PI / 2 + (i * 2 * Math.PI) / n;
    pts.push([+(cx + r * Math.cos(a)).toFixed(1), +(cy + r * Math.sin(a)).toFixed(1)]);
  }
  return pts;
}
function dBuildGrid() {
  const shapes = [{ k: "rect", x: 0, y: 0, w: 100, h: 141, g: 0.05 }];
  const cols = 5;
  const rows = 7;
  const m = 9;
  const cw = (100 - 2 * m) / cols;
  const ch = (141 - 2 * m) / rows;
  const sz = Math.min(cw, ch) * 0.6;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const cx = m + cw * (c + 0.5);
      const cy = m + ch * (r + 0.5);
      const g = (c + r) % 2 === 0 ? 0.5 : 0.14;
      const kind = (c + 2 * r) % 3;
      if (kind === 0) shapes.push({ k: "circle", cx, cy, r: sz, g, st: 0.55, sw: 0.6 });
      else if (kind === 1) shapes.push({ k: "rect", x: cx - sz, y: cy - sz, w: sz * 2, h: sz * 2, g, st: 0.55, sw: 0.6 });
      else shapes.push({ k: "tri", pts: [[cx, cy - sz], [cx - sz, cy + sz], [cx + sz, cy + sz]], g, st: 0.55, sw: 0.6 });
    }
  }
  return shapes;
}
function dBuildDots() {
  const shapes = [{ k: "rect", x: 0, y: 0, w: 100, h: 141, g: 0.05 }];
  const cols = 6;
  const rows = 8;
  const m = 11;
  const cw = (100 - 2 * m) / cols;
  const ch = (141 - 2 * m) / rows;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const cx = m + cw * (c + 0.5);
      const cy = m + ch * (r + 0.5);
      const dark = (c * 2 + r * 3 + 1) % 5 === 0;
      shapes.push({ k: "circle", cx, cy, r: Math.min(cw, ch) * 0.34, g: dark ? 0.52 : 0.16, st: 0.5, sw: 0.5 });
    }
  }
  return shapes;
}
const D_IMAGES = {
  grid: dBuildGrid(),
  scatter: [
    { k: "rect", x: 0, y: 0, w: 100, h: 141, g: 0.05 },
    { k: "poly", pts: dReg(22, 24, 10, 5, 20), g: 0.5, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(58, 18, 9, 6, 40), g: 0.3, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(82, 30, 11, 5, 200), g: 0.42, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(16, 52, 8, 6, 15), g: 0.22, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(45, 48, 12, 8, 30), g: 0.5, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(76, 58, 9, 5, 250), g: 0.16, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(30, 76, 11, 6, 60), g: 0.4, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(60, 82, 9, 8, 20), g: 0.28, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(84, 92, 10, 5, 300), g: 0.5, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(20, 102, 10, 6, 200), g: 0.34, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(48, 114, 11, 5, 76), g: 0.22, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(76, 120, 9, 6, 216), g: 0.44, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(36, 130, 8, 8, 150), g: 0.3, st: 0.5, sw: 0.6 },
  ],
  letter: [
    { k: "rect", x: 0, y: 0, w: 100, h: 141, g: 0.05 },
    { k: "poly", pts: [[30, 30], [70, 30], [70, 41], [43, 41], [43, 64], [62, 64], [62, 75], [43, 75], [43, 99], [70, 99], [70, 110], [30, 110]], g: 0.5, st: 0.5, sw: 0.6 },
  ],
  constancy: [
    { k: "rect", x: 0, y: 0, w: 100, h: 141, g: 0.06 },
    { k: "poly", pts: dReg(28, 34, 15, 5, 0), g: 0.4, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(70, 46, 10, 5, 34), g: 0.28, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(38, 82, 18, 5, 198), g: 0.34, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(74, 104, 12, 5, 90), g: 0.22, st: 0.5, sw: 0.6 },
    { k: "poly", pts: dReg(30, 120, 9, 5, 145), g: 0.46, st: 0.5, sw: 0.6 },
  ],
  dots: dBuildDots(),
};
const D_KEYS = ["grid", "scatter", "letter", "constancy", "dots"];
const D_LABELS = ["SPATIAL RELATION", "FEATURE BINDING", "LETTER DISAMBIGUATION", "FORM CONSTANCY", "FIGURE-GROUND"];

function dInside(s, px, py) {
  if (s.k === "rect") return px >= s.x && px <= s.x + s.w && py >= s.y && py <= s.y + s.h;
  if (s.k === "circle") {
    const dx = px - s.cx;
    const dy = py - s.cy;
    return dx * dx + dy * dy <= s.r * s.r;
  }
  if (s.k === "ellipse") {
    const dx = (px - s.cx) / s.rx;
    const dy = (py - s.cy) / s.ry;
    return dx * dx + dy * dy <= 1;
  }
  if (s.k === "poly") {
    const p = s.pts; // ray-casting point-in-polygon (handles concave)
    let inside = false;
    for (let i = 0, j = p.length - 1; i < p.length; j = i, i += 1) {
      const xi = p[i][0], yi = p[i][1], xj = p[j][0], yj = p[j][1];
      if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }
  const [a, b, c] = s.pts; // triangle: barycentric sign test
  const den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
  const w1 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / den;
  const w2 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / den;
  return w1 >= 0 && w2 >= 0 && w1 + w2 <= 1;
}
function dSampleG(shapes, px, py) {
  let g = 0;
  for (let i = 0; i < shapes.length; i += 1) if (dInside(shapes[i], px, py)) g = shapes[i].g;
  return g;
}
function dCleanShapes(shapes) {
  return shapes.map((s, i) => {
    const p = s.hole
      ? { fill: "var(--surface-2)", fillOpacity: 1 }
      : { fill: "currentColor", fillOpacity: s.g };
    if (s.st) {
      p.stroke = "currentColor";
      p.strokeOpacity = s.st;
      p.strokeWidth = s.sw || 0.6;
    }
    if (s.k === "rect") return <rect key={i} x={s.x} y={s.y} width={s.w} height={s.h} {...p} />;
    if (s.k === "circle") return <circle key={i} cx={s.cx} cy={s.cy} r={s.r} {...p} />;
    if (s.k === "ellipse") return <ellipse key={i} cx={s.cx} cy={s.cy} rx={s.rx} ry={s.ry} {...p} />;
    return <polygon key={i} points={s.pts.map((pt) => pt.join(",")).join(" ")} {...p} />;
  });
}
const D_NX = 9;
const D_NY = 13;
function dPixelCells(shapes, seed) {
  const cw = DW / D_NX;
  const ch = DH / D_NY;
  const cells = [];
  for (let r = 0; r < D_NY; r += 1) {
    for (let c = 0; c < D_NX; c += 1) {
      if ((c * 3 + r * 5 + seed * 11) % 7 === 0) continue; // dropped ("missing") tile
      const g = dSampleG(shapes, (c + 0.5) * cw, (r + 0.5) * ch);
      cells.push(<rect key={`${r}-${c}`} x={c * cw + 0.7} y={r * ch + 0.7} width={cw - 1.4} height={ch - 1.4} fill="currentColor" fillOpacity={g} />);
    }
  }
  return cells;
}
// technical grid that fills the whole hero (behind the frame). Extends well past
// the viewBox so it still fills the letterbox band on either side (the frame is
// right-aligned via preserveAspectRatio=xMaxYMid).
const D_GRID = (() => {
  const lines = [];
  const S = 44;
  for (let x = -440; x <= 1620; x += S) lines.push(<line key={`v${x}`} x1={x} y1={-260} x2={x} y2={900} />);
  for (let y = -260; y <= 900; y += S) lines.push(<line key={`h${y}`} x1={-440} y1={y} x2={1620} y2={y} />);
  return lines;
})();

function PerceptionArt() {
  const reduce =
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [t, setT] = useState(0);
  const rafRef = useRef(0);
  useEffect(() => {
    if (reduce) return undefined;
    let s;
    const tick = (ts) => {
      if (s === undefined) s = ts;
      setT((ts - s) / 1000);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [reduce]);

  const N = D_KEYS.length;
  const SLOT = 3.6;
  const idx = reduce ? 0 : Math.floor(t / SLOT) % N;
  const q = reduce ? 0 : (t % SLOT) / SLOT;

  const clean = useMemo(() => dCleanShapes(D_IMAGES[D_KEYS[idx]]), [idx]);
  const pixels = useMemo(() => dPixelCells(D_IMAGES[D_KEYS[idx]], idx), [idx]);

  const scanProg = reduce ? 0.6 : Math.max(0, Math.min(1, (q - 0.14) / 0.58));
  const scanY = DH * scanProg;
  const scanning = reduce ? true : q > 0.14 && q < 0.78;
  const imgOpacity = reduce ? 1 : seg(q, 0, 0.06) - seg(q, 0.93, 0.99);

  const label = D_LABELS[idx];
  const lp = reduce ? 1 : Math.max(0, Math.min(1, (q - 0.05) / 0.16));
  const labelShown = label.slice(0, Math.round(lp * label.length));
  const labelO = reduce ? 1 : seg(q, 0.05, 0.15) - seg(q, 0.9, 0.98);
  const num = String(idx + 1).padStart(2, "0");

  const { w: FW, x: FX, y: FY } = DFRAME;

  return (
    <svg viewBox="0 0 1180 600" preserveAspectRatio="xMaxYMid meet" aria-hidden="true" className="hero-art-svg">
      <Glow id="hero-glow-dysm" cx={FX + FW / 2} cy={300} r={360} />
      <g stroke="currentColor" strokeWidth="1" strokeOpacity="0.1">
        {D_GRID}
      </g>

      <g transform={`translate(${FX} ${FY}) scale(${DSCX} ${DSCY})`}>
        {/* opaque card so the frame reads cleanly over the grid */}
        <rect x="0" y="0" width={DW} height={DH} fill="var(--surface-2)" />

        <defs>
          <clipPath id="dysm-below">
            <rect x="0" y={scanY} width={DW} height={DH - scanY} />
          </clipPath>
          <clipPath id="dysm-above">
            <rect x="0" y="0" width={DW} height={scanY} />
          </clipPath>
        </defs>
        <g opacity={imgOpacity}>
          <g clipPath="url(#dysm-below)">{clean}</g>
          <g clipPath="url(#dysm-above)">{pixels}</g>
        </g>

        {/* scanner sweeping top -> bottom */}
        {scanning && (
          <g>
            <rect x="0" y={Math.max(0, scanY - 4)} width={DW} height="8" fill="currentColor" opacity="0.1" />
            <line x1="0" y1={scanY} x2={DW} y2={scanY} stroke="currentColor" strokeWidth="0.6" opacity="0.8" />
          </g>
        )}

        {/* double sharp frame border + corner registration marks */}
        <rect x="0" y="0" width={DW} height={DH} fill="none" stroke="currentColor" strokeWidth="0.5" strokeOpacity="0.6" />
        <rect x="1.5" y="1.5" width={DW - 3} height={DH - 3} fill="none" stroke="currentColor" strokeWidth="0.3" strokeOpacity="0.38" />
        <g fill="none" stroke="currentColor" strokeWidth="0.7" strokeOpacity="0.55">
          <path d="M-2 8 L-2 -2 L8 -2" />
          <path d="M92 -2 L102 -2 L102 8" />
          <path d="M-2 133 L-2 143 L8 143" />
          <path d="M92 143 L102 143 L102 133" />
        </g>
      </g>

      {/* animated category label above the frame — drawn in viewBox coords so the
          non-uniform (wider) frame scale doesn't stretch the text */}
      {labelO > 0.02 && (
        <text x={FX + 2} y={FY - 16} fontFamily="var(--font)" fontSize="17" letterSpacing="1" fill="currentColor" fillOpacity={labelO}>
          {`${num}  ${labelShown}`}
        </text>
      )}
    </svg>
  );
}

// --- minds_eye: "fold into form" — six flat faces rotate about their hinges
// to fold up into a 3D cube (mental composition / paper folding). Driven by
// requestAnimationFrame so every flap rigidly rotates — a real fold, not a fade.
const FISO = { tw: 62, th: 35, tz: 74, ox: 860, oy: 310 };
function fi(x, y, z) {
  return [FISO.ox + (x - y) * FISO.tw, FISO.oy + (x + y) * FISO.th - z * FISO.tz];
}

// Static isometric floor grid (z=0) filling the hero on integer lines, so it
// aligns with the cube's bottom face. Plus artist "construction" guides: two
// floor axes through the cube base and vertical lines up its corner edges.
const MI_G = 15;
const MI_GRID = [];
for (let k = -MI_G; k <= MI_G; k += 1) {
  MI_GRID.push([fi(k, -MI_G, 0), fi(k, MI_G, 0)]);
  MI_GRID.push([fi(-MI_G, k, 0), fi(MI_G, k, 0)]);
}
const MI_AXES = [
  [fi(0, -MI_G, 0), fi(0, MI_G, 0)],
  [fi(2, -MI_G, 0), fi(2, MI_G, 0)],
  [fi(-MI_G, 0, 0), fi(MI_G, 0, 0)],
  [fi(-MI_G, 2, 0), fi(MI_G, 2, 0)],
];
const MI_TOP_AXES = [
  [fi(0, -MI_G, 2), fi(0, MI_G, 2)],
  [fi(2, -MI_G, 2), fi(2, MI_G, 2)],
  [fi(-MI_G, 0, 2), fi(MI_G, 0, 2)],
  [fi(-MI_G, 2, 2), fi(MI_G, 2, 2)],
];
// vertical guides span the full hero height; horizontal guides mark the base
// (z=0) and the top edges (z=2) where the folded sides meet.
const MI_VERTS = [...new Set([[0, 0], [2, 0], [2, 2], [0, 2]].map(([x, y]) => fi(x, y, 0)[0]))].map((sx) => [[sx, -400], [sx, 1000]]);

// The six faces of a unit-2 cube, parametrized by the wall-fold angle `w` and
// the lid-fold angle `t` (radians). At w=t=0 they lie flat as a cross net; at
// w=t=π/2 they close into the cube. Each side rotates rigidly about its hinge
// edge; the lid is hinged to the south wall (its direction = w + t).
// The six faces of a unit-2 cube. Each wall has its OWN fold angle so the
// sides can fold up one at a time (wN,wE,wS,wW), plus the lid angle t. At all
// angles 0 they lie flat as a cross net; at all = π/2 they close into the cube.
// The lid is hinged to the south wall, so its direction is wS + t.
function foldFaces(wN, wE, wS, wW, t) {
  const cN = Math.cos(wN), sN = Math.sin(wN);
  const cE = Math.cos(wE), sE = Math.sin(wE);
  const cW = Math.cos(wW), sW = Math.sin(wW);
  const cS = Math.cos(wS), sS = Math.sin(wS);
  const ey = 2 + 2 * cS;
  const ez = 2 * sS;
  const cl = Math.cos(wS + t);
  const sl = Math.sin(wS + t);
  return [
    { key: "bottom", tint: 0.18, p: [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]] },
    { key: "north", tint: 0.13, p: [[0, 0, 0], [2, 0, 0], [2, -2 * cN, 2 * sN], [0, -2 * cN, 2 * sN]] },
    { key: "west", tint: 0.11, p: [[0, 0, 0], [-2 * cW, 0, 2 * sW], [-2 * cW, 2, 2 * sW], [0, 2, 0]] },
    { key: "east", tint: 0.09, p: [[2, 0, 0], [2 + 2 * cE, 0, 2 * sE], [2 + 2 * cE, 2, 2 * sE], [2, 2, 0]] },
    { key: "south", tint: 0.14, p: [[0, 2, 0], [2, 2, 0], [2, ey, ez], [0, ey, ez]] },
    { key: "top", tint: 0.24, p: [[0, ey, ez], [2, ey, ez], [2, ey + 2 * cl, ez + 2 * sl], [0, ey + 2 * cl, ez + 2 * sl]] },
  ];
}

const smoothstep = (x) => {
  const c = Math.max(0, Math.min(1, x));
  return c * c * (3 - 2 * c);
};
const seg = (p, a, b) => smoothstep((p - a) / (b - a));

// progress p in [0,1) -> [north, east, south, west, lid] angles. The top-left
// side (west) folds first, then clockwise: north, east, south; the lid closes
// last. Hold as a cube; then unfold in reverse. Loops.
function foldAngles(p) {
  const H = Math.PI / 2;
  return [
    H * (seg(p, 0.18, 0.28) - seg(p, 0.9, 0.95)),
    H * (seg(p, 0.3, 0.4) - seg(p, 0.85, 0.9)),
    H * (seg(p, 0.42, 0.52) - seg(p, 0.78, 0.84)),
    H * (seg(p, 0.06, 0.16) - seg(p, 0.94, 1.0)),
    H * (seg(p, 0.54, 0.63) - seg(p, 0.7, 0.77)),
  ];
}

function FoldArt() {
  const prefersReduced =
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [p, setP] = useState(prefersReduced ? 0.66 : 0);
  const rafRef = useRef(0);

  useEffect(() => {
    if (prefersReduced) return undefined;
    let startTs;
    const period = 10000;
    const tick = (ts) => {
      if (startTs === undefined) startTs = ts;
      setP(((ts - startTs) % period) / period);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [prefersReduced]);

  const [wN, wE, wS, wW, t] = foldAngles(p);
  const faces = foldFaces(wN, wE, wS, wW, t)
    .map((f) => ({
      key: f.key,
      tint: f.tint,
      pts: f.p.map(([x, y, z]) => fi(x, y, z)),
      depth: f.p.reduce((s, c) => s + c[0] + c[1] + c[2], 0), // iso painter's key
    }))
    .sort((a, b) => a.depth - b.depth);

  return (
    <svg viewBox="440 40 740 590" preserveAspectRatio="xMaxYMid meet" aria-hidden="true" className="hero-art-svg">
      <Glow id="hero-glow-me" cx={fi(1, 1, 0.6)[0]} cy={fi(1, 1, 0.6)[1]} r={360} />

      {/* isometric floor grid filling the hero, aligned to the cube base */}
      <g stroke="currentColor" strokeWidth="1" fill="none" opacity="0.1">
        {MI_GRID.map(([a, b], i) => (
          <line key={i} x1={a[0].toFixed(1)} y1={a[1].toFixed(1)} x2={b[0].toFixed(1)} y2={b[1].toFixed(1)} />
        ))}
      </g>

      {/* construction guides: floor + top-edge axes, full-height corner verticals */}
      <g stroke="currentColor" strokeWidth="1" fill="none" strokeDasharray="6 5">
        <g opacity="0.2">
          {MI_AXES.map(([a, b], i) => (
            <line key={i} x1={a[0].toFixed(1)} y1={a[1].toFixed(1)} x2={b[0].toFixed(1)} y2={b[1].toFixed(1)} />
          ))}
        </g>
        <g opacity="0.17">
          {MI_TOP_AXES.map(([a, b], i) => (
            <line key={i} x1={a[0].toFixed(1)} y1={a[1].toFixed(1)} x2={b[0].toFixed(1)} y2={b[1].toFixed(1)} />
          ))}
        </g>
        <g opacity="0.22">
          {MI_VERTS.map(([a, b], i) => (
            <line key={i} x1={a[0].toFixed(1)} y1={a[1].toFixed(1)} x2={b[0].toFixed(1)} y2={b[1].toFixed(1)} />
          ))}
        </g>
      </g>

      {/* folding cube faces with sketch hatching */}
      {faces.map((f) => {
        const [A, B, , D] = f.pts;
        const at = (u, v) => [A[0] + u * (B[0] - A[0]) + v * (D[0] - A[0]), A[1] + u * (B[1] - A[1]) + v * (D[1] - A[1])];
        const d = f.pts.map((c) => `${c[0].toFixed(1)},${c[1].toFixed(1)}`).join(" ");
        // closely-spaced grid texture that folds along with the face
        const texture = [];
        for (let k = 1; k <= 6; k += 1) {
          const g = k / 7;
          const a0 = at(0, g), a1 = at(1, g);
          const b0 = at(g, 0), b1 = at(g, 1);
          texture.push(<line key={`h${k}`} x1={a0[0].toFixed(1)} y1={a0[1].toFixed(1)} x2={a1[0].toFixed(1)} y2={a1[1].toFixed(1)} strokeWidth="0.6" strokeOpacity="0.18" />);
          texture.push(<line key={`v${k}`} x1={b0[0].toFixed(1)} y1={b0[1].toFixed(1)} x2={b1[0].toFixed(1)} y2={b1[1].toFixed(1)} strokeWidth="0.6" strokeOpacity="0.18" />);
        }
        return (
          <g key={f.key} stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round">
            <polygon points={d} fill="var(--surface-2)" stroke="none" />
            <polygon points={d} fill="currentColor" fillOpacity={f.tint} />
            <g strokeLinecap="round">{texture}</g>
          </g>
        );
      })}
    </svg>
  );
}

// --- isometric room scene (spatial page) ---
const ISO = { tw: 44, th: 23, tz: 50, ox: 890, oy: 380, fx: 17, fy: 22, wall: 18 };

function iso(gx, gy, gz = 0) {
  return [
    ISO.ox + (gx - gy) * ISO.tw,
    ISO.oy + (gx + gy) * ISO.th - gz * ISO.tz,
  ];
}

function pt(gx, gy, gz) {
  const [x, y] = iso(gx, gy, gz);
  return `${x.toFixed(1)},${y.toFixed(1)}`;
}

// --- isometric building blocks: a box and a cone, each projecting its
// vertices through iso(). A ground circle of radius r maps to an axis-aligned
// ellipse (rx = r*tw*√2, ry = r*th*√2) under this projection.
function IsoBox({ gx, gy, w, d, h, z0 = 0, tone = 1 }) {
  const t = z0 + h;
  const top = [pt(gx, gy, t), pt(gx + w, gy, t), pt(gx + w, gy + d, t), pt(gx, gy + d, t)].join(" ");
  const right = [pt(gx + w, gy, z0), pt(gx + w, gy + d, z0), pt(gx + w, gy + d, t), pt(gx + w, gy, t)].join(" ");
  const left = [pt(gx, gy + d, z0), pt(gx + w, gy + d, z0), pt(gx + w, gy + d, t), pt(gx, gy + d, t)].join(" ");
  return (
    <g stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round">
      <polygon points={left} fill="var(--surface-2)" stroke="none" />
      <polygon points={right} fill="var(--surface-2)" stroke="none" />
      <polygon points={top} fill="var(--surface-2)" stroke="none" />
      <polygon points={left} fill="currentColor" fillOpacity={0.14 * tone} />
      <polygon points={right} fill="currentColor" fillOpacity={0.07 * tone} />
      <polygon points={top} fill="currentColor" fillOpacity={0.24 * tone} />
    </g>
  );
}

function IsoCone({ gx, gy, w, d, h, z0 = 0 }) {
  const cxg = gx + w / 2;
  const cyg = gy + d / 2;
  const rg = Math.min(w, d) / 2;
  const cx = iso(cxg, cyg, z0)[0];
  const cyBase = iso(cxg, cyg, z0)[1];
  const cyApex = iso(cxg, cyg, z0 + h)[1];
  const rx = rg * ISO.tw * Math.SQRT2;
  const ry = rg * ISO.th * Math.SQRT2;
  const body = `M ${cx} ${cyApex} L ${cx - rx} ${cyBase} A ${rx} ${ry} 0 0 0 ${cx + rx} ${cyBase} Z`;
  return (
    <g stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round">
      <path d={body} fill="var(--surface-2)" stroke="none" />
      <path d={`M ${cx - rx} ${cyBase} A ${rx} ${ry} 0 0 1 ${cx + rx} ${cyBase}`} fill="none" strokeOpacity="0.5" />
      <path d={body} fill="currentColor" fillOpacity="0.17" />
    </g>
  );
}

// --- everyday furniture, composed from the primitives above (grid units).
// Sub-parts are ordered back -> front so occlusion reads correctly.
function Legs({ gx, gy, w, d, t, h }) {
  return (
    <>
      <IsoBox gx={gx} gy={gy} w={t} d={t} h={h} />
      <IsoBox gx={gx + w - t} gy={gy} w={t} d={t} h={h} />
      <IsoBox gx={gx} gy={gy + d - t} w={t} d={t} h={h} />
      <IsoBox gx={gx + w - t} gy={gy + d - t} w={t} d={t} h={h} />
    </>
  );
}

function Books({ gx, gy, z0 }) {
  return (
    <g>
      <IsoBox gx={gx} gy={gy} w={0.52} d={0.38} h={0.07} z0={z0} tone={1.3} />
      <IsoBox gx={gx + 0.03} gy={gy + 0.03} w={0.47} d={0.34} h={0.06} z0={z0 + 0.07} tone={0.8} />
      <IsoBox gx={gx + 0.09} gy={gy + 0.06} w={0.4} d={0.29} h={0.06} z0={z0 + 0.13} tone={1.5} />
    </g>
  );
}

function Table({ gx, gy, w, d, h }) {
  const t = 0.1;
  const topT = 0.09;
  const legH = h - topT;
  return (
    <g>
      <Legs gx={gx} gy={gy} w={w} d={d} t={t} h={legH} />
      <IsoBox gx={gx} gy={gy} w={w} d={d} h={topT} z0={legH} />
      <Books gx={gx + w * 0.26} gy={gy + d * 0.3} z0={h} />
    </g>
  );
}

function Chair({ gx, gy, w, d, h }) {
  const seatH = h * 0.52;
  const t = 0.09;
  return (
    <g>
      <IsoBox gx={gx} gy={gy} w={w} d={t} h={h - seatH + 0.05} z0={seatH} tone={0.9} />
      <Legs gx={gx} gy={gy} w={w} d={d} t={t} h={seatH} />
      <IsoBox gx={gx} gy={gy} w={w} d={d} h={0.08} z0={seatH - 0.08} />
    </g>
  );
}

function Sofa({ gx, gy, w, d, h }) {
  const baseH = h * 0.4;
  const arm = 0.26;
  const back = 0.28;
  return (
    <g>
      <IsoBox gx={gx} gy={gy} w={w} d={back} h={h} tone={0.9} />
      <IsoBox gx={gx} gy={gy + back} w={w} d={d - back} h={baseH} />
      <IsoBox gx={gx + arm} gy={gy + back} w={w - 2 * arm} d={d - back - 0.06} h={0.13} z0={baseH} tone={1.25} />
      <IsoBox gx={gx} gy={gy + back} w={arm} d={d - back} h={baseH + 0.24} />
      <IsoBox gx={gx + w - arm} gy={gy + back} w={arm} d={d - back} h={baseH + 0.24} />
    </g>
  );
}

function Bed({ gx, gy, w, d, h }) {
  const mattH = h * 0.52;
  const head = 0.18;
  return (
    <g>
      <IsoBox gx={gx} gy={gy} w={w} d={head} h={h} tone={0.9} />
      <IsoBox gx={gx} gy={gy + head} w={w} d={d - head} h={mattH} />
      <IsoBox gx={gx + 0.16} gy={gy + head + 0.08} w={w * 0.34} d={0.34} h={0.12} z0={mattH} tone={1.3} />
      <IsoBox gx={gx + w * 0.5} gy={gy + head + 0.08} w={w * 0.34} d={0.34} h={0.12} z0={mattH} tone={1.3} />
    </g>
  );
}

function Bookshelf({ gx, gy, w, d, h }) {
  const n = 4;
  const shelfLines = [];
  for (let i = 1; i < n; i += 1) {
    const z = (h * i) / n;
    shelfLines.push(
      <line key={`s${i}`} x1={iso(gx, gy + d, z)[0]} y1={iso(gx, gy + d, z)[1]} x2={iso(gx + w, gy + d, z)[0]} y2={iso(gx + w, gy + d, z)[1]} stroke="currentColor" strokeWidth="1" opacity="0.45" />,
    );
  }
  const spines = [];
  for (let s = 0; s < 3; s += 1) {
    const base = (h * s) / n;
    const shelfH = h / n;
    let cx = gx + 0.12;
    let k = 0;
    while (cx < gx + w - 0.18) {
      const bw = 0.1 + ((k * 17) % 5) / 55;
      const bh = shelfH * (0.6 + ((k * 29) % 30) / 100);
      spines.push(<IsoBox key={`sp${s}-${k}`} gx={cx} gy={gy + d - 0.16} w={bw} d={0.13} h={bh} z0={base} tone={1.25} />);
      cx += bw + 0.04;
      k += 1;
    }
  }
  return (
    <g>
      <IsoBox gx={gx} gy={gy} w={w} d={d} h={h} tone={0.85} />
      {shelfLines}
      {spines}
    </g>
  );
}

function Lamp({ gx, gy, w, d, h }) {
  const cx = gx + w / 2;
  const cy = gy + d / 2;
  const poleT = 0.07;
  const shadeH = h * 0.24;
  return (
    <g>
      <IsoBox gx={cx - 0.2} gy={cy - 0.2} w={0.4} d={0.4} h={0.06} />
      <IsoBox gx={cx - poleT / 2} gy={cy - poleT / 2} w={poleT} d={poleT} h={h - shadeH} tone={0.9} />
      <IsoCone gx={cx - w * 0.41} gy={cy - d * 0.41} w={w * 0.82} d={d * 0.82} h={shadeH} z0={h - shadeH} />
    </g>
  );
}

function Nightstand({ gx, gy, w, d, h }) {
  return (
    <g>
      <IsoBox gx={gx} gy={gy} w={w} d={d} h={h} />
      {[0.36, 0.68].map((f) => {
        const z = h * f;
        return <line key={f} x1={iso(gx, gy + d, z)[0]} y1={iso(gx, gy + d, z)[1]} x2={iso(gx + w, gy + d, z)[0]} y2={iso(gx + w, gy + d, z)[1]} stroke="currentColor" strokeWidth="1" opacity="0.5" />;
      })}
    </g>
  );
}

function Plant({ gx, gy, w, d, h }) {
  const cx = gx + w / 2;
  const cy = gy + d / 2;
  const potH = h * 0.32;
  const potR = Math.min(w, d) * 0.3;
  const folR = Math.min(w, d) * 0.5;
  const grow = h - potH;
  return (
    <g>
      <IsoBox gx={cx - potR} gy={cy - potR} w={potR * 2} d={potR * 2} h={potH} tone={1.1} />
      <IsoCone gx={cx - folR} gy={cy - folR} w={folR * 2} d={folR * 2} h={grow * 0.72} z0={potH} />
      <IsoCone gx={cx - folR * 0.68} gy={cy - folR * 0.68} w={folR * 1.36} d={folR * 1.36} h={grow * 0.55} z0={potH + grow * 0.34} />
    </g>
  );
}

function Rug({ gx, gy, w, d }) {
  const outer = [pt(gx, gy, 0), pt(gx + w, gy, 0), pt(gx + w, gy + d, 0), pt(gx, gy + d, 0)].join(" ");
  const m = 0.22;
  const inner = [pt(gx + m, gy + m, 0), pt(gx + w - m, gy + m, 0), pt(gx + w - m, gy + d - m, 0), pt(gx + m, gy + d - m, 0)].join(" ");
  const stripes = [];
  for (let i = 1; i <= 2; i += 1) {
    const t = gy + (d * i) / 3;
    stripes.push(<line key={`rs${i}`} x1={iso(gx + m, t, 0)[0]} y1={iso(gx + m, t, 0)[1]} x2={iso(gx + w - m, t, 0)[0]} y2={iso(gx + w - m, t, 0)[1]} stroke="currentColor" strokeWidth="1" strokeOpacity="0.32" />);
  }
  return (
    <g stroke="currentColor" strokeLinejoin="round">
      <polygon points={outer} fill="currentColor" fillOpacity="0.11" strokeWidth="1.5" strokeOpacity="0.85" />
      <polygon points={inner} fill="none" strokeWidth="1" strokeOpacity="0.5" />
      {stripes}
    </g>
  );
}

function WallArt({ wall, at, z, w, h }) {
  const P = (u, zz) => {
    const [x, y] = wall === "right" ? iso(u, 0, zz) : iso(0, u, zz);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  };
  const xy = (u, zz) => (wall === "right" ? iso(u, 0, zz) : iso(0, u, zz));
  const a = at - w / 2;
  const b = at + w / 2;
  const z1 = z - h / 2;
  const z2 = z + h / 2;
  const mw = w * 0.13;
  const mh = h * 0.13;
  const frame = [P(a, z1), P(b, z1), P(b, z2), P(a, z2)].join(" ");
  const canvas = [P(a + mw, z1 + mh), P(b - mw, z1 + mh), P(b - mw, z2 - mh), P(a + mw, z2 - mh)].join(" ");
  const [hx1, hy1] = xy(a + mw, z1 + h * 0.42);
  const [hx2, hy2] = xy(b - mw, z1 + h * 0.42);
  return (
    <g stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round">
      <polygon points={frame} fill="var(--surface-2)" stroke="none" />
      <polygon points={frame} fill="currentColor" fillOpacity="0.08" />
      <polygon points={canvas} fill="currentColor" fillOpacity="0.17" />
      <line x1={hx1} y1={hy1} x2={hx2} y2={hy2} strokeWidth="1" strokeOpacity="0.5" />
    </g>
  );
}

const ISO_SHAPES = { table: Table, chair: Chair, sofa: Sofa, bed: Bed, bookshelf: Bookshelf, lamp: Lamp, nightstand: Nightstand, plant: Plant };

// A believable studio layout. The right wall (gy≈0) is the "back": the bed
// (headboard to the wall), bookshelf and nightstand line up against it. The
// living zone sits in front — a sofa facing a coffee table on a rug, a floor
// lamp beside the sofa, an accent chair and a plant. Furniture "faces" +gy
// (toward the viewer), so the sofa looks onto the table.
const RUGS = [{ gx: 2.6, gy: 5.3, w: 4.8, d: 3.2 }];

// framed paintings hung on the two back walls (right wall = gy 0, left = gx 0).
const WALLART = [
  { wall: "right", at: 7.9, z: 4.0, w: 2.8, h: 2.6 },
  { wall: "right", at: 4.7, z: 4.0, w: 1.15, h: 1.5 },
  { wall: "right", at: 2.3, z: 4.0, w: 1.7, h: 2.0 },
  { wall: "left", at: 3.7, z: 4.0, w: 2.3, h: 2.2 },
];

const ROOM_OBJECTS = [
  { id: "shelf", gx: 1.3, gy: 0.35, w: 2.0, d: 0.6, h: 2.2, label: "SHELF", prob: 0.98, shape: "bookshelf" },
  { id: "plant3", gx: 3.9, gy: 0.4, w: 0.9, d: 0.7, h: 1.15, label: "PLANT", prob: 0.88, shape: "plant" },
  { id: "bed", gx: 6.4, gy: 0.35, w: 3.2, d: 2.4, h: 0.95, label: "BED", prob: 0.99, shape: "bed" },
  { id: "nightstand", gx: 9.8, gy: 0.4, w: 0.85, d: 0.95, h: 0.72, label: "STAND", prob: 0.9, shape: "nightstand" },
  { id: "plant2", gx: 0.6, gy: 1.6, w: 0.95, d: 0.95, h: 1.4, label: "PLANT", prob: 0.91, shape: "plant" },
  { id: "lamp", gx: 1.2, gy: 3.6, w: 0.85, d: 0.85, h: 2.3, label: "LAMP", prob: 0.93, shape: "lamp" },
  { id: "sofa", gx: 2.7, gy: 3.3, w: 3.2, d: 1.5, h: 1.05, label: "SOFA", prob: 0.97, shape: "sofa" },
  { id: "plant", gx: 6.6, gy: 3.4, w: 1.0, d: 1.0, h: 1.5, label: "PLANT", prob: 0.95, shape: "plant" },
  { id: "table", gx: 3.4, gy: 6.0, w: 2.0, d: 1.35, h: 0.6, label: "TABLE", prob: 0.96, shape: "table" },
  { id: "chair", gx: 5.9, gy: 5.8, w: 1.1, d: 1.1, h: 1.05, label: "CHAIR", prob: 0.94, shape: "chair" },
];

function SpaceArt() {
  const { fx, fy, wall: H } = ISO;
  const rightWallLen = fx;
  const leftWallLen = fy;
  // texture the wall to its full height so the grid recedes off the top of the
  // frame (cropped) instead of ending in a hard edge partway up.
  const wz = H;

  // textured floor: iso checkerboard tiles + grid lines, spanning the full hero
  const tiles = [];
  for (let gx = 0; gx < fx; gx += 1) {
    for (let gy = 0; gy < fy; gy += 1) {
      if ((gx + gy) % 2 === 0) {
        tiles.push(
          <polygon
            key={`t${gx}-${gy}`}
            points={[pt(gx, gy, 0), pt(gx + 1, gy, 0), pt(gx + 1, gy + 1, 0), pt(gx, gy + 1, 0)].join(" ")}
            fill="currentColor"
            fillOpacity="0.05"
            stroke="none"
          />,
        );
      }
    }
  }

  const floorLines = [];
  for (let gx = 0; gx <= fx; gx += 1) {
    floorLines.push(<line key={`fx${gx}`} x1={iso(gx, 0)[0]} y1={iso(gx, 0)[1]} x2={iso(gx, fy)[0]} y2={iso(gx, fy)[1]} />);
  }
  for (let gy = 0; gy <= fy; gy += 1) {
    floorLines.push(<line key={`fy${gy}`} x1={iso(0, gy)[0]} y1={iso(0, gy)[1]} x2={iso(fx, gy)[0]} y2={iso(fx, gy)[1]} />);
  }

  // room walls: right wall to the border, left wall extended to the far edge;
  // both are tall so their top edges sit above the frame (cropped off).
  const wallRight = [pt(0, 0, 0), pt(rightWallLen, 0, 0), pt(rightWallLen, 0, H), pt(0, 0, H)].join(" ");
  const wallLeft = [pt(0, 0, 0), pt(0, leftWallLen, 0), pt(0, leftWallLen, H), pt(0, 0, H)].join(" ");
  const wallLines = [];
  for (let z = 1; z < wz; z += 1) {
    wallLines.push(<line key={`wr${z}`} x1={iso(0, 0, z)[0]} y1={iso(0, 0, z)[1]} x2={iso(rightWallLen, 0, z)[0]} y2={iso(rightWallLen, 0, z)[1]} />);
    wallLines.push(<line key={`wl${z}`} x1={iso(0, 0, z)[0]} y1={iso(0, 0, z)[1]} x2={iso(0, leftWallLen, z)[0]} y2={iso(0, leftWallLen, z)[1]} />);
  }
  for (let gx = 1; gx < rightWallLen; gx += 1) {
    wallLines.push(<line key={`wrs${gx}`} x1={iso(gx, 0, 0)[0]} y1={iso(gx, 0, 0)[1]} x2={iso(gx, 0, wz)[0]} y2={iso(gx, 0, wz)[1]} />);
  }
  for (let gy = 1; gy < leftWallLen; gy += 1) {
    wallLines.push(<line key={`wls${gy}`} x1={iso(0, gy, 0)[0]} y1={iso(0, gy, 0)[1]} x2={iso(0, gy, wz)[0]} y2={iso(0, gy, wz)[1]} />);
  }

  const detectIdx = {};
  ROOM_OBJECTS.slice()
    .sort((a, b) => iso(a.gx + a.w / 2, a.gy + a.d / 2, 0)[0] - iso(b.gx + b.w / 2, b.gy + b.d / 2, 0)[0])
    .forEach((o, i) => { detectIdx[o.id] = i; });
  const anchored = ROOM_OBJECTS.map((o) => {
    const [ax, ay] = iso(o.gx + o.w / 2, o.gy + o.d / 2, o.h);
    return { ...o, ax, ay, di: detectIdx[o.id] };
  });

  return (
    <svg viewBox="0 0 1360 820" aria-hidden="true" className="hero-art-svg" preserveAspectRatio="xMaxYMax slice">
      <Glow id="hero-glow-spatial" cx={iso(2, 4)[0]} cy={iso(2, 4)[1]} r={420} />

      {/* textured floor */}
      <g>{tiles}</g>
      <g stroke="currentColor" strokeWidth="1" fill="none" opacity="0.26">{floorLines}</g>

      {/* room shell */}
      <g stroke="currentColor" strokeLinejoin="round">
        <polygon points={wallRight} fill="currentColor" fillOpacity="0.05" strokeWidth="1.3" strokeOpacity="0.5" />
        <polygon points={wallLeft} fill="currentColor" fillOpacity="0.09" strokeWidth="1.3" strokeOpacity="0.5" />
      </g>
      <g stroke="currentColor" strokeWidth="1" fill="none" opacity="0.24">{wallLines}</g>

      {/* framed wall art */}
      {WALLART.map((a, i) => (
        <WallArt key={`art${i}`} {...a} />
      ))}

      {/* rug — floor decal beneath the seating group */}
      {RUGS.map((r, i) => (
        <Rug key={`rug${i}`} gx={r.gx} gy={r.gy} w={r.w} d={r.d} />
      ))}

      {/* furniture + per-object detection flash, drawn back-to-front */}
      {anchored
        .slice()
        .sort((a, b) => a.gx + a.gy - (b.gx + b.gy))
        .map((o) => {
          const Shape = ISO_SHAPES[o.shape] || IsoBox;
          return (
            <g key={o.id}>
              <Shape gx={o.gx} gy={o.gy} w={o.w} d={o.d} h={o.h} />
              <g className="detect-flash" style={{ "--di": o.di }}>
                <Shape gx={o.gx} gy={o.gy} w={o.w} d={o.d} h={o.h} />
              </g>
            </g>
          );
        })}

      {/* "model output" detection overlay — one object at a time, looping */}
      {anchored.map((o) => {
        const cs = [];
        for (const X of [o.gx, o.gx + o.w]) for (const Y of [o.gy, o.gy + o.d]) for (const Z of [0, o.h]) cs.push(iso(X, Y, Z));
        const xs = cs.map((c) => c[0]);
        const ys = cs.map((c) => c[1]);
        const pad = 5;
        const x0 = Math.min(...xs) - pad;
        const y0 = Math.min(...ys) - pad;
        const bw = Math.max(...xs) - Math.min(...xs) + pad * 2;
        const bh = Math.max(...ys) - Math.min(...ys) + pad * 2;
        const tag = `${o.label} ${o.prob.toFixed(2)}`;
        const tw = tag.length * 6.3 + 12;
        return (
          <g key={`det-${o.id}`} className="detect-box" style={{ "--di": o.di }}>
            <rect className="detect-rect" x={x0} y={y0} width={bw} height={bh} rx="1" />
            <rect className="detect-tag" x={x0} y={y0 - 15} width={tw} height="15" rx="1" />
            <text className="detect-tag-text" x={x0 + 6} y={y0 - 7} dominantBaseline="middle" fontSize="10.5" letterSpacing="0.04em">{tag}</text>
          </g>
        );
      })}
    </svg>
  );
}

const ART = {
  dysm: PerceptionArt,
  minds_eye: FoldArt,
  spatial: SpaceArt,
};

export function HeroArt({ variant }) {
  const Art = ART[variant];
  if (!Art) return null;
  return (
    <div className="page-hero-art" data-variant={variant} aria-hidden="true">
      <Art />
    </div>
  );
}

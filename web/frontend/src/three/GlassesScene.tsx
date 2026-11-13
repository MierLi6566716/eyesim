import { Canvas, useFrame } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { useMemo, useRef, type ReactNode } from "react";
import * as THREE from "three";

/**
 * CAD-style model of the measuring prototype: adjustable (Alvarez) glasses plus
 * the clip-on refraction module in front of the right lens. 1 unit = 1 mm.
 * Dimensions are design intent, not a measured drawing: lens box 42x32 mm,
 * relay focal length 100 mm (module tube length), module bore 26 mm.
 *
 * `spread` (0..1) explodes the assembly; parts separate along the optical axis
 * (+z toward the scene camera) and the module internals fan out on +y.
 */

export type PartId =
  | "frame" | "plates" | "slider" | "temples"
  | "housing" | "relay" | "camera" | "leds" | "motor" | "encoder" | "mcu" | "battery";

export const PART_LABELS: Record<PartId, string> = {
  frame: "1 · Frame front",
  plates: "2 · Alvarez plate pair (×2)",
  slider: "3 · Plate slider tab",
  temples: "4 · Temples",
  housing: "5 · Clip-on housing",
  relay: "6 · Relay lens, f = 100 mm",
  camera: "7 · Camera at the relay focus",
  leds: "8 · 850 nm LEDs (2 mm off axis)",
  motor: "9 · Micro lead-screw motor",
  encoder: "10 · Position encoder strip",
  mcu: "11 · Controller board",
  battery: "12 · Battery",
};

const INK = "#111111";
const STEEL = "#8d8d8d";
const DARK = "#2e2e2e";
const GLASS = "#9db8e8";
const PCB = "#0a7a3d";
const ACCENT = "#0b3fae";
const LED_RED = "#c4161c";

function Mat({ color, glass = false, dim }: { color: string; glass?: boolean; dim?: boolean }) {
  return glass ? (
    <meshPhysicalMaterial color={color} transparent opacity={0.4} roughness={0.05} clearcoat={1} side={THREE.DoubleSide} />
  ) : (
    <meshStandardMaterial color={color} roughness={0.55} metalness={0.25} transparent={dim} opacity={dim ? 0.25 : 1} />
  );
}

type PartProps = { spread: number; selected: PartId | null; onPick: (p: PartId) => void };

function usePartGroup(home: THREE.Vector3, away: THREE.Vector3, spread: number) {
  const ref = useRef<THREE.Group>(null);
  useFrame((_, dt) => {
    if (!ref.current) return;
    const k = 1 - Math.exp(-dt * 8);
    const target = home.clone().lerp(away, spread);
    ref.current.position.lerp(target, k);
  });
  return ref;
}

function Part({
  id, home, away, spread, selected, onPick, children,
}: PartProps & { id: PartId; home: [number, number, number]; away: [number, number, number]; children: ReactNode }) {
  const ref = usePartGroup(new THREE.Vector3(...home), new THREE.Vector3(...away), spread);
  const dim = selected !== null && selected !== id;
  return (
    <group
      ref={ref}
      onClick={(e) => { e.stopPropagation(); onPick(id); }}
      onPointerOver={(e) => { e.stopPropagation(); document.body.style.cursor = "pointer"; }}
      onPointerOut={() => { document.body.style.cursor = "default"; }}
    >
      <group scale={selected === id ? 1.02 : 1}>{children}</group>
      {dim && <group />}
    </group>
  );
}

/** Alvarez plate: thin slab with a cubic front surface, as on the Alvarez page. */
function alvarezGeometry(sign: 1 | -1) {
  const n = 32, R = 19, A = 4 / (4 * 0.49) / 1000, SAG = 4;
  const g = new THREE.PlaneGeometry(2 * R, 1.6 * R, n, n);
  const pos = g.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i);
    pos.setZ(i, sign * (A * (x * y * y + (x * x * x) / 3)) * SAG);
  }
  g.computeVertexNormals();
  return g;
}

function LensStack({ x, sel }: { x: number; sel: boolean }) {
  const g1 = useMemo(() => alvarezGeometry(1), []);
  const g2 = useMemo(() => alvarezGeometry(-1), []);
  return (
    <group position={[x, 0, 0]}>
      <mesh geometry={g1} position={[0, 0, 1.4]}>
        <meshPhysicalMaterial color={GLASS} transparent opacity={sel ? 0.65 : 0.4} roughness={0.05} clearcoat={1} side={THREE.DoubleSide} />
      </mesh>
      <mesh geometry={g2} position={[0, 0, -1.4]}>
        <meshPhysicalMaterial color={GLASS} transparent opacity={sel ? 0.65 : 0.4} roughness={0.05} clearcoat={1} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

function FrameFront() {
  // Two rounded-rectangle rims joined by a bridge, extruded 3 mm.
  const shape = useMemo(() => {
    const s = new THREE.Shape();
    const rim = (cx: number) => {
      const w = 23, h = 17, r = 7;
      s.moveTo(cx - w + r, h);
      s.lineTo(cx + w - r, h); s.absarc(cx + w - r, h - r, r, Math.PI / 2, 0, true);
      s.lineTo(cx + w, -h + r); s.absarc(cx + w - r, -h + r, r, 0, -Math.PI / 2, true);
      s.lineTo(cx - w + r, -h); s.absarc(cx - w + r, -h + r, r, -Math.PI / 2, Math.PI, true);
      s.lineTo(cx - w, h - r); s.absarc(cx - w + r, h - r, r, Math.PI, Math.PI / 2, true);
      const hole = new THREE.Path();
      const w2 = 21, h2 = 15, r2 = 6;
      hole.moveTo(cx - w2 + r2, h2);
      hole.lineTo(cx + w2 - r2, h2); hole.absarc(cx + w2 - r2, h2 - r2, r2, Math.PI / 2, 0, true);
      hole.lineTo(cx + w2, -h2 + r2); hole.absarc(cx + w2 - r2, -h2 + r2, r2, 0, -Math.PI / 2, true);
      hole.lineTo(cx - w2 + r2, -h2); hole.absarc(cx - w2 + r2, -h2 + r2, r2, -Math.PI / 2, Math.PI, true);
      hole.lineTo(cx - w2, h2 - r2); hole.absarc(cx - w2 + r2, h2 - r2, r2, Math.PI, Math.PI / 2, true);
      s.holes.push(hole);
    };
    rim(-31); rim(31);
    return s;
  }, []);
  return (
    <group>
      <mesh geometry={useMemo(() => new THREE.ExtrudeGeometry(shape, { depth: 3, bevelEnabled: false }), [shape])} position={[0, 0, -1.5]}>
        <Mat color={INK} />
      </mesh>
      <mesh position={[0, 4, 0]}>
        <boxGeometry args={[18, 3.4, 3]} />
        <Mat color={INK} />
      </mesh>
    </group>
  );
}

function Module({ spread, selected, onPick }: PartProps) {
  // Tube pointing forward (+z) from the right lens (x = +31). Length 112 mm.
  const x = 31;
  const sel = (id: PartId) => selected === id;
  return (
    <group position={[x, 0, 0]}>
      <Part id="housing" home={[0, 0, 60]} away={[0, 26, 60]} spread={spread} selected={selected} onPick={onPick}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[14, 14, 112, 28, 1, true]} />
          <meshStandardMaterial color={DARK} roughness={0.7} side={THREE.DoubleSide} transparent opacity={sel("housing") ? 0.85 : 0.35} />
        </mesh>
      </Part>
      <Part id="relay" home={[0, 0, 10]} away={[0, -34, 10]} spread={spread} selected={selected} onPick={onPick}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[11, 11, 4, 32]} />
          <Mat color={GLASS} glass />
        </mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[11.4, 1, 10, 32]} />
          <Mat color={STEEL} />
        </mesh>
      </Part>
      <Part id="camera" home={[0, 0, 110]} away={[0, 0, 150]} spread={spread} selected={selected} onPick={onPick}>
        <mesh>
          <boxGeometry args={[24, 24, 2]} />
          <Mat color={PCB} />
        </mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -3]}>
          <cylinderGeometry args={[4, 4, 6, 24]} />
          <Mat color={INK} />
        </mesh>
      </Part>
      <Part id="leds" home={[0, 0, 106]} away={[0, -30, 130]} spread={spread} selected={selected} onPick={onPick}>
        {[[2.0, 0], [0, 2.0]].map(([lx, ly], i) => (
          <mesh key={i} position={[lx === 0 ? 0 : 6.5, ly === 0 ? 0 : 6.5, 0]}>
            <boxGeometry args={[2.6, 2.6, 1.4]} />
            <meshStandardMaterial color={LED_RED} emissive={LED_RED} emissiveIntensity={sel("leds") ? 1.4 : 0.5} />
          </mesh>
        ))}
      </Part>
      <Part id="mcu" home={[0, -17.5, 70]} away={[0, -46, 90]} spread={spread} selected={selected} onPick={onPick}>
        <mesh>
          <boxGeometry args={[18, 2, 42]} />
          <Mat color={PCB} />
        </mesh>
      </Part>
      <Part id="battery" home={[0, -17.5, 28]} away={[0, -60, 40]} spread={spread} selected={selected} onPick={onPick}>
        <mesh>
          <boxGeometry args={[16, 5, 24]} />
          <Mat color={STEEL} />
        </mesh>
      </Part>
    </group>
  );
}

function Rays({ visible }: { visible: boolean }) {
  // Measurement path: eye (behind lens) -> relay -> camera.
  if (!visible) return null;
  const pts: [number, number, number][][] = [-4, 0, 4].map((h) => [
    [31 + h, 0, -14],
    [31 + h, 0, 10],
    [31, 0, 110],
  ]);
  return (
    <group>
      {pts.map((p, i) => (
        <Line key={i} points={p} color={ACCENT} lineWidth={1.4} dashed dashSize={2.4} gapSize={1.6} />
      ))}
    </group>
  );
}

export function GlassesScene({
  spread, selected, onPick, showRays,
}: { spread: number; selected: PartId | null; onPick: (p: PartId | null) => void; showRays: boolean }) {
  return (
    <Canvas camera={{ position: [95, 55, 150], fov: 34 }} dpr={[1, 2]} onPointerMissed={() => onPick(null)}>
      <color attach="background" args={["#ffffff"]} />
      <ambientLight intensity={0.85} />
      <directionalLight position={[60, 120, 90]} intensity={1.2} />
      <directionalLight position={[-80, -40, -60]} intensity={0.35} />
      <group position={[0, 0, -20]}>
        <Part id="frame" home={[0, 0, 0]} away={[0, 0, -34]} spread={spread} selected={selected} onPick={onPick}>
          <FrameFront />
        </Part>
        <Part id="plates" home={[0, 0, 0]} away={[0, 0, 18]} spread={spread} selected={selected} onPick={onPick}>
          <LensStack x={-31} sel={selected === "plates"} />
          <LensStack x={31} sel={selected === "plates"} />
        </Part>
        <Part id="slider" home={[31, -19, 0]} away={[31, -40, 14]} spread={spread} selected={selected} onPick={onPick}>
          <mesh>
            <boxGeometry args={[10, 4, 6]} />
            <Mat color={ACCENT} />
          </mesh>
        </Part>
        <Part id="motor" home={[47, -16, 6]} away={[66, -34, 18]} spread={spread} selected={selected} onPick={onPick}>
          <mesh rotation={[0, 0, Math.PI / 2]}>
            <cylinderGeometry args={[3.2, 3.2, 14, 20]} />
            <Mat color={STEEL} />
          </mesh>
          <mesh rotation={[0, 0, Math.PI / 2]} position={[-12, 0, 0]}>
            <cylinderGeometry args={[1, 1, 12, 12]} />
            <Mat color={INK} />
          </mesh>
        </Part>
        <Part id="encoder" home={[31, -23, 1]} away={[31, -52, 26]} spread={spread} selected={selected} onPick={onPick}>
          <mesh>
            <boxGeometry args={[26, 2, 0.8]} />
            <Mat color={INK} />
          </mesh>
        </Part>
        <Part id="temples" home={[0, 0, 0]} away={[0, 34, -14]} spread={spread} selected={selected} onPick={onPick}>
          {[-54, 54].map((tx) => (
            <mesh key={tx} position={[tx, 10, -60]} rotation={[0, 0, 0]}>
              <boxGeometry args={[2.6, 4, 120]} />
              <Mat color={INK} />
            </mesh>
          ))}
        </Part>
        <Module spread={spread} selected={selected} onPick={onPick} />
        <Rays visible={showRays} />
      </group>
      <OrbitControls enablePan={false} minDistance={80} maxDistance={320} />
    </Canvas>
  );
}

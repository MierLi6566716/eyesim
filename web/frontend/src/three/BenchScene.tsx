import { Canvas, useFrame } from "@react-three/fiber";
import { Html, Line, OrbitControls } from "@react-three/drei";
import { useMemo, useRef, type ReactNode } from "react";
import * as THREE from "three";
import styles from "./BenchScene.module.css";

/**
 * Model-eye bench (Option B) at 1 scene unit = 10 mm. Optical axis is +x, from
 * the camera toward the "retina" screen. Positions follow the proposal: camera
 * and LEDs at the working distance, then cylinder trial lens on a rotation stage,
 * iris, 50 mm "cornea" lens, and the diffuser screen on a lead-screw rail.
 */

export type BenchProps = {
  f0mm: number;
  dxMm: number;          // screen offset from the emmetropic position
  axisDeg: number;       // cylinder axis (rotation-stage angle)
  hasCyl: boolean;
  pupilMm: number;       // iris radius
  firing: "h" | "v" | null;
  labels: boolean;
  dark: boolean;
};

const U = 0.1; // units per mm
const LENS_X = 30;
const CAM_X = 0;
const AXIS_Y = 4.2;

function Label({ children, position, show }: { children: ReactNode; position: [number, number, number]; show: boolean }) {
  if (!show) return null;
  return (
    <Html position={position} center distanceFactor={26} zIndexRange={[10, 0]}>
      <div className={styles.label}>{children}</div>
    </Html>
  );
}

function holeTexture(dark: boolean) {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  g.fillStyle = dark ? "#2a2c30" : "#3b3f46";
  g.fillRect(0, 0, 64, 64);
  g.fillStyle = dark ? "#121315" : "#1d1f23";
  g.beginPath();
  g.arc(32, 32, 7, 0, Math.PI * 2);
  g.fill();
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(46, 18);
  t.anisotropy = 8;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function Post({ x, z = 0, h = AXIS_Y - 0.5 }: { x: number; z?: number; h?: number }) {
  return (
    <group position={[x, 0, z]}>
      <mesh position={[0, h / 2, 0]}>
        <cylinderGeometry args={[0.32, 0.32, h, 24]} />
        <meshStandardMaterial color="#b9bcc2" metalness={0.8} roughness={0.28} />
      </mesh>
      <mesh position={[0, 0.2, 0]}>
        <cylinderGeometry args={[0.7, 0.7, 0.4, 24]} />
        <meshStandardMaterial color="#26282c" metalness={0.5} roughness={0.5} />
      </mesh>
    </group>
  );
}

/** Biconvex singlet built by lathing a lens profile. */
function LensMesh({ radius, thickness }: { radius: number; thickness: number }) {
  const geo = useMemo(() => {
    const pts: THREE.Vector2[] = [];
    const sag = (r: number) => (thickness / 2) * (1 - (r * r) / (radius * radius)) + 0.04;
    for (let i = 0; i <= 24; i++) {
      const r = (i / 24) * radius;
      pts.push(new THREE.Vector2(r, sag(r)));
    }
    for (let i = 24; i >= 0; i--) {
      const r = (i / 24) * radius;
      pts.push(new THREE.Vector2(r, -sag(r)));
    }
    return new THREE.LatheGeometry(pts, 64);
  }, [radius, thickness]);
  return (
    <mesh geometry={geo} rotation={[0, 0, Math.PI / 2]}>
      <meshPhysicalMaterial color="#cfe3ff" transparent opacity={0.45} roughness={0.05} metalness={0} clearcoat={1} />
    </mesh>
  );
}

function Ring({ r, t = 0.18, color = "#2b2d31" }: { r: number; t?: number; color?: string }) {
  return (
    <mesh rotation={[0, Math.PI / 2, 0]}>
      <torusGeometry args={[r, t, 16, 64]} />
      <meshStandardMaterial color={color} metalness={0.6} roughness={0.35} />
    </mesh>
  );
}

function Leds({ firing }: { firing: BenchProps["firing"] }) {
  const ecc = 10 * U;
  const led = (on: boolean, pos: [number, number, number]) => (
    <mesh position={pos}>
      <sphereGeometry args={[0.16, 16, 16]} />
      <meshStandardMaterial color={on ? "#ff5a3c" : "#5a2a22"} emissive={on ? "#ff3b1f" : "#000"} emissiveIntensity={on ? 3 : 0} />
    </mesh>
  );
  return (
    <group position={[CAM_X + 1.25, AXIS_Y, 0]}>
      {led(firing === "h", [0, 0, ecc])}
      {led(firing === "v", [0, ecc, 0])}
      {firing && (
        <pointLight position={firing === "h" ? [0.3, 0, ecc] : [0.3, ecc, 0]} color="#ff4020" intensity={1.4} distance={6} />
      )}
    </group>
  );
}

function Scene(p: BenchProps) {
  const tex = useMemo(() => holeTexture(p.dark), [p.dark]);
  const f0 = p.f0mm * U;
  const screenX = LENS_X + f0 + p.dxMm * U;
  const carriage = useRef<THREE.Group>(null);
  const cylMount = useRef<THREE.Group>(null);
  const screwMarks = useRef<THREE.Mesh>(null);

  // Ease the moving parts toward their commanded positions, like a stepper would.
  useFrame((_, dt) => {
    const k = 1 - Math.exp(-dt * 6);
    if (carriage.current) {
      const x = carriage.current.position.x;
      carriage.current.position.x = x + (screenX - x) * k;
      if (screwMarks.current) screwMarks.current.rotation.x = (carriage.current.position.x * 2 * Math.PI) / (8 * U);
    }
    if (cylMount.current) {
      const target = (p.axisDeg * Math.PI) / 180;
      const r = cylMount.current.rotation.x;
      cylMount.current.rotation.x = r + (target - r) * k;
    }
  });

  const pupil = p.pupilMm * U;
  const railStart = LENS_X + f0 - 2.2, railEnd = LENS_X + f0 + 3.6;

  // Rays: collimated beam through the iris → focus at f0 → on to the screen.
  const rays = useMemo(() => {
    const out: THREE.Vector3[][] = [];
    const n = 8;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      const y = Math.sin(a) * pupil * 0.92, z = Math.cos(a) * pupil * 0.92;
      const start = new THREE.Vector3(CAM_X + 3, AXIS_Y + y, z);
      const atLens = new THREE.Vector3(LENS_X, AXIS_Y + y, z);
      const t = (screenX - LENS_X) / f0; // fraction of the way to focus
      const end = new THREE.Vector3(screenX - 0.03, AXIS_Y + y * (1 - t), z * (1 - t));
      out.push([start, atLens, end]);
    }
    return out;
  }, [pupil, screenX, f0]);
  const spotR = Math.max(Math.abs(pupil * (1 - (screenX - LENS_X) / f0)), 0.02);

  return (
    <>
      <color attach="background" args={[p.dark ? "#1e1e1e" : "#f0f0f0"]} />
      <hemisphereLight args={[p.dark ? "#9aa4b8" : "#ffffff", p.dark ? "#1a1a1a" : "#c4c4c4", p.dark ? 0.9 : 1.2]} />
      <directionalLight position={[10, 25, 14]} intensity={p.dark ? 1.6 : 2.2} />
      <directionalLight position={[-12, 8, -10]} intensity={0.5} />

      {/* breadboard */}
      <mesh position={[21.5, -0.25, 0]}>
        <boxGeometry args={[46, 0.5, 18]} />
        <meshStandardMaterial map={tex} metalness={0.6} roughness={0.55} />
      </mesh>

      {/* camera + IR LEDs */}
      <group position={[CAM_X, AXIS_Y, 0]}>
        <mesh>
          <boxGeometry args={[1.6, 1.8, 2.2]} />
          <meshStandardMaterial color="#1e2024" metalness={0.3} roughness={0.5} />
        </mesh>
        <mesh position={[1.0, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.55, 0.55, 0.6, 32]} />
          <meshStandardMaterial color="#0d0e10" metalness={0.5} roughness={0.3} />
        </mesh>
      </group>
      <Post x={CAM_X} />
      <Leds firing={p.firing} />
      <Label position={[CAM_X, AXIS_Y + 2.1, 0]} show={p.labels}>NoIR camera + 850 nm LEDs</Label>

      {/* cylinder trial lens on rotation stage */}
      <group position={[LENS_X - 2.4, AXIS_Y, 0]}>
        <Ring r={1.45} t={0.3} color="#34363b" />
        <group ref={cylMount}>
          <mesh rotation={[0, 0, Math.PI / 2]} visible={p.hasCyl}>
            <cylinderGeometry args={[1.2, 1.2, 0.12, 48]} />
            <meshPhysicalMaterial color="#ffd9c7" transparent opacity={0.4} roughness={0.1} />
          </mesh>
          <mesh position={[0.08, 0, 0]} visible={p.hasCyl}>
            <boxGeometry args={[0.02, 2.3, 0.07]} />
            <meshBasicMaterial color="#c4161c" />
          </mesh>
        </group>
      </group>
      <Post x={LENS_X - 2.4} />
      <mesh position={[LENS_X - 2.4, 1.3, -1.8]}>
        <boxGeometry args={[1.2, 1.4, 1.2]} />
        <meshStandardMaterial color="#1f2125" />
      </mesh>
      <Label position={[LENS_X - 2.4, AXIS_Y + 2.3, 0]} show={p.labels}>Cylinder lens · rotation stage</Label>

      {/* iris */}
      <group position={[LENS_X - 0.9, AXIS_Y, 0]}>
        <mesh rotation={[0, Math.PI / 2, 0]}>
          <ringGeometry args={[pupil, 1.35, 48]} />
          <meshStandardMaterial color="#17181b" side={THREE.DoubleSide} metalness={0.4} roughness={0.6} />
        </mesh>
        <Ring r={1.4} t={0.12} />
      </group>
      <Label position={[LENS_X - 0.9, AXIS_Y - 2.0, 0]} show={p.labels}>Iris (pupil)</Label>

      {/* cornea lens */}
      <group position={[LENS_X, AXIS_Y, 0]}>
        <LensMesh radius={1.2} thickness={0.5} />
        <Ring r={1.25} t={0.1} color="#8c9097" />
      </group>
      <Post x={LENS_X} />
      <Label position={[LENS_X, AXIS_Y + 2.1, 0]} show={p.labels}>“Cornea” lens, f₀ = {p.f0mm} mm</Label>

      {/* rail, lead screw, stepper */}
      <mesh position={[(railStart + railEnd) / 2, 1.3, 0]}>
        <boxGeometry args={[railEnd - railStart, 0.5, 1.2]} />
        <meshStandardMaterial color="#2a2c30" metalness={0.7} roughness={0.3} />
      </mesh>
      <mesh ref={screwMarks} position={[(railStart + railEnd) / 2, 1.85, 0]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.08, 0.08, railEnd - railStart, 12]} />
        <meshStandardMaterial color="#c9a45a" metalness={0.9} roughness={0.25} />
      </mesh>
      <mesh position={[railEnd + 0.9, 1.9, 0]}>
        <boxGeometry args={[1.6, 1.6, 1.6]} />
        <meshStandardMaterial color="#1b1c1f" metalness={0.4} roughness={0.5} />
      </mesh>
      <Label position={[railEnd + 0.9, 3.5, 0]} show={p.labels}>NEMA17 + T8 lead screw</Label>

      {/* carriage + diffuser screen */}
      <group ref={carriage} position={[screenX, 0, 0]}>
        <mesh position={[0, 1.75, 0]}>
          <boxGeometry args={[0.9, 0.4, 1.6]} />
          <meshStandardMaterial color="#5c6068" metalness={0.6} roughness={0.35} />
        </mesh>
        <mesh position={[0.1, 2.6, 0]}>
          <boxGeometry args={[0.12, 1.4, 0.4]} />
          <meshStandardMaterial color="#5c6068" />
        </mesh>
        <mesh position={[0, AXIS_Y, 0]}>
          <boxGeometry args={[0.1, 2.6, 2.6]} />
          <meshStandardMaterial color="#f5f5f5" roughness={0.95} />
        </mesh>
        <mesh position={[-0.06, AXIS_Y, 0]} rotation={[0, -Math.PI / 2, 0]}>
          <circleGeometry args={[spotR, 40]} />
          <meshBasicMaterial color="#0b3fae" transparent opacity={0.55} />
        </mesh>
      </group>
      <Label position={[screenX, AXIS_Y + 2.2, 0]} show={p.labels}>PTFE “retina” on carriage</Label>

      {rays.map((pts, i) => (
        <Line key={i} points={pts} color={p.dark ? "#7aa7ff" : "#0b3fae"} lineWidth={1.2} transparent opacity={0.75} />
      ))}

      <OrbitControls
        target={[LENS_X - 2, AXIS_Y - 0.5, 0]}
        enablePan
        minDistance={6}
        maxDistance={60}
        maxPolarAngle={Math.PI / 2 - 0.05}
      />
    </>
  );
}

export function BenchScene(props: BenchProps) {
  return (
    <Canvas camera={{ position: [18, 13, 17], fov: 38 }} dpr={[1, 2]}>
      <Scene {...props} />
    </Canvas>
  );
}

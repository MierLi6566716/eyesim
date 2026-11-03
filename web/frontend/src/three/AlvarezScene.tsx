import { Canvas, useFrame } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";

/**
 * Two complementary Alvarez plates (1 unit = 1 mm, sag exaggerated for visibility).
 * The optical axis is +z. Plate 1 carries +t(x,y), plate 2 carries −t(x,y); they
 * translate in opposite directions. A bundle of collimated rays shows the resulting
 * power; ray angles are exaggerated so ±6 D fits on screen.
 */

export type AlvarezSceneProps = {
  kind: "sphere" | "astig";
  dx: number; // mm, per-plate shift along x
  dy: number; // mm, per-plate shift along y (astig pair only)
  M: number;
  J0: number;
  J45: number;
  dark: boolean;
};

const R = 8;          // plate half-width (mm)
const SAG_X = 7;      // visual exaggeration of the cubic sag
const A = 4 / (4 * 0.49) / 1000;

function surface(kind: AlvarezSceneProps["kind"], x: number, y: number) {
  return kind === "sphere" ? A * (x * y * y + (x * x * x) / 3) : A * ((x * x * x) / 3 - x * y * y);
}

function plateGeometry(kind: AlvarezSceneProps["kind"], sign: 1 | -1) {
  const n = 48;
  const g = new THREE.PlaneGeometry(2 * R, 2 * R, n, n);
  const pos = g.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i);
    pos.setZ(i, sign * surface(kind, x, y) * SAG_X);
  }
  g.computeVertexNormals();
  return g;
}

function Plate({ kind, sign, shift, z, color }: {
  kind: AlvarezSceneProps["kind"]; sign: 1 | -1; shift: [number, number]; z: number; color: string;
}) {
  const geo = useMemo(() => plateGeometry(kind, sign), [kind, sign]);
  const ref = useRef<THREE.Group>(null);
  useFrame((_, dt) => {
    if (!ref.current) return;
    const k = 1 - Math.exp(-dt * 10);
    ref.current.position.x += (shift[0] - ref.current.position.x) * k;
    ref.current.position.y += (shift[1] - ref.current.position.y) * k;
  });
  return (
    <group ref={ref} position={[0, 0, z]}>
      <mesh geometry={geo}>
        <meshPhysicalMaterial color={color} transparent opacity={0.55} roughness={0.08} clearcoat={1} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, 0, -sign * 0.9]}>
        <planeGeometry args={[2 * R, 2 * R]} />
        <meshPhysicalMaterial color={color} transparent opacity={0.25} roughness={0.1} side={THREE.DoubleSide} />
      </mesh>
      <lineSegments position={[0, 0, -sign * 0.45]}>
        <edgesGeometry args={[new THREE.BoxGeometry(2 * R, 2 * R, 0.9)]} />
        <lineBasicMaterial color={color} transparent opacity={0.6} />
      </lineSegments>
    </group>
  );
}

function Rays({ M, J0, J45, color }: { M: number; J0: number; J45: number; color: string }) {
  const lines = useMemo(() => {
    const out: THREE.Vector3[][] = [];
    const K = 0.008; // exaggeration: ray slope per dioptre per mm of height
    for (const hx of [-4.5, -1.5, 1.5, 4.5]) {
      for (const hy of [-4.5, -1.5, 1.5, 4.5]) {
        // Power-vector matrix acting on ray height gives the deflection.
        const sx = -((M + J0) * hx + J45 * hy) * K;
        const sy = -((M - J0) * hy + J45 * hx) * K;
        const L = 26;
        out.push([
          new THREE.Vector3(hx, hy, -14),
          new THREE.Vector3(hx, hy, 1.5),
          new THREE.Vector3(hx + sx * L, hy + sy * L, 1.5 + L),
        ]);
      }
    }
    return out;
  }, [M, J0, J45]);
  return <>{lines.map((p, i) => <Line key={i} points={p} color={color} lineWidth={1.1} transparent opacity={0.7} />)}</>;
}

export function AlvarezScene(p: AlvarezSceneProps) {
  const blue = p.dark ? "#7aa7ff" : "#0b3fae";
  return (
    <Canvas camera={{ position: [30, 18, 30], fov: 36 }} dpr={[1, 2]}>
      <color attach="background" args={[p.dark ? "#1e1e1e" : "#f0f0f0"]} />
      <hemisphereLight args={["#ffffff", p.dark ? "#1a1a1a" : "#c4c4c4", p.dark ? 1 : 1.3]} />
      <directionalLight position={[10, 20, 15]} intensity={2} />
      <Plate kind={p.kind} sign={1} shift={[p.dx, p.dy]} z={-0.6} color={p.dark ? "#9dbcff" : "#4f78d6"} />
      <Plate kind={p.kind} sign={-1} shift={[-p.dx, -p.dy]} z={0.6} color={p.dark ? "#ff9a92" : "#d9534f"} />
      <Rays M={p.M} J0={p.J0} J45={p.J45} color={blue} />
      <mesh position={[0, 0, 27.5]}>
        <planeGeometry args={[18, 18]} />
        <meshBasicMaterial color={p.dark ? "#2a2a2a" : "#d8d8d8"} side={THREE.DoubleSide} transparent opacity={0.6} />
      </mesh>
      <OrbitControls target={[0, 0, 6]} minDistance={15} maxDistance={90} />
    </Canvas>
  );
}

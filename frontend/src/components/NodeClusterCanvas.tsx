import React, { useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Line, Html } from '@react-three/drei';
import * as THREE from 'three';
import { NodeTelemetry } from '../types/telemetry';
import { Maximize2, RotateCcw, Eye, Compass } from 'lucide-react';

interface NodeClusterProps {
  nodes: NodeTelemetry[];
  selectedNode: NodeTelemetry | null;
  onSelectNode: (node: NodeTelemetry) => void;
}

const NodeSphere: React.FC<{
  node: NodeTelemetry;
  isSelected: boolean;
  onSelect: () => void;
}> = ({ node, isSelected, onSelect }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  const isSybil = node.role === 'sybil';
  const isSlashed = node.status === 'Slashed';

  // Aesthetic color assignments
  let color = '#10b981'; // emerald
  let emissiveColor = '#047857';

  if (isSybil) {
    if (isSlashed) {
      color = '#475569'; // charcoal slate
      emissiveColor = '#b91c1c'; // subdued red core
    } else {
      color = '#f43f5e'; // glowing rose
      emissiveColor = '#e11d48';
    }
  }

  // Floating animation and pulsation
  useFrame((state) => {
    if (meshRef.current) {
      const t = state.clock.elapsedTime;
      meshRef.current.position.y =
        node.position[1] + Math.sin(t * 1.6 + node.position[0]) * 0.08;

      if (ringRef.current && isSybil && !isSlashed) {
        ringRef.current.rotation.z = t * 2.0;
        const s = 1.0 + Math.sin(t * 4.0) * 0.15;
        ringRef.current.scale.set(s, s, s);
      }
    }
  });

  return (
    <group position={[node.position[0], node.position[1], node.position[2]]}>
      {/* Outer Warning Ring for Sybils */}
      {isSybil && !isSlashed && (
        <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.32, 0.36, 32]} />
          <meshBasicMaterial color="#f43f5e" side={THREE.DoubleSide} transparent opacity={0.6} />
        </mesh>
      )}

      {/* Main Node Mesh */}
      <mesh
        ref={meshRef}
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        scale={isSelected ? 1.4 : hovered ? 1.25 : 1.0}
      >
        <sphereGeometry args={[0.22, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={emissiveColor}
          emissiveIntensity={hovered || isSelected ? 1.0 : isSybil && !isSlashed ? 0.7 : 0.4}
          roughness={0.15}
          metalness={0.85}
          wireframe={isSlashed}
        />
      </mesh>

      {/* Collinear Attack Line for Active Sybils */}
      {isSybil && !isSlashed && (
        <Line
          points={[
            [0, 0, 0],
            [1.2, 0.35, 0.15]
          ]}
          color="#f43f5e"
          lineWidth={2.5}
        />
      )}

      {/* Floating 3D Tooltip */}
      {(hovered || isSelected) && (
        <Html distanceFactor={11} position={[0, 0.48, 0]}>
          <div className="bg-[#0b101f]/95 backdrop-blur-md border border-cyan-500/40 text-xs p-3 rounded-xl shadow-[0_0_20px_rgba(0,0,0,0.8)] pointer-events-none whitespace-nowrap font-mono z-50">
            <div className="flex items-center gap-2 border-b border-slate-800 pb-1.5 mb-1.5">
              <span className={`w-2 h-2 rounded-full ${isSybil ? 'bg-rose-500 animate-ping' : 'bg-emerald-400'}`}></span>
              <p className="text-cyan-300 font-bold">{node.address.slice(0, 10)}...{node.address.slice(-6)}</p>
            </div>
            <div className="space-y-1 text-[11px]">
              <p className="flex justify-between gap-4">
                <span className="text-slate-400">Classification:</span>
                <span className={isSybil ? 'text-rose-400 font-bold' : 'text-emerald-400 font-bold'}>
                  {node.role.toUpperCase()} ({node.status})
                </span>
              </p>
              {node.domain && (
                <p className="flex justify-between gap-4">
                  <span className="text-slate-400">Domain:</span>
                  <span className="text-slate-200">{node.domain}</span>
                </p>
              )}
              <p className="flex justify-between gap-4">
                <span className="text-slate-400">SVD Projection:</span>
                <span className="text-cyan-300 font-semibold">{node.projection_score.toFixed(4)}</span>
              </p>
              <p className="flex justify-between gap-4">
                <span className="text-slate-400">Collateral:</span>
                <span className="text-yellow-400 font-semibold">{node.staked_amount} DLRA</span>
              </p>
            </div>
          </div>
        </Html>
      )}
    </group>
  );
};

export const NodeClusterCanvas: React.FC<NodeClusterProps> = ({ nodes, selectedNode, onSelectNode }) => {
  const controlsRef = useRef<any>(null);

  // Spatial auto-normalization so coordinates are always comfortably in view [-3.2, 3.2]
  const maxCoord = Math.max(
    ...nodes.flatMap(n => [
      Math.abs(n.position?.[0] || 0),
      Math.abs(n.position?.[1] || 0),
      Math.abs(n.position?.[2] || 0)
    ]),
    1e-6
  );
  const scale = maxCoord > 3.2 ? 3.2 / maxCoord : 1.0;

  const normalizedNodes = nodes.map(n => ({
    ...n,
    position: [
      (n.position?.[0] || 0) * scale,
      (n.position?.[1] || 0) * scale,
      (n.position?.[2] || 0) * scale
    ] as [number, number, number]
  }));

  // Sybil cluster centroid vector
  const sybilNodes = normalizedNodes.filter(n => n.role === 'sybil');
  const sybilVector: [number, number, number] = sybilNodes.length > 0 ? [
    sybilNodes.reduce((acc, n) => acc + n.position[0], 0) / sybilNodes.length,
    sybilNodes.reduce((acc, n) => acc + n.position[1], 0) / sybilNodes.length,
    sybilNodes.reduce((acc, n) => acc + n.position[2], 0) / sybilNodes.length
  ] : [1.4, 0.4, 0.2];

  // Camera presets
  const setCameraView = (x: number, y: number, z: number) => {
    if (controlsRef.current) {
      controlsRef.current.object.position.set(x, y, z);
      controlsRef.current.target.set(0, 0, 0);
      controlsRef.current.update();
    }
  };

  return (
    <div className="relative w-full h-full bg-[#040711] rounded-2xl overflow-hidden border border-cyan-950/80 shadow-[0_0_40px_rgba(0,0,0,0.8)]">
      {/* 3D WebGL Canvas */}
      <Canvas
        camera={{ position: [0, 2.5, 9], fov: 48 }}
        className="w-full h-full cursor-grab active:cursor-grabbing"
      >
        <ambientLight intensity={0.7} />
        <pointLight position={[12, 12, 12]} intensity={1.5} color="#38bdf8" />
        <pointLight position={[-12, -12, -12]} intensity={0.8} color="#a855f7" />

        {/* Deep space starfield background */}
        <Stars radius={45} depth={50} count={2500} factor={4} saturation={0} fade speed={1.2} />

        {/* 3D Coordinate Grid Reference */}
        <gridHelper args={[18, 18, '#1e293b', '#0f172a']} position={[0, -2.2, 0]} />

        {/* Dominant Eigenvector Ray (u_1) pointing from origin through Sybil cluster */}
        {sybilNodes.length > 0 && (
          <group>
            <Line
              points={[[0, 0, 0], [sybilVector[0] * 1.5, sybilVector[1] * 1.5, sybilVector[2] * 1.5]]}
              color="#f43f5e"
              lineWidth={3.5}
            />
            {/* Origin marker */}
            <mesh position={[0, 0, 0]}>
              <sphereGeometry args={[0.08, 16, 16]} />
              <meshBasicMaterial color="#38bdf8" />
            </mesh>
          </group>
        )}

        {/* Nodes */}
        {normalizedNodes.map((node) => (
          <NodeSphere
            key={node.address}
            node={node}
            isSelected={selectedNode?.address === node.address}
            onSelect={() => onSelectNode(node)}
          />
        ))}

        <OrbitControls
          ref={controlsRef}
          enableDamping
          dampingFactor={0.06}
          maxDistance={22}
          minDistance={3}
        />
      </Canvas>

      {/* Floating Camera Control HUD (Top Right) */}
      <div className="absolute top-4 right-4 flex items-center gap-2 bg-[#0b101f]/90 backdrop-blur-md border border-slate-800 p-1.5 rounded-xl shadow-xl font-mono text-xs">
        <button
          onClick={() => setCameraView(0, 2.5, 9)}
          className="px-2.5 py-1 hover:bg-cyan-500/15 text-slate-300 hover:text-cyan-300 rounded-lg transition-colors flex items-center gap-1.5"
          title="Reset to Isometric Perspective"
        >
          <RotateCcw className="w-3 h-3 text-cyan-400" />
          <span>Isometric</span>
        </button>
        <button
          onClick={() => setCameraView(0, 11, 0.01)}
          className="px-2.5 py-1 hover:bg-cyan-500/15 text-slate-300 hover:text-cyan-300 rounded-lg transition-colors flex items-center gap-1.5"
          title="Top-Down View (PCA Axis 1 vs 2)"
        >
          <Eye className="w-3 h-3 text-cyan-400" />
          <span>Top-Down</span>
        </button>
        <button
          onClick={() => setCameraView(10, 0, 0)}
          className="px-2.5 py-1 hover:bg-cyan-500/15 text-slate-300 hover:text-cyan-300 rounded-lg transition-colors flex items-center gap-1.5"
          title="Side Profile (PCA Axis 2 vs 3)"
        >
          <Compass className="w-3 h-3 text-cyan-400" />
          <span>Side View</span>
        </button>
      </div>

      {/* Floating Canvas Legend (Bottom Left) */}
      <div className="absolute bottom-4 left-4 bg-[#0b101f]/90 backdrop-blur-md border border-cyan-950 p-3.5 rounded-xl text-xs font-mono space-y-2 shadow-2xl">
        <div className="flex items-center gap-2.5">
          <span className="w-3 h-3 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.8)]"></span>
          <span className="text-slate-200">Honest Domain Specialist (Dispersed)</span>
        </div>
        <div className="flex items-center gap-2.5">
          <span className="w-3 h-3 rounded-full bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.9)] animate-pulse"></span>
          <span className="text-slate-200">Sybil Backdoor Swarm (Locked on u₁)</span>
        </div>
        <div className="flex items-center gap-2.5">
          <span className="w-3 h-3 rounded-full bg-slate-500 border border-slate-400"></span>
          <span className="text-slate-400">Slashed / Forfeited Node</span>
        </div>
        <p className="text-[10px] text-slate-500 pt-1.5 border-t border-slate-800">
          Rotate: Drag Left Click | Pan: Right Click | Zoom: Scroll
        </p>
      </div>
    </div>
  );
};

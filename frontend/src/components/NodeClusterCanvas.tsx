import React, { useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line, Html } from '@react-three/drei';
import * as THREE from 'three';
import { NodeTelemetry } from '../types/telemetry';

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
  const [hovered, setHovered] = useState(false);

  const isSybil = node.role === 'sybil';
  const color = isSybil ? '#ef4444' : '#10b981';
  const emissiveColor = isSybil ? '#b91c1c' : '#047857';

  // Subtle floating motion
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.position.y =
        node.position[1] + Math.sin(state.clock.elapsedTime * 1.5 + node.position[0]) * 0.08;
    }
  });

  return (
    <group position={[node.position[0], node.position[1], node.position[2]]}>
      <mesh
        ref={meshRef}
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        scale={isSelected ? 1.4 : hovered ? 1.2 : 1.0}
      >
        <sphereGeometry args={[0.22, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={emissiveColor}
          emissiveIntensity={hovered || isSelected ? 0.9 : 0.4}
          roughness={0.2}
          metalness={0.8}
        />
      </mesh>

      {/* Sybil Attack Direction Vector Indicator */}
      {isSybil && (
        <Line
          points={[
            [0, 0, 0],
            [1.2, 0.4, 0.2]
          ]}
          color="#f87171"
          lineWidth={2}
          dashed={false}
        />
      )}

      {/* Label on Hover */}
      {(hovered || isSelected) && (
        <Html distanceFactor={10} position={[0, 0.45, 0]}>
          <div className="bg-slate-900/90 backdrop-blur border border-slate-700 text-xs px-2.5 py-1.5 rounded-lg shadow-xl pointer-events-none whitespace-nowrap">
            <p className="font-mono text-cyan-400 font-semibold">{node.address.slice(0, 8)}...{node.address.slice(-6)}</p>
            <p className={isSybil ? "text-rose-400 font-bold" : "text-emerald-400 font-bold"}>
              {node.role.toUpperCase()} ({node.status})
            </p>
            <p className="text-[10px] text-slate-300">SVD Score: {node.projection_score.toFixed(3)}</p>
          </div>
        </Html>
      )}
    </group>
  );
};

export const NodeClusterCanvas: React.FC<NodeClusterProps> = ({ nodes, selectedNode, onSelectNode }) => {
  // Spatial auto-normalization so coordinates are always framed in view [-3.5, 3.5]
  const maxCoord = Math.max(
    ...nodes.flatMap(n => [
      Math.abs(n.position?.[0] || 0),
      Math.abs(n.position?.[1] || 0),
      Math.abs(n.position?.[2] || 0)
    ]),
    1e-6
  );
  const scale = maxCoord > 3.5 ? 3.5 / maxCoord : 1.0;

  // Normalized node representations
  const normalizedNodes = nodes.map(n => ({
    ...n,
    position: [
      (n.position?.[0] || 0) * scale,
      (n.position?.[1] || 0) * scale,
      (n.position?.[2] || 0) * scale
    ] as [number, number, number]
  }));

  // Empirical Sybil cluster centroid vector
  const sybilNodes = normalizedNodes.filter(n => n.role === 'sybil');
  const sybilVector: [number, number, number] = sybilNodes.length > 0 ? [
    sybilNodes.reduce((acc, n) => acc + n.position[0], 0) / sybilNodes.length,
    sybilNodes.reduce((acc, n) => acc + n.position[1], 0) / sybilNodes.length,
    sybilNodes.reduce((acc, n) => acc + n.position[2], 0) / sybilNodes.length
  ] : [1.2, 0.4, 0.2];

  return (
    <div className="relative w-full h-full bg-[#050811] rounded-xl overflow-hidden border border-slate-800">
      <Canvas
        camera={{ position: [0, 2, 8], fov: 50 }}
        className="w-full h-full cursor-grab active:cursor-grabbing"
      >
        <ambientLight intensity={0.6} />
        <pointLight position={[10, 10, 10]} intensity={1.2} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />

        {/* 3D Coordinate Grid Reference */}
        <gridHelper args={[16, 16, '#1e293b', '#0f172a']} position={[0, -2, 0]} />

        {/* Coordinated Sybil Cluster Alignment Vector from Origin */}
        {sybilNodes.length > 0 && (
          <Line
            points={[[0, 0, 0], sybilVector]}
            color="#f87171"
            lineWidth={2.5}
          />
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

        <OrbitControls enableDamping dampingFactor={0.05} maxDistance={18} minDistance={3} />
      </Canvas>

      {/* Canvas Overlay Legend */}
      <div className="absolute bottom-4 left-4 bg-slate-900/85 backdrop-blur border border-slate-800 p-3 rounded-lg text-xs font-mono space-y-1.5 shadow-lg">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]"></span>
          <span className="text-slate-300">Honest Node (Dispersed Spectrum)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-rose-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]"></span>
          <span className="text-slate-300">Sybil Botnet (λ1 Directional Lock)</span>
        </div>
        <p className="text-[10px] text-slate-500 pt-1 border-t border-slate-800">
          Rotate: Drag | Zoom: Scroll | Select: Click Node
        </p>
      </div>
    </div>
  );
};

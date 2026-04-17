"use client";

import { useRef, useMemo, useState, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Text, Line } from "@react-three/drei";
import * as THREE from "three";
import type {
  LayeredGraphData,
  TraversalTrace,
  GraphNode,
  GraphEdge,
  InterLayerEdge,
  LayerKind,
} from "@/lib/types";

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

const LAYER_Y: Record<LayerKind, number> = {
  document: 4,
  semantic: 0,
  detail: -4,
};

const LAYER_COLORS: Record<LayerKind, string> = {
  document: "#8bb1ff",
  semantic: "#8bffaf",
  detail: "#ffa28b",
};

const LAYER_LABELS: Record<LayerKind, string> = {
  document: "Document Layer",
  semantic: "Semantic Layer",
  detail: "Detail Layer",
};

// ---------------------------------------------------------------------------
// Layout engine — deterministic force-free circular + grid layout
// ---------------------------------------------------------------------------

interface PositionedNode {
  id: string;
  layer: LayerKind;
  label: string;
  kind: string;
  position: [number, number, number];
  visited: boolean;
}

function layoutNodes(
  graph: LayeredGraphData,
  visited: Record<string, string[]> | undefined
): PositionedNode[] {
  const allVisited = new Set<string>();
  if (visited) {
    Object.values(visited).forEach((ids) => ids.forEach((id) => allVisited.add(id)));
  }

  const positions: PositionedNode[] = [];
  const layers: LayerKind[] = ["document", "semantic", "detail"];

  for (const layer of layers) {
    const data = graph[layer];
    const y = LAYER_Y[layer];
    const nodes = data.nodes;
    const count = nodes.length;

    if (count === 0) continue;

    // Use a spiral layout for better distribution
    const radius = Math.max(2, Math.sqrt(count) * 1.2);
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));

    for (let i = 0; i < count; i++) {
      const node = nodes[i];
      const r = radius * Math.sqrt(i / count);
      const theta = i * goldenAngle;
      const x = r * Math.cos(theta);
      const z = r * Math.sin(theta);

      positions.push({
        id: node.id,
        layer,
        label: node.label || node.id,
        kind: node.kind,
        position: [x, y, z],
        visited: allVisited.has(node.id),
      });
    }
  }

  return positions;
}

// ---------------------------------------------------------------------------
// Individual 3D components
// ---------------------------------------------------------------------------

function NodeSphere({
  node,
  onHover,
  onUnhover,
  hovered,
}: {
  node: PositionedNode;
  onHover: (id: string) => void;
  onUnhover: () => void;
  hovered: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const baseColor = LAYER_COLORS[node.layer];
  const color = node.visited ? baseColor : "#3a3b3b";
  const scale = hovered ? 1.6 : node.visited ? 1.0 : 0.5;
  const emissive = node.visited ? baseColor : "#000000";
  const emissiveIntensity = hovered ? 0.6 : node.visited ? 0.3 : 0;

  return (
    <group position={node.position}>
      <mesh
        ref={meshRef}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover(node.id);
        }}
        onPointerOut={onUnhover}
        scale={scale}
      >
        <sphereGeometry args={[0.12, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity}
          transparent
          opacity={node.visited ? 1.0 : 0.3}
        />
      </mesh>
      {hovered && (
        <Text
          position={[0, 0.3, 0]}
          fontSize={0.18}
          color="#e5e7eb"
          anchorX="center"
          anchorY="bottom"
          maxWidth={3}
        >
          {node.label.slice(0, 40)}
        </Text>
      )}
    </group>
  );
}

function EdgeLine({
  from,
  to,
  color,
  opacity,
  dashed,
}: {
  from: [number, number, number];
  to: [number, number, number];
  color: string;
  opacity: number;
  dashed?: boolean;
}) {
  return (
    <Line
      points={[from, to]}
      color={color}
      lineWidth={dashed ? 0.5 : 0.8}
      opacity={opacity}
      transparent
      dashed={dashed}
      dashSize={0.15}
      gapSize={0.1}
    />
  );
}

function LayerPlane({ layer }: { layer: LayerKind }) {
  const y = LAYER_Y[layer];
  const color = LAYER_COLORS[layer];

  return (
    <group position={[0, y, 0]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]}>
        <planeGeometry args={[16, 16]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.03}
          side={THREE.DoubleSide}
        />
      </mesh>
      <Text
        position={[-7.5, 0, -7.5]}
        fontSize={0.25}
        color={color}
        anchorX="left"
        anchorY="middle"
      >
        {LAYER_LABELS[layer]}
      </Text>
    </group>
  );
}

function AnimatedCamera() {
  const { camera } = useThree();
  const initialized = useRef(false);

  useFrame(() => {
    if (!initialized.current) {
      camera.position.set(8, 6, 10);
      camera.lookAt(0, 0, 0);
      initialized.current = true;
    }
  });

  return null;
}

// ---------------------------------------------------------------------------
// Main scene
// ---------------------------------------------------------------------------

function GraphScene({
  graph,
  traversal,
}: {
  graph: LayeredGraphData;
  traversal?: TraversalTrace | null;
}) {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  const nodes = useMemo(
    () => layoutNodes(graph, traversal?.visited_nodes),
    [graph, traversal]
  );

  const posMap = useMemo(() => {
    const m = new Map<string, [number, number, number]>();
    for (const n of nodes) m.set(n.id, n.position);
    return m;
  }, [nodes]);

  const visitedSet = useMemo(() => {
    const s = new Set<string>();
    if (traversal?.visited_nodes) {
      Object.values(traversal.visited_nodes).forEach((ids) =>
        ids.forEach((id) => s.add(id))
      );
    }
    return s;
  }, [traversal]);

  // Collect intra-layer edges
  const intraEdges = useMemo(() => {
    const edges: { from: [number, number, number]; to: [number, number, number]; color: string; visited: boolean }[] = [];
    const layers: LayerKind[] = ["document", "semantic", "detail"];
    for (const layer of layers) {
      const color = LAYER_COLORS[layer];
      for (const edge of graph[layer].edges) {
        const from = posMap.get(edge.source);
        const to = posMap.get(edge.target);
        if (from && to) {
          const vis = visitedSet.has(edge.source) && visitedSet.has(edge.target);
          edges.push({ from, to, color, visited: vis });
        }
      }
    }
    return edges;
  }, [graph, posMap, visitedSet]);

  // Inter-layer edges
  const interEdges = useMemo(() => {
    const edges: { from: [number, number, number]; to: [number, number, number]; visited: boolean }[] = [];
    for (const edge of graph.inter_layer_edges) {
      const from = posMap.get(edge.source);
      const to = posMap.get(edge.target);
      if (from && to) {
        const vis = visitedSet.has(edge.source) && visitedSet.has(edge.target);
        edges.push({ from, to, visited: vis });
      }
    }
    return edges;
  }, [graph, posMap, visitedSet]);

  const handleHover = useCallback((id: string) => setHoveredNode(id), []);
  const handleUnhover = useCallback(() => setHoveredNode(null), []);

  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -5, -10]} intensity={0.3} />

      <AnimatedCamera />
      <OrbitControls
        enableDamping
        dampingFactor={0.1}
        minDistance={3}
        maxDistance={25}
      />

      {/* Layer planes */}
      <LayerPlane layer="document" />
      <LayerPlane layer="semantic" />
      <LayerPlane layer="detail" />

      {/* Intra-layer edges */}
      {intraEdges.map((edge, i) => (
        <EdgeLine
          key={`intra-${i}`}
          from={edge.from}
          to={edge.to}
          color={edge.visited ? edge.color : "#2a2b2b"}
          opacity={edge.visited ? 0.5 : 0.08}
        />
      ))}

      {/* Inter-layer edges */}
      {interEdges.map((edge, i) => (
        <EdgeLine
          key={`inter-${i}`}
          from={edge.from}
          to={edge.to}
          color={edge.visited ? "#ffffff" : "#2a2b2b"}
          opacity={edge.visited ? 0.4 : 0.05}
          dashed
        />
      ))}

      {/* Nodes */}
      {nodes.map((node) => (
        <NodeSphere
          key={node.id}
          node={node}
          onHover={handleHover}
          onUnhover={handleUnhover}
          hovered={hoveredNode === node.id}
        />
      ))}
    </>
  );
}

// ---------------------------------------------------------------------------
// Exported component
// ---------------------------------------------------------------------------

export interface Graph3DProps {
  graph: LayeredGraphData;
  traversal?: TraversalTrace | null;
}

export function Graph3D({ graph, traversal }: Graph3DProps) {
  return (
    <div className="w-full h-full bg-[#0a0a0a] rounded-lg overflow-hidden">
      <Canvas
        camera={{ position: [8, 6, 10], fov: 50 }}
        gl={{ antialias: true, alpha: false }}
        style={{ background: "#0a0a0a" }}
      >
        <GraphScene graph={graph} traversal={traversal} />
      </Canvas>
    </div>
  );
}

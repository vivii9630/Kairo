"use client";

import { useRef, useMemo, useState, useCallback } from "react";
import { Canvas, useFrame, useThree, ThreeEvent } from "@react-three/fiber";
import { Html, OrbitControls, Text, Line } from "@react-three/drei";
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

const CITED_COLOR = "#ffd56b";

// ---------------------------------------------------------------------------
// Layout engine — deterministic force-free circular + grid layout
// ---------------------------------------------------------------------------

interface PositionedNode {
  id: string;
  layer: LayerKind;
  label: string;
  kind: string;
  attrs: Record<string, unknown>;
  position: [number, number, number];
  visited: boolean;
  cited: boolean;
}

function layoutNodes(
  graph: LayeredGraphData,
  visited: Record<string, string[]> | undefined,
  citedSet: Set<string>
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
        attrs: node.attrs ?? {},
        position: [x, y, z],
        visited: allVisited.has(node.id),
        cited: citedSet.has(node.id),
      });
    }
  }

  return positions;
}

// ---------------------------------------------------------------------------
// Hover detail panel — floating HTML above the focal point
// ---------------------------------------------------------------------------

interface HoverPayload {
  title: string;
  subtitle: string;
  body: string;
  url: string | null;
}

function HoverPanel({
  position,
  payload,
}: {
  position: [number, number, number];
  payload: HoverPayload;
}) {
  return (
    <Html
      position={position}
      center
      distanceFactor={8}
      style={{ pointerEvents: "none" }}
      zIndexRange={[100, 0]}
    >
      <div
        style={{
          minWidth: 180,
          maxWidth: 320,
          padding: "8px 10px",
          borderRadius: 8,
          background: "rgba(18,18,20,0.96)",
          border: "1px solid #2a2b2b",
          color: "#e5e7eb",
          fontSize: 11,
          lineHeight: 1.35,
          boxShadow: "0 4px 14px rgba(0,0,0,0.45)",
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 2 }}>{payload.title}</div>
        <div style={{ color: "#8b8c8e", fontSize: 10, marginBottom: 6 }}>
          {payload.subtitle}
        </div>
        {payload.body && (
          <div style={{ color: "#c7c7c9", whiteSpace: "pre-wrap" }}>
            {payload.body}
          </div>
        )}
        {payload.url && (
          <div
            style={{
              marginTop: 6,
              color: "#8be1ff",
              fontSize: 10,
              wordBreak: "break-all",
            }}
          >
            {payload.url}
          </div>
        )}
      </div>
    </Html>
  );
}

// ---------------------------------------------------------------------------
// Node sphere — halo added for cited nodes
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
  const haloRef = useRef<THREE.Mesh>(null);

  const baseColor = node.kind === "web" ? "#c78bff" : LAYER_COLORS[node.layer];
  const color = node.visited ? baseColor : "#3a3b3b";
  const scale = hovered ? 1.7 : node.cited ? 1.35 : node.visited ? 1.0 : 0.5;
  const emissive = node.cited ? CITED_COLOR : node.visited ? baseColor : "#000000";
  const emissiveIntensity = hovered
    ? 0.8
    : node.cited
    ? 0.7
    : node.visited
    ? 0.3
    : 0;

  // Gentle pulse for cited nodes so they read as "the grounded answer".
  useFrame((state) => {
    if (!node.cited || !haloRef.current) return;
    const t = state.clock.getElapsedTime();
    const pulse = 1 + Math.sin(t * 2 + node.position[0]) * 0.08;
    haloRef.current.scale.setScalar(pulse);
  });

  return (
    <group position={node.position}>
      {/* Halo ring for cited nodes */}
      {node.cited && (
        <mesh ref={haloRef} scale={1}>
          <sphereGeometry args={[0.22, 24, 24]} />
          <meshBasicMaterial
            color={CITED_COLOR}
            transparent
            opacity={0.18}
            depthWrite={false}
          />
        </mesh>
      )}
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
    </group>
  );
}

// ---------------------------------------------------------------------------
// Edge rendering — visible Line + invisible cylinder pick target for hover
// ---------------------------------------------------------------------------

interface EdgeVis {
  id: string;
  from: [number, number, number];
  to: [number, number, number];
  color: string;
  opacity: number;
  lineWidth: number;
  dashed: boolean;
  interLayer: boolean;
  kind: string;
  sourceLabel: string;
  targetLabel: string;
  visited: boolean;
  cited: boolean;
}

function HoverableEdge({
  edge,
  hovered,
  onHover,
  onUnhover,
}: {
  edge: EdgeVis;
  hovered: boolean;
  onHover: (id: string) => void;
  onUnhover: () => void;
}) {
  // Midpoint + orientation for the pick-target cylinder.
  const { midpoint, rotation, length } = useMemo(() => {
    const a = new THREE.Vector3(...edge.from);
    const b = new THREE.Vector3(...edge.to);
    const mid = a.clone().add(b).multiplyScalar(0.5);
    const dir = b.clone().sub(a);
    const len = dir.length();
    // Cylinder is oriented along +Y by default.  Build a quaternion that
    // rotates +Y to match the edge direction.
    const q = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      dir.clone().normalize()
    );
    const euler = new THREE.Euler().setFromQuaternion(q);
    return {
      midpoint: [mid.x, mid.y, mid.z] as [number, number, number],
      rotation: [euler.x, euler.y, euler.z] as [number, number, number],
      length: len,
    };
  }, [edge.from, edge.to]);

  const visibleWidth = hovered ? edge.lineWidth + 1.2 : edge.lineWidth;
  const visibleOpacity = hovered ? Math.min(1, edge.opacity + 0.4) : edge.opacity;

  return (
    <group>
      <Line
        points={[edge.from, edge.to]}
        color={edge.color}
        lineWidth={visibleWidth}
        opacity={visibleOpacity}
        transparent
        dashed={edge.dashed}
        dashSize={0.15}
        gapSize={0.1}
      />
      {/* Invisible hit target — thick cylinder along the edge */}
      <mesh
        position={midpoint}
        rotation={rotation}
        onPointerOver={(e: ThreeEvent<PointerEvent>) => {
          e.stopPropagation();
          onHover(edge.id);
        }}
        onPointerOut={onUnhover}
      >
        <cylinderGeometry args={[0.08, 0.08, Math.max(length, 0.001), 6, 1]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>
    </group>
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
// Hover payload builders
// ---------------------------------------------------------------------------

function buildNodePayload(node: PositionedNode): HoverPayload {
  const attrs = node.attrs ?? {};
  const fullText = typeof attrs.full_text === "string" ? attrs.full_text : "";
  const url = typeof attrs.url === "string" ? attrs.url : null;
  const body = fullText.slice(0, 260) + (fullText.length > 260 ? "…" : "");

  const subtitleBits: string[] = [`${node.layer} · ${node.kind}`];
  if (node.cited) subtitleBits.push("cited");
  else if (node.visited) subtitleBits.push("visited");
  return {
    title: node.label,
    subtitle: subtitleBits.join(" · "),
    body,
    url,
  };
}

function buildEdgePayload(edge: EdgeVis): HoverPayload {
  const tag = edge.interLayer ? "inter-layer" : "intra-layer";
  const status = edge.cited
    ? "cited"
    : edge.visited
    ? "visited"
    : "unvisited";
  return {
    title: `${edge.kind} · ${tag}`,
    subtitle: status,
    body: `${edge.sourceLabel}  →  ${edge.targetLabel}`,
    url: null,
  };
}

// ---------------------------------------------------------------------------
// Main scene
// ---------------------------------------------------------------------------

function GraphScene({
  graph,
  traversal,
  citedNodeIds,
}: {
  graph: LayeredGraphData;
  traversal?: TraversalTrace | null;
  citedNodeIds?: string[];
}) {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);

  const citedSet = useMemo(
    () => new Set(citedNodeIds ?? []),
    [citedNodeIds]
  );

  const nodes = useMemo(
    () => layoutNodes(graph, traversal?.visited_nodes, citedSet),
    [graph, traversal, citedSet]
  );

  const nodeMap = useMemo(() => {
    const m = new Map<string, PositionedNode>();
    for (const n of nodes) m.set(n.id, n);
    return m;
  }, [nodes]);

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

  // Intra-layer edges
  const intraEdges = useMemo<EdgeVis[]>(() => {
    const edges: EdgeVis[] = [];
    const layers: LayerKind[] = ["document", "semantic", "detail"];
    for (const layer of layers) {
      const baseColor = LAYER_COLORS[layer];
      for (const edge of graph[layer].edges) {
        const from = posMap.get(edge.source);
        const to = posMap.get(edge.target);
        if (!from || !to) continue;
        const sVis = visitedSet.has(edge.source);
        const tVis = visitedSet.has(edge.target);
        const visited = sVis && tVis;
        const cited =
          citedSet.has(edge.source) && citedSet.has(edge.target);
        edges.push({
          id: `intra:${layer}:${edge.source}->${edge.target}:${edge.kind}`,
          from,
          to,
          color: cited ? CITED_COLOR : visited ? baseColor : "#2a2b2b",
          opacity: cited ? 0.85 : visited ? 0.5 : 0.08,
          lineWidth: cited ? 1.6 : 0.8,
          dashed: false,
          interLayer: false,
          kind: edge.kind,
          sourceLabel: nodeMap.get(edge.source)?.label ?? edge.source,
          targetLabel: nodeMap.get(edge.target)?.label ?? edge.target,
          visited,
          cited,
        });
      }
    }
    return edges;
  }, [graph, posMap, visitedSet, citedSet, nodeMap]);

  // Inter-layer edges
  const interEdges = useMemo<EdgeVis[]>(() => {
    const edges: EdgeVis[] = [];
    for (const edge of graph.inter_layer_edges) {
      const from = posMap.get(edge.source);
      const to = posMap.get(edge.target);
      if (!from || !to) continue;
      const sVis = visitedSet.has(edge.source);
      const tVis = visitedSet.has(edge.target);
      const visited = sVis && tVis;
      const cited = citedSet.has(edge.source) && citedSet.has(edge.target);
      edges.push({
        id: `inter:${edge.source}->${edge.target}:${edge.kind}`,
        from,
        to,
        color: cited ? CITED_COLOR : visited ? "#ffffff" : "#2a2b2b",
        opacity: cited ? 0.9 : visited ? 0.4 : 0.05,
        lineWidth: cited ? 1.6 : 0.8,
        dashed: true,
        interLayer: true,
        kind: edge.kind,
        sourceLabel: nodeMap.get(edge.source)?.label ?? edge.source,
        targetLabel: nodeMap.get(edge.target)?.label ?? edge.target,
        visited,
        cited,
      });
    }
    return edges;
  }, [graph, posMap, visitedSet, citedSet, nodeMap]);

  const handleNodeHover = useCallback((id: string) => {
    setHoveredNode(id);
    setHoveredEdge(null);
  }, []);
  const handleNodeUnhover = useCallback(() => setHoveredNode(null), []);
  const handleEdgeHover = useCallback((id: string) => {
    setHoveredEdge(id);
    setHoveredNode(null);
  }, []);
  const handleEdgeUnhover = useCallback(() => setHoveredEdge(null), []);

  const hoveredNodePayload =
    hoveredNode && nodeMap.has(hoveredNode)
      ? {
          position: nodeMap.get(hoveredNode)!.position,
          payload: buildNodePayload(nodeMap.get(hoveredNode)!),
        }
      : null;

  const hoveredEdgeObj = useMemo(() => {
    if (!hoveredEdge) return null;
    const all = [...intraEdges, ...interEdges];
    return all.find((e) => e.id === hoveredEdge) ?? null;
  }, [hoveredEdge, intraEdges, interEdges]);

  const hoveredEdgePayload = hoveredEdgeObj
    ? {
        position: [
          (hoveredEdgeObj.from[0] + hoveredEdgeObj.to[0]) / 2,
          (hoveredEdgeObj.from[1] + hoveredEdgeObj.to[1]) / 2 + 0.3,
          (hoveredEdgeObj.from[2] + hoveredEdgeObj.to[2]) / 2,
        ] as [number, number, number],
        payload: buildEdgePayload(hoveredEdgeObj),
      }
    : null;

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

      {/* Edges (visible line + invisible pick target) */}
      {intraEdges.map((edge) => (
        <HoverableEdge
          key={edge.id}
          edge={edge}
          hovered={hoveredEdge === edge.id}
          onHover={handleEdgeHover}
          onUnhover={handleEdgeUnhover}
        />
      ))}
      {interEdges.map((edge) => (
        <HoverableEdge
          key={edge.id}
          edge={edge}
          hovered={hoveredEdge === edge.id}
          onHover={handleEdgeHover}
          onUnhover={handleEdgeUnhover}
        />
      ))}

      {/* Nodes */}
      {nodes.map((node) => (
        <NodeSphere
          key={node.id}
          node={node}
          onHover={handleNodeHover}
          onUnhover={handleNodeUnhover}
          hovered={hoveredNode === node.id}
        />
      ))}

      {/* Hover panels — only one at a time */}
      {hoveredNodePayload && (
        <HoverPanel
          position={[
            hoveredNodePayload.position[0],
            hoveredNodePayload.position[1] + 0.5,
            hoveredNodePayload.position[2],
          ]}
          payload={hoveredNodePayload.payload}
        />
      )}
      {!hoveredNodePayload && hoveredEdgePayload && (
        <HoverPanel
          position={hoveredEdgePayload.position}
          payload={hoveredEdgePayload.payload}
        />
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Exported component
// ---------------------------------------------------------------------------

export interface Graph3DProps {
  graph: LayeredGraphData;
  traversal?: TraversalTrace | null;
  citedNodeIds?: string[];
}

export function Graph3D({ graph, traversal, citedNodeIds }: Graph3DProps) {
  return (
    <div className="w-full h-full bg-[#0a0a0a] rounded-lg overflow-hidden">
      <Canvas
        camera={{ position: [8, 6, 10], fov: 50 }}
        gl={{ antialias: true, alpha: false }}
        style={{ background: "#0a0a0a" }}
      >
        <GraphScene
          graph={graph}
          traversal={traversal}
          citedNodeIds={citedNodeIds}
        />
      </Canvas>
    </div>
  );
}

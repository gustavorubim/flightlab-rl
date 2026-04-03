import { useEffect, useMemo, useRef } from 'react';

import { Html, Line, OrbitControls } from '@react-three/drei';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

import type { SessionSnapshot } from '@/types';

type CameraMode = 'orbit' | 'chase';

interface MissionSceneProps {
  snapshot: SessionSnapshot;
  cameraMode: CameraMode;
  onCameraModeChange: (mode: CameraMode) => void;
}

const SCENE_SCALE = 0.02;
const RUNWAY_LENGTH = 210;
const RUNWAY_WIDTH = 12;

function toScenePosition(x_m: number, y_m: number, altitude_m: number): [number, number, number] {
  return [x_m * SCENE_SCALE, altitude_m * SCENE_SCALE, y_m * SCENE_SCALE];
}

function SceneEnvironment({ snapshot, cameraMode }: Omit<MissionSceneProps, 'onCameraModeChange'>) {
  const aircraftPosition = useMemo(
    () =>
      new THREE.Vector3(
        snapshot.aircraft.position_x_m * SCENE_SCALE,
        snapshot.aircraft.altitude_m * SCENE_SCALE,
        snapshot.aircraft.position_y_m * SCENE_SCALE,
      ),
    [snapshot.aircraft.altitude_m, snapshot.aircraft.position_x_m, snapshot.aircraft.position_y_m],
  );

  const orbitControls = useRef<any>(null);
  const targetWaypoint =
    snapshot.mission.waypoints[snapshot.mission.active_waypoint_index ?? 0] ?? snapshot.mission.waypoints[0];
  const activeTarget = useMemo(
    () =>
      targetWaypoint
        ? new THREE.Vector3(
            targetWaypoint.x_m * SCENE_SCALE,
            targetWaypoint.altitude_m * SCENE_SCALE,
            targetWaypoint.y_m * SCENE_SCALE,
          )
        : aircraftPosition.clone(),
    [aircraftPosition, targetWaypoint],
  );

  const trailPoints = useMemo(
    () => snapshot.trail.map((point) => toScenePosition(point.x_m, point.y_m, point.altitude_m)),
    [snapshot.trail],
  );

  const routePoints = useMemo(
    () =>
      snapshot.mission.waypoints.map((waypoint) =>
        toScenePosition(waypoint.x_m, waypoint.y_m, waypoint.altitude_m),
      ),
    [snapshot.mission.waypoints],
  );

  useFrame(({ camera }) => {
    if (cameraMode === 'chase') {
      const heading = snapshot.aircraft.heading_rad;
      const offset = new THREE.Vector3(-12, 6, 18).applyAxisAngle(
        new THREE.Vector3(0, 1, 0),
        heading,
      );
      const desired = aircraftPosition.clone().add(offset);
      camera.position.lerp(desired, 0.08);
      camera.lookAt(aircraftPosition);
    }
  });

  useEffect(() => {
    if (orbitControls.current && cameraMode === 'orbit') {
      orbitControls.current.target.copy(aircraftPosition);
      orbitControls.current.update();
    }
  }, [aircraftPosition, cameraMode]);

  return (
    <>
      <ambientLight intensity={0.85} />
      <directionalLight position={[10, 20, 10]} intensity={1.5} />
      <directionalLight position={[-10, 8, -14]} intensity={0.35} color="#d8f6b0" />
      <fog attach="fog" args={['#08120d', 30, 260]} />

      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[300, 300]} />
        <meshStandardMaterial color="#0c1711" roughness={1} metalness={0} />
      </mesh>

      <gridHelper args={[300, 60, '#35523a', '#15301d']} position={[0, 0.01, 0]} />

      <mesh position={[0, 0.08, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <boxGeometry args={[RUNWAY_LENGTH, RUNWAY_WIDTH, 0.3]} />
        <meshStandardMaterial color="#141b15" roughness={0.95} />
      </mesh>

      <mesh position={[0, 0.13, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[RUNWAY_LENGTH - 8, 1.1]} />
        <meshStandardMaterial color="#d9d1a6" emissive="#5f5a23" emissiveIntensity={0.18} />
      </mesh>

      <group position={aircraftPosition.toArray() as [number, number, number]}>
        <group rotation={[snapshot.aircraft.pitch_rad, snapshot.aircraft.heading_rad, -snapshot.aircraft.roll_rad]}>
          <mesh position={[0, 0.4, 0]} rotation={[0, Math.PI / 2, 0]}>
            <coneGeometry args={[0.9, 4.8, 8]} />
            <meshStandardMaterial color="#a9b48b" emissive="#55644b" emissiveIntensity={0.18} />
          </mesh>
          <mesh position={[-2, 0.12, 0]}>
            <boxGeometry args={[4.4, 0.18, 1.2]} />
            <meshStandardMaterial color="#6b7756" />
          </mesh>
          <mesh position={[-0.8, 0.25, 0]}>
            <boxGeometry args={[1.1, 0.1, 4.6]} />
            <meshStandardMaterial color="#72825a" />
          </mesh>
          <mesh position={[-2.25, 0.45, 0]}>
            <boxGeometry args={[1.0, 0.07, 1.8]} />
            <meshStandardMaterial color="#4c553f" />
          </mesh>
        </group>
      </group>

      {routePoints.length > 1 ? (
        <Line points={routePoints} color="#87b96d" lineWidth={2} dashed={false} />
      ) : null}

      {trailPoints.length > 1 ? (
        <Line points={trailPoints} color="#7ed9c0" lineWidth={2} dashed={false} />
      ) : null}

      <mesh position={activeTarget.toArray() as [number, number, number]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[2.4, 0.08, 12, 36]} />
        <meshStandardMaterial color="#d9a441" emissive="#d9a441" emissiveIntensity={0.35} />
      </mesh>

      <mesh position={[activeTarget.x, 0.08, activeTarget.z]}>
        <ringGeometry args={[1.4, 2.6, 32]} />
        <meshBasicMaterial color="#d9a441" transparent opacity={0.22} />
      </mesh>

      <Html position={activeTarget.toArray() as [number, number, number]} center distanceFactor={12}>
        <div className="scene-label scene-label-target">ACTIVE TARGET</div>
      </Html>

      <Html position={[aircraftPosition.x, aircraftPosition.y + 4, aircraftPosition.z]} center distanceFactor={16}>
        <div className="scene-label scene-label-aircraft">AIRCRAFT</div>
      </Html>

      {cameraMode === 'orbit' ? (
        <OrbitControls
          ref={orbitControls}
          makeDefault
          enableDamping
          dampingFactor={0.08}
          maxPolarAngle={Math.PI * 0.48}
          minDistance={20}
          maxDistance={220}
          target={aircraftPosition.toArray() as [number, number, number]}
        />
      ) : null}
    </>
  );
}

export function MissionScene({
  snapshot,
  cameraMode,
  onCameraModeChange,
}: MissionSceneProps) {
  return (
    <section className="panel scene-panel">
      <div className="panel-header scene-header">
        <div>
          <p className="eyebrow">3D asset</p>
          <h2>Cinematic flight deck</h2>
        </div>
        <div className="camera-toggle" role="tablist" aria-label="Camera mode">
          <button
            type="button"
            className={`toggle ${cameraMode === 'orbit' ? 'is-active' : ''}`}
            onClick={() => onCameraModeChange('orbit')}
          >
            Orbit
          </button>
          <button
            type="button"
            className={`toggle ${cameraMode === 'chase' ? 'is-active' : ''}`}
            onClick={() => onCameraModeChange('chase')}
          >
            Chase
          </button>
        </div>
      </div>

      <div className="scene-wrap">
        <Canvas
          shadows
          camera={{ position: [20, 18, 24], fov: 48, near: 0.1, far: 1000 }}
          dpr={[1, 1.5]}
        >
          <SceneEnvironment snapshot={snapshot} cameraMode={cameraMode} />
        </Canvas>
        <div className="scene-hud">
          <div className="hud-line">
            <span>Heading</span>
            <strong>{snapshot.aircraft.heading_rad.toFixed(2)} rad</strong>
          </div>
          <div className="hud-line">
            <span>Altitude</span>
            <strong>{Math.round(snapshot.aircraft.altitude_m)} m</strong>
          </div>
          <div className="hud-line">
            <span>Speed</span>
            <strong>{Math.round(snapshot.aircraft.airspeed_mps)} m/s</strong>
          </div>
          <div className="hud-line">
            <span>Phase</span>
            <strong>{snapshot.session.phase}</strong>
          </div>
        </div>
      </div>
    </section>
  );
}

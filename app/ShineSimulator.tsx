"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  HandLandmarker,
  NormalizedLandmark,
} from "@mediapipe/tasks-vision";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// 0~100: 일반 광택, 100~MAX: 태양 모드(화면이 점점 하얘짐), MAX 도달 시 완전 백화
const SUN_START = 100;
const MAX_SHINE = 130;
const GAIN = 2.2; // 비비는 세기 1당 초당 광택 증가량
const DECAY = 3; // 안 비빌 때 초당 광택 감소량
const RUB_DEADZONE = 1;
const RUB_CAP = 12;

const PALM_IDX = [0, 5, 9, 13, 17];

const LEVELS: [number, string][] = [
  [0, "무광 매트"],
  [15, "은은한 윤기"],
  [35, "반질반질"],
  [60, "거울 수준"],
  [85, "눈뽕 주의"],
  [SUN_START, "☀️ 태양 모드"],
];

const SPARKLES = [
  { x: 128, y: 82, r: 14, d: 0 },
  { x: 262, y: 70, r: 10, d: 0.4 },
  { x: 300, y: 128, r: 16, d: 0.8 },
  { x: 98, y: 150, r: 9, d: 0.2 },
  { x: 205, y: 38, r: 18, d: 0.6 },
  { x: 330, y: 70, r: 12, d: 1.0 },
  { x: 70, y: 90, r: 13, d: 0.3 },
  { x: 180, y: 120, r: 8, d: 0.9 },
  { x: 345, y: 170, r: 10, d: 0.5 },
  { x: 55, y: 175, r: 11, d: 0.7 },
];

type CamStatus = "off" | "loading" | "on" | "error";
type Vec = { x: number; y: number };
type View = {
  shine: number;
  rub: number;
  hands: number;
  onHead: number;
  burnedOut: boolean;
};
type ScreenPalm = { x: number; y: number; size: number; onHead: boolean };

function palmOf(lm: NormalizedLandmark[]) {
  const c = PALM_IDX.reduce(
    (acc, i) => ({ x: acc.x + lm[i].x / 5, y: acc.y + lm[i].y / 5 }),
    { x: 0, y: 0 },
  );
  const size = Math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y) || 0.1;
  return { center: c, size };
}

// 비디오 정규화 좌표 → 화면 좌표 (object-cover 크롭 + 좌우 반전 반영)
function toScreen(video: HTMLVideoElement, p: Vec) {
  const r = video.getBoundingClientRect();
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;
  const scale = Math.max(r.width / vw, r.height / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  const ux = (r.width - dw) / 2 + p.x * dw;
  return { x: r.left + r.width - ux, y: r.top + (r.height - dh) / 2 + p.y * dh };
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

function lerpColor(a: string, b: string, t: number) {
  const pa = a.match(/\w\w/g)!.map((h) => parseInt(h, 16));
  const pb = b.match(/\w\w/g)!.map((h) => parseInt(h, 16));
  return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(",")})`;
}

const STAR =
  "M0,-1 L0.22,-0.22 L1,0 L0.22,0.22 L0,1 L-0.22,0.22 L-1,0 L-0.22,-0.22Z";

export default function ShineSimulator() {
  const [view, setView] = useState<View>({
    shine: 0,
    rub: 0,
    hands: 0,
    onHead: 0,
    burnedOut: false,
  });
  const [cam, setCam] = useState<CamStatus>("off");
  const [camError, setCamError] = useState("");

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const shineRef = useRef(0);
  const rubRef = useRef(0);
  const burnedOutRef = useRef(false);
  const handSpeedRef = useRef(0);
  const handsRef = useRef(0);
  const onHeadRef = useRef(0);
  const prevPalmsRef = useRef<{ palms: ScreenPalm[]; t: number } | null>(
    null,
  );
  const lastVideoTimeRef = useRef(-1);
  const headRef = useRef<SVGEllipseElement>(null);

  // 아저씨 머리 위에서 움직이는 손의 속도(손 크기 단위/초)를 계산
  const measureRub = useCallback(
    (landmarks: NormalizedLandmark[][], t: number) => {
      const video = videoRef.current;
      const head = headRef.current;
      if (!video || !head) return { speed: 0, onHead: [] as boolean[] };

      const hr = head.getBoundingClientRect();
      const cx = hr.left + hr.width / 2;
      const cy = hr.top + hr.height / 2;
      const rx = (hr.width / 2) * 1.15;
      const ry = (hr.height / 2) * 1.15;

      const palms: ScreenPalm[] = landmarks.map((lm) => {
        const c = toScreen(video, palmOf(lm).center);
        const wrist = toScreen(video, lm[0]);
        const mid = toScreen(video, lm[9]);
        return {
          ...c,
          size: Math.hypot(mid.x - wrist.x, mid.y - wrist.y) || 50,
          onHead: ((c.x - cx) / rx) ** 2 + ((c.y - cy) / ry) ** 2 <= 1,
        };
      });
      handsRef.current = palms.length;
      onHeadRef.current = palms.filter((p) => p.onHead).length;

      const prev = prevPalmsRef.current;
      prevPalmsRef.current = { palms, t };
      const onHead = palms.map((p) => p.onHead);
      if (!prev) return { speed: 0, onHead };
      const dt = (t - prev.t) / 1000;
      if (dt <= 0 || dt > 0.3) return { speed: 0, onHead };

      let best = 0;
      for (const p of palms) {
        if (!p.onHead) continue;
        // 직전 프레임에서 가장 가까운 손을 같은 손으로 간주
        let nearest = Infinity;
        for (const q of prev.palms) {
          nearest = Math.min(nearest, Math.hypot(p.x - q.x, p.y - q.y));
        }
        if (nearest > p.size * 2) continue;
        best = Math.max(best, nearest / dt / p.size);
      }
      const speed = best < RUB_DEADZONE ? 0 : Math.min(best, RUB_CAP);
      return { speed, onHead };
    },
    [],
  );

  const drawOverlay = useCallback(
    (landmarks: NormalizedLandmark[][], onHead: boolean[]) => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) return;
      if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
      if (canvas.height !== video.videoHeight)
        canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      landmarks.forEach((lm, i) => {
        ctx.fillStyle = onHead[i]
          ? "rgba(255, 230, 120, 0.95)"
          : "rgba(255, 255, 255, 0.6)";
        for (const p of lm) {
          ctx.beginPath();
          ctx.arc(p.x * canvas.width, p.y * canvas.height, 3, 0, Math.PI * 2);
          ctx.fill();
        }
        const { center } = palmOf(lm);
        ctx.beginPath();
        ctx.arc(
          center.x * canvas.width,
          center.y * canvas.height,
          onHead[i] ? 18 : 12,
          0,
          Math.PI * 2,
        );
        if (onHead[i]) {
          ctx.fillStyle = "rgba(255, 220, 90, 0.45)";
          ctx.fill();
        }
        ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
        ctx.lineWidth = 3;
        ctx.stroke();
      });
    },
    [],
  );

  // 메인 루프: 손 인식 + 광택 물리
  useEffect(() => {
    let raf = 0;
    let last = performance.now();

    const tick = () => {
      const now = performance.now();
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;

      const video = videoRef.current;
      const lmk = landmarkerRef.current;
      if (
        lmk &&
        video &&
        video.readyState >= 2 &&
        video.currentTime !== lastVideoTimeRef.current
      ) {
        lastVideoTimeRef.current = video.currentTime;
        const result = lmk.detectForVideo(video, now);
        const { speed, onHead } = measureRub(result.landmarks, now);
        handSpeedRef.current = Math.max(handSpeedRef.current, speed);
        drawOverlay(result.landmarks, onHead);
      }

      const target = handSpeedRef.current;
      handSpeedRef.current *= Math.exp(-dt * 6);
      rubRef.current += (target - rubRef.current) * Math.min(1, dt * 8);

      if (!burnedOutRef.current) {
        let s = shineRef.current + rubRef.current * GAIN * dt;
        if (rubRef.current < 0.5) {
          s -= DECAY * dt * (s > SUN_START ? 2 : 1);
        }
        s = Math.min(MAX_SHINE, Math.max(0, s));
        if (s >= MAX_SHINE) burnedOutRef.current = true;
        shineRef.current = s;
      }

      setView({
        shine: shineRef.current,
        rub: rubRef.current,
        hands: handsRef.current,
        onHead: onHeadRef.current,
        burnedOut: burnedOutRef.current,
      });
      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [measureRub, drawOverlay]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      landmarkerRef.current?.close();
    };
  }, []);

  const loadLandmarker = async () => {
    const { FilesetResolver, HandLandmarker } = await import(
      "@mediapipe/tasks-vision"
    );
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    const create = (delegate: "GPU" | "CPU") =>
      HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.4,
        minHandPresenceConfidence: 0.4,
        minTrackingConfidence: 0.4,
      });
    try {
      return await create("GPU");
    } catch (e) {
      console.warn("GPU 손 인식 실패, CPU로 재시도", e);
      return await create("CPU");
    }
  };

  const startCamera = async () => {
    setCamError("");
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setCam("error");
      setCamError(
        `카메라는 https 또는 localhost 주소에서만 켤 수 있어요. http://localhost:${window.location.port || 80} 으로 열어 주세요.`,
      );
      return;
    }

    setCam("loading");
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
    } catch (e) {
      console.error(e);
      const name = e instanceof DOMException ? e.name : "";
      setCam("error");
      setCamError(
        name === "NotAllowedError"
          ? "카메라 권한이 거부됐어요. 주소창 왼쪽 아이콘에서 카메라를 허용해 주세요. (macOS라면 시스템 설정 › 개인정보 보호 › 카메라에서 브라우저도 허용)"
          : name === "NotFoundError"
            ? "연결된 카메라를 찾지 못했어요."
            : name === "NotReadableError"
              ? "다른 앱(Zoom, FaceTime 등)이 카메라를 쓰고 있어요."
              : `카메라를 켜지 못했어요. (${name || String(e)})`,
      );
      return;
    }

    streamRef.current = stream;
    const video = videoRef.current!;
    video.srcObject = stream;
    try {
      await video.play();
    } catch (e) {
      console.warn("video.play() 실패", e);
    }

    try {
      landmarkerRef.current ??= await loadLandmarker();
      setCam("on");
    } catch (e) {
      console.error(e);
      // 카메라 화면은 계속 보여주되 손 인식만 실패한 상태
      setCam("on");
      setCamError(
        "손 인식 모델을 불러오지 못했어요. 인터넷 연결을 확인하고 새로고침해 주세요.",
      );
    }
  };

  const reset = () => {
    shineRef.current = 0;
    rubRef.current = 0;
    burnedOutRef.current = false;
  };

  // ---- 렌더링용 파생 값 ----
  const { shine, rub, hands, onHead, burnedOut } = view;
  const s = clamp01(shine / SUN_START);
  const sun = clamp01((shine - 85) / (MAX_SHINE - 85));
  const whiteout = burnedOut
    ? 1
    : Math.pow(clamp01((shine - SUN_START) / (MAX_SHINE - SUN_START)), 1.4);
  const level = [...LEVELS].reverse().find(([min]) => shine >= min)![1];
  const skin = lerpColor("e6b48c", "fff2c2", sun);
  const skinShade = lerpColor("cf9670", "ffe08a", sun);
  const sparkleCount = Math.floor(s * SPARKLES.length + 0.001);
  const rubbing = rub > 1.5;

  return (
    <div className="relative h-screen w-full overflow-hidden bg-[#10121f] text-white">
      {/* 전체 화면 카메라 */}
      <video
        ref={videoRef}
        playsInline
        muted
        className="absolute inset-0 h-full w-full -scale-x-100 object-cover"
      />
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 z-[15] h-full w-full -scale-x-100 object-cover"
      />
      {cam !== "on" && (
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_40%,#2a2f4a,#10121f_70%)]" />
      )}

      {/* 아저씨 머리에서 번지는 빛 */}
      <div
        className="pointer-events-none absolute inset-0 mix-blend-screen"
        style={{
          background:
            "radial-gradient(circle at 78% 52%, rgba(255,230,140,0.9), rgba(255,200,80,0.35) 35%, transparent 70%)",
          opacity: Math.pow(s, 1.5) * 0.6 + sun * 0.4,
        }}
      />

      {/* 옆에 서 있는 아저씨 */}
      <section className="pointer-events-none absolute bottom-0 right-0 z-10 w-[min(72vw,520px)] select-none md:right-[4vw]">
        <div className="relative w-full">
          <svg viewBox="0 0 400 460" className="w-full overflow-visible">
            <defs>
              <radialGradient id="glow">
                <stop offset="0%" stopColor="#fffbe6" stopOpacity="1" />
                <stop offset="45%" stopColor="#ffe27a" stopOpacity="0.6" />
                <stop offset="100%" stopColor="#ffb703" stopOpacity="0" />
              </radialGradient>
              <radialGradient id="spec" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#ffffff" stopOpacity="1" />
                <stop offset="60%" stopColor="#ffffff" stopOpacity="0.7" />
                <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
              </radialGradient>
              <radialGradient id="dome" cx="38%" cy="22%" r="70%">
                <stop offset="0%" stopColor="#ffffff" stopOpacity="0.9" />
                <stop offset="35%" stopColor="#ffffff" stopOpacity="0.25" />
                <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
              </radialGradient>
              <clipPath id="headClip">
                <ellipse cx="200" cy="185" rx="110" ry="125" />
              </clipPath>
            </defs>

            {/* 후광 */}
            <circle
              cx="200"
              cy="150"
              r={140 + 160 * s + 120 * sun}
              fill="url(#glow)"
              opacity={Math.pow(s, 1.5)}
            />

            {/* 태양 광선 */}
            <g opacity={sun} transform="translate(200 150)">
              <g className="spin">
                {Array.from({ length: 16 }, (_, i) => (
                  <path
                    key={i}
                    d="M-14,-150 L0,-290 L14,-150Z"
                    fill="#fff3b0"
                    transform={`rotate(${i * 22.5})`}
                  />
                ))}
              </g>
            </g>

            {/* 몸 */}
            <path
              d="M70,460 C80,370 130,340 200,340 C270,340 320,370 330,460Z"
              fill="#3d5a80"
            />
            <path d="M175,340 L200,385 L225,340Z" fill="#f1f1f1" />
            <rect x="172" y="295" width="56" height="55" rx="12" fill={skinShade} />

            {/* 귀 */}
            <ellipse cx="92" cy="205" rx="20" ry="30" fill={skinShade} />
            <ellipse cx="308" cy="205" rx="20" ry="30" fill={skinShade} />

            {/* 머리 */}
            <ellipse
              ref={headRef}
              cx="200"
              cy="185"
              rx="110"
              ry="125"
              fill={skin}
            />
            <g clipPath="url(#headClip)">
              <ellipse
                cx="200"
                cy="185"
                rx="110"
                ry="125"
                fill="url(#dome)"
                opacity={0.15 + 0.85 * s}
              />
            </g>

            {/* 옆머리 */}
            <path
              d="M92,175 C88,215 100,250 115,262 C105,235 104,205 110,180Z"
              fill="#8d8d8d"
            />
            <path
              d="M308,175 C312,215 300,250 285,262 C295,235 296,205 290,180Z"
              fill="#8d8d8d"
            />

            {/* 하이라이트 */}
            <g transform="translate(155 100) rotate(-28)">
              <ellipse
                rx={30 + 22 * s}
                ry={16 + 10 * s}
                fill="url(#spec)"
                opacity={0.12 + 0.88 * s}
              />
            </g>
            <ellipse
              cx="238"
              cy="82"
              rx={8 + 6 * s}
              ry={5 + 3 * s}
              fill="#fff"
              opacity={clamp01(s * 1.5 - 0.2)}
            />

            {/* 눈썹 */}
            <path
              d="M140,180 Q160,168 180,178"
              stroke="#6b6b6b"
              strokeWidth="7"
              fill="none"
              strokeLinecap="round"
            />
            <path
              d="M220,178 Q240,168 260,180"
              stroke="#6b6b6b"
              strokeWidth="7"
              fill="none"
              strokeLinecap="round"
            />

            {/* 눈 */}
            {sun > 0.25 ? (
              <g>
                <rect x="132" y="190" width="56" height="30" rx="10" fill="#111" />
                <rect x="212" y="190" width="56" height="30" rx="10" fill="#111" />
                <path d="M188,200 L212,200" stroke="#111" strokeWidth="5" />
                <path
                  d="M140,197 L156,197"
                  stroke="#fff"
                  strokeWidth="3"
                  strokeLinecap="round"
                  opacity="0.7"
                />
                <path
                  d="M220,197 L236,197"
                  stroke="#fff"
                  strokeWidth="3"
                  strokeLinecap="round"
                  opacity="0.7"
                />
              </g>
            ) : s > 0.35 ? (
              <g stroke="#2b2b2b" strokeWidth="6" fill="none" strokeLinecap="round">
                <path d="M146,208 Q160,192 174,208" />
                <path d="M226,208 Q240,192 254,208" />
              </g>
            ) : (
              <g fill="#2b2b2b">
                <circle cx="160" cy="205" r="8" />
                <circle cx="240" cy="205" r="8" />
              </g>
            )}

            {/* 볼터치 */}
            <ellipse cx="135" cy="245" rx="16" ry="9" fill="#ff7b7b" opacity={s * 0.5} />
            <ellipse cx="265" cy="245" rx="16" ry="9" fill="#ff7b7b" opacity={s * 0.5} />

            {/* 코, 콧수염, 입 */}
            <path
              d="M200,212 Q188,240 198,248"
              stroke={skinShade}
              strokeWidth="6"
              fill="none"
              strokeLinecap="round"
            />
            <path
              d="M160,262 Q180,248 200,260 Q220,248 240,262 Q220,272 200,266 Q180,272 160,262Z"
              fill="#7a7a7a"
            />
            <path
              d={`M178,280 Q200,${284 + 14 * s} 222,280`}
              stroke="#8a3b3b"
              strokeWidth="5"
              fill="none"
              strokeLinecap="round"
            />

            {/* 반짝이 */}
            {SPARKLES.slice(0, sparkleCount).map((p, i) => (
              <g
                key={i}
                transform={`translate(${p.x} ${p.y}) scale(${p.r * (1 + sun)})`}
              >
                <path
                  d={STAR}
                  className="twinkle"
                  fill="#fff"
                  style={{ animationDelay: `${p.d}s` }}
                />
              </g>
            ))}
          </svg>

          {cam === "on" && onHead === 0 && !burnedOut && (
            <div className="absolute left-1/2 top-[2%] -translate-x-1/2 animate-bounce whitespace-nowrap rounded-full bg-black/50 px-4 py-1.5 text-lg font-bold text-yellow-200">
              👋 여기를 문질러요
            </div>
          )}
          {rubbing && !burnedOut && (
            <div
              className="pointer-events-none absolute left-1/2 top-[38%] -translate-x-1/2 animate-pulse whitespace-nowrap text-3xl font-black text-yellow-200 drop-shadow-[0_2px_6px_rgba(0,0,0,0.6)]"
              style={{ opacity: clamp01(rub / 6) }}
            >
              슥삭슥삭!
            </div>
          )}
        </div>
      </section>

      {/* HUD */}
      <aside className="absolute left-4 top-4 z-10 flex w-[min(calc(100vw-2rem),18rem)] flex-col gap-3 rounded-2xl bg-black/45 p-4 backdrop-blur-md">
        <h1 className="text-lg font-bold tracking-tight">
          대머리 광택 시뮬레이터 ✨
        </h1>
        <div>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="text-xs text-white/70">광택 레벨</span>
            <span className="font-mono text-xs">
              {Math.round((shine / SUN_START) * 100)}%
            </span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-white/15">
            <div
              className="h-full rounded-full"
              style={{
                width: `${(shine / MAX_SHINE) * 100}%`,
                background:
                  "linear-gradient(90deg, #c08a5c, #ffd166 70%, #fff8dc)",
              }}
            />
          </div>
          <div className="mt-1.5 text-lg font-bold">{level}</div>
        </div>

        <div>
          <div className="mb-1 text-xs text-white/70">비비는 세기</div>
          <div className="h-2 overflow-hidden rounded-full bg-white/15">
            <div
              className="h-full rounded-full bg-orange-400"
              style={{ width: `${(rub / RUB_CAP) * 100}%` }}
            />
          </div>
        </div>

        {cam === "on" && camError && (
          <div className="text-xs text-red-300">{camError}</div>
        )}
        {cam === "on" && !camError && (
          <div className="text-xs text-white/70">
            인식된 손: {hands}개{" "}
            {hands === 0
              ? "· 손을 화면에 보여주세요"
              : onHead === 0
                ? "· 손을 아저씨 머리 위로!"
                : "· 머리를 문질러요!"}
          </div>
        )}

        <button
          onClick={reset}
          className="rounded-lg border border-white/30 px-3 py-1.5 text-xs hover:bg-white/10"
        >
          광택 초기화
        </button>
      </aside>

      {/* 카메라 시작 안내 */}
      {cam !== "on" && (
        <div className="absolute inset-0 z-[5] flex flex-col items-center justify-center gap-3 p-6 text-center">
          {cam === "off" && (
            <>
              <button
                onClick={startCamera}
                className="rounded-full bg-yellow-300 px-7 py-3 text-lg font-bold text-black shadow-lg hover:bg-yellow-200"
              >
                📷 카메라 켜기
              </button>
              <span className="text-sm text-white/70">
                화면 속 아저씨 머리 위에 손을 올리고 슥슥 문지르면 반짝여요
              </span>
            </>
          )}
          {cam === "loading" && (
            <span className="text-white/80">손 인식 모델 불러오는 중…</span>
          )}
          {cam === "error" && (
            <>
              <span className="max-w-md text-sm text-white/80">
                {camError}
              </span>
              <button
                onClick={startCamera}
                className="text-sm text-white/70 underline"
              >
                다시 시도
              </button>
            </>
          )}
        </div>
      )}

      {/* 백화 */}
      <div
        className="pointer-events-none fixed inset-0 z-20 bg-white"
        style={{ opacity: whiteout }}
      />
      {burnedOut && (
        <div className="fixed inset-0 z-30 flex flex-col items-center justify-center gap-4 p-6 text-center text-neutral-800">
          <div className="text-6xl">☀️</div>
          <div className="text-3xl font-black">아저씨가 태양이 되었습니다</div>
          <div className="text-neutral-500">너무 많이 비볐어요… 눈이 부셔서 아무것도 안 보여요</div>
          <button
            onClick={reset}
            className="mt-2 rounded-full bg-neutral-900 px-6 py-3 font-bold text-white hover:bg-neutral-700"
          >
            선글라스 끼고 다시 하기 🕶️
          </button>
        </div>
      )}
    </div>
  );
}

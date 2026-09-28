/* GLSL for the process orb.
 *
 * Three programs share one displacement field, defined once in DISPLACE and
 * injected into all of them. That sharing is the whole reason the layers read
 * as one object: the core, the glass shell and the particle skin are three
 * different materials evaluating the *same* function at the same time, so
 * they deform together instead of sliding through one another.
 *
 * Everything is driven from uniforms rather than from geometry, so a mode
 * change is an eased number and never a rebuild.
 */
export const NOISE = /* glsl */`
// Ashima / Stefan Gustavson simplex noise, 3D. Public domain (MIT).
// https://github.com/ashima/webgl-noise
vec3 mod289(vec3 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 mod289(vec4 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 permute(vec4 x){ return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod289(i);
  vec4 p = permute(permute(permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
`;

/* The shared displacement field.
 *
 * One function, four morph weights. The weights are eased on the CPU and
 * arrive here already blended, which is why a mode change never snaps: at any
 * instant this is evaluating a mixture of all four states, not switching
 * between them.
 *
 * uRipples carries up to MAX_RIPPLES live surface events as
 * vec4(x, y, z, age01) -- a unit direction and how far through its life the
 * ripple is. A travelling band of displacement is drawn where the angular
 * distance from that direction matches the ripple's expanding radius, which
 * is what makes an event read as something crossing the surface rather than
 * as a spot that blinks.
 */
export const DISPLACE = /* glsl */`
uniform float uTime;
uniform float uIdle;       // morph weights, already eased and summing to ~1
uniform float uActive;
uniform float uIntense;
uniform float uAlert;
uniform float uBreath;     // idle breathing scale, ±2%
uniform float uFlow;       // how fast the noise field evolves
uniform float uAmp;        // global amplitude, cut under reduced motion
uniform vec4  uRipples[MAX_RIPPLES];
uniform float uRippleStrength;

// How far, in radians, a ripple has travelled by the time it dies. Just over
// a hemisphere: far enough to read as crossing the orb, short enough that it
// has faded before it would converge on the far pole and interfere.
const float RIPPLE_REACH = 2.2;

float rippleField(vec3 dir) {
  float total = 0.0;
  for (int i = 0; i < MAX_RIPPLES; i++) {
    vec4 r = uRipples[i];
    float age = r.w;
    if (age <= 0.0 || age >= 1.0) continue;
    float arc = acos(clamp(dot(dir, normalize(r.xyz)), -1.0, 1.0));
    float front = age * RIPPLE_REACH;
    // A narrow band at the wavefront, tightening as it goes so the ring
    // thins rather than smears.
    float band = exp(-pow((arc - front) * (5.0 + 4.0 * age), 2.0));
    // Fade in fast, out slow: an event should announce itself and then let go.
    float life = sin(age * 3.14159265) * (1.0 - age * 0.35);
    total += band * life;
  }
  return total;
}

float displacement(vec3 dir, out float lobe) {
  float t = uTime * uFlow;

  // idle: almost a sphere, breathing. The tiny noise term is deliberate --
  // a perfectly still sphere reads as a frozen render, not a resting thing.
  float nIdle = snoise(dir * 1.1 + vec3(0.0, 0.0, t * 0.18)) * 0.035;

  // active: low-frequency organic wobble.
  float nActive = snoise(dir * 1.9 + vec3(t * 0.30, 0.0, 0.0)) * 0.13
                + snoise(dir * 3.4 - vec3(0.0, t * 0.22, 0.0)) * 0.05;

  // intense: higher-frequency lobes pushing outward, faster flow. The abs()
  // on the second octave is what makes it lumpy rather than merely noisy --
  // it creates creases where the field crosses zero.
  float nIntense = snoise(dir * 2.6 + vec3(0.0, t * 0.55, t * 0.35)) * 0.20
                 + abs(snoise(dir * 5.1 + vec3(t * 0.7, 0.0, 0.0))) * 0.13
                 + snoise(dir * 8.0 - vec3(t * 0.9, 0.0, 0.0)) * 0.035;

  // alert: the body narrows as it sags. This is only the pinch -- the
  // downward stretch cannot live here, because everything in this function
  // scales the surface along its own radius, and a negative radius on the
  // lower hemisphere pulls it *inward* rather than drawing it down. The orb
  // got smaller when it was supposed to be melting. See meltOffset.
  float below = smoothstep(0.15, -1.0, dir.y);
  float nAlert = -below * 0.10;

  lobe = nIntense;

  float d = nIdle * uIdle + nActive * uActive + nIntense * uIntense
          + nAlert * uAlert;
  d += rippleField(dir) * uRippleStrength;
  return d * uAmp + uBreath;
}

/* The melt, as a vector rather than a radius.
 *
 * Gravity does not act along a surface normal, so the alert state cannot be
 * expressed as a scalar displacement the way the other three are. Vertices on
 * the lower hemisphere are drawn straight down in world Y, with a falloff that
 * is strongest at the pole and nothing at the equator, and the amount is
 * noise-modulated so the underside breaks into strands instead of sagging as
 * one smooth teardrop.
 */
vec3 meltOffset(vec3 dir) {
  if (uAlert <= 0.001) return vec3(0.0);
  float t = uTime * uFlow;
  float below = smoothstep(0.30, -1.0, dir.y);
  float drip = snoise(vec3(dir.x * 3.6, dir.y * 1.2 - t * 0.9, dir.z * 3.6));
  float amount = below * below * (0.34 + 0.26 * drip);
  return vec3(0.0, -amount * uAlert * uAmp, 0.0);
}
`;

/* INNER CORE -- the glossy liquid mass.
 *
 * Colour is not a texture. Each active process owns a moving attractor on the
 * unit sphere, and every fragment is a softmax-style blend of all of them
 * weighted by angular proximity and by the process's activity. That is what
 * gives zones with no hard edges whose size tracks activity: a weak process
 * still colours the fragments nearest its attractor, which is the "thin vein,
 * never fully gone" the brief asks for.
 *
 * The attractors drift on their own noise, so the zones swirl.
 */
export const CORE_VERT = /* glsl */`
varying vec3 vNormalW;
varying vec3 vViewDir;
varying vec3 vDir;
varying float vLobe;

void main() {
  vec3 dir = normalize(position);
  float lobe;
  float d = displacement(dir, lobe);
  vLobe = lobe;
  vDir = dir;

  vec3 displaced = position * (1.0 + d) + meltOffset(dir);

  // Re-derive a normal from two neighbours on the displaced surface. Cheaper
  // and steadier than recomputing the geometry's own normals, and good enough
  // for a body this smooth.
  vec3 tangent = normalize(cross(dir, abs(dir.y) < 0.99 ? vec3(0.0,1.0,0.0)
                                                        : vec3(1.0,0.0,0.0)));
  vec3 bitan = normalize(cross(dir, tangent));
  float eps = 0.06;
  float dA; float dB; float junk;
  dA = displacement(normalize(dir + tangent * eps), junk);
  dB = displacement(normalize(dir + bitan * eps), junk);
  vec3 pA = normalize(dir + tangent * eps) * (1.0 + dA)
            + meltOffset(normalize(dir + tangent * eps));
  vec3 pB = normalize(dir + bitan * eps) * (1.0 + dB)
            + meltOffset(normalize(dir + bitan * eps));
  vec3 n = normalize(cross(pA - displaced, pB - displaced));
  if (dot(n, dir) < 0.0) n = -n;

  vec4 world = modelMatrix * vec4(displaced, 1.0);
  vNormalW = normalize(mat3(modelMatrix) * n);
  vViewDir = normalize(cameraPosition - world.xyz);
  gl_Position = projectionMatrix * viewMatrix * world;
}
`;

export const CORE_FRAG = /* glsl */`
uniform float uTime;
uniform int   uCount;
uniform vec3  uColors[MAX_PROCESSES];
uniform float uWeights[MAX_PROCESSES];
uniform vec3  uSeeds[MAX_PROCESSES];
uniform vec3  uDominant;
uniform float uHeartbeat;
uniform float uGlow;

varying vec3 vNormalW;
varying vec3 vViewDir;
varying vec3 vDir;
varying float vLobe;

void main() {
  // Where each process sits on the orb right now. The attractor drifts on its
  // own slow noise, which is what makes the zones flow into each other rather
  // than sit in fixed patches.
  vec3 acc = vec3(0.0);
  float total = 0.0;
  for (int i = 0; i < MAX_PROCESSES; i++) {
    if (i >= uCount) break;
    float w = uWeights[i];
    if (w <= 0.0004) continue;

    vec3 seed = uSeeds[i];
    vec3 drift = vec3(
      snoise(seed * 1.7 + vec3(uTime * 0.11, 0.0, 0.0)),
      snoise(seed * 1.7 + vec3(0.0, uTime * 0.13, 0.0)),
      snoise(seed * 1.7 + vec3(0.0, 0.0, uTime * 0.09))
    );
    vec3 centre = normalize(seed + drift * 0.55);

    float arc = acos(clamp(dot(vDir, centre), -1.0, 1.0));
    // Zone radius grows with activity. The floor is what keeps a weak
    // process visible as a wisp instead of vanishing while it is still
    // running -- a process that is alive and invisible is a lie.
    float radius = 0.55 + 1.35 * pow(w, 0.55);
    float field = exp(-pow(arc / radius, 2.0)) * (0.12 + w);

    acc += uColors[i] * field;
    total += field;
  }

  vec3 base = total > 0.0001 ? acc / total : vec3(0.28, 0.30, 0.34);

  // Liquid sheen. A tight specular lobe plus a broad wrap term reads as a
  // glossy mass; without the wrap it looks like a hard plastic ball.
  vec3 N = normalize(vNormalW);
  float fres = pow(1.0 - clamp(dot(N, normalize(vViewDir)), 0.0, 1.0), 2.6);
  vec3 L = normalize(vec3(0.45, 0.85, 0.55));
  float spec = pow(max(dot(reflect(-L, N), normalize(vViewDir)), 0.0), 34.0);
  float wrap = clamp(dot(N, L) * 0.5 + 0.5, 0.0, 1.0);

  // Weighted toward the zone colour rather than toward the lighting. The
  // core is the one place the process palette is legible, so lighting shapes
  // it and must not drain it.
  vec3 col = base * (0.26 + 0.52 * wrap);
  // Creases in the displacement field catch a little more light, which is
  // what makes the intense state read as lumpy rather than as a blurred ball.
  col += base * clamp(vLobe, 0.0, 1.0) * 0.22;
  // The rim glow is scaled by how much is actually happening. At rest the orb
  // is nearly unlit -- the whole point of the palette is that colour on this
  // screen means an event, so a body glowing while nothing runs is noise.
  col += uDominant * fres * (0.10 + 0.40 * uHeartbeat) * uGlow;
  col += vec3(spec) * 0.10;

  gl_FragColor = vec4(col, 1.0);
}
`;

/* OUTER SHELL -- frosted glass, as a purpose-built shader.
 *
 * This started as a MeshPhysicalMaterial with transmission ~0.9, which is the
 * obvious way to get real refraction. It was replaced after measurement, and
 * the reasons are worth keeping because the obvious way is the tempting one:
 *
 *   1. THE ARTIFACT. Three's transmission samples a mip of the scene chosen
 *      by the surface's roughness. At any roughness that reads as "frosted"
 *      it reads a heavily downsampled copy, which on this scene rendered as a
 *      blocky white rectangle hanging beside the orb -- 459 blown pixels at
 *      roughness 0.34, still hundreds at 0.10, and only clean at 0.02, where
 *      the glass is a mirror and not frosted at all.
 *   2. THE HOLES. A transmissive mesh has to pick a side. DoubleSide drew the
 *      back faces as dark creases across the body wherever the displacement
 *      folded the surface through itself; FrontSide culled those same folds
 *      into holes you could see the background through.
 *   3. THE COST. Transmission re-renders the whole scene into a buffer every
 *      frame. On a panel that already carries a 43k-vertex body and a point
 *      cloud, it was the single most expensive thing here.
 *
 * So the refraction is faked, which for a body whose interior is a colour
 * field rather than a scene is not much of a lie: the shell samples the core's
 * own zone colours along a refracted direction and tints them. The brief
 * allows this explicitly -- "or an equivalent custom shader with fresnel rim
 * lighting" -- and the result has no artifact, no holes, and no second render
 * pass.
 *
 * Two membranes at slightly different radii, both double-sided with no depth
 * write, which is what gives the overlapping folded layers rather than one
 * balloon.
 */
export const SHELL_VERT = /* glsl */`
// How closely the shell follows the core's displacement. Under one, so the
// membrane lags the mass it wraps and reads as a separate skin rather than as
// a second copy of the same surface.
uniform float uShellFollow;
uniform float uShellRadius;

varying vec3 vShellNormal;
varying vec3 vShellView;
varying vec3 vShellDir;

void main() {
  vec3 dir = normalize(position);
  float lobe;
  float d = displacement(dir, lobe);
  vShellDir = dir;

  vec3 displaced = position * (1.0 + d * uShellFollow) * uShellRadius
                   + meltOffset(dir);

  vec3 tangent = normalize(cross(dir, abs(dir.y) < 0.99 ? vec3(0.0,1.0,0.0)
                                                        : vec3(1.0,0.0,0.0)));
  vec3 bitan = normalize(cross(dir, tangent));
  float eps = 0.07;
  float junk;
  vec3 nA = normalize(dir + tangent * eps);
  vec3 nB = normalize(dir + bitan * eps);
  vec3 pA = nA * (1.0 + displacement(nA, junk) * uShellFollow) * uShellRadius
            + meltOffset(nA);
  vec3 pB = nB * (1.0 + displacement(nB, junk) * uShellFollow) * uShellRadius
            + meltOffset(nB);
  vec3 n = normalize(cross(pA - displaced, pB - displaced));
  if (dot(n, dir) < 0.0) n = -n;

  vec4 world = modelMatrix * vec4(displaced, 1.0);
  vShellNormal = normalize(mat3(modelMatrix) * n);
  vShellView = normalize(cameraPosition - world.xyz);
  gl_Position = projectionMatrix * viewMatrix * world;
}
`;

export const SHELL_FRAG = /* glsl */`
uniform int   uCount;
uniform vec3  uColors[MAX_PROCESSES];
uniform float uWeights[MAX_PROCESSES];
uniform vec3  uSeeds[MAX_PROCESSES];
uniform vec3  uDominant;
uniform float uShellAlpha;
uniform float uIor;

varying vec3 vShellNormal;
varying vec3 vShellView;
varying vec3 vShellDir;

void main() {
  vec3 N = normalize(vShellNormal);
  vec3 V = normalize(vShellView);

  // Schlick, with the ior the brief asks for rather than a hand-picked curve.
  float f0 = pow((1.0 - uIor) / (1.0 + uIor), 2.0);
  float ndv = clamp(dot(N, V), 0.0, 1.0);
  float fres = f0 + (1.0 - f0) * pow(1.0 - ndv, 4.2);

  // What is behind this patch of membrane: the core's zone field, sampled
  // along a refracted direction rather than straight through. Bending the
  // lookup is what makes the colour inside appear displaced at the rim, which
  // is the part of refraction the eye actually reads.
  vec3 refracted = normalize(refract(-V, N, 1.0 / uIor));
  vec3 probe = normalize(vShellDir + refracted * 0.35);

  vec3 acc = vec3(0.0);
  float total = 0.0;
  for (int i = 0; i < MAX_PROCESSES; i++) {
    if (i >= uCount) break;
    float w = uWeights[i];
    if (w <= 0.0004) continue;
    float arc = acos(clamp(dot(probe, normalize(uSeeds[i])), -1.0, 1.0));
    float field = exp(-pow(arc / (0.7 + 1.3 * pow(w, 0.55)), 2.0)) * (0.1 + w);
    acc += uColors[i] * field;
    total += field;
  }
  vec3 inside = total > 0.0001 ? acc / total : vec3(0.30, 0.33, 0.38);

  // Iridescence: a slow hue rotation with view angle, which is what a thin
  // film does and what sells "glass" more than any amount of specular.
  vec3 irid = 0.5 + 0.5 * cos(6.2831 * (vec3(0.0, 0.33, 0.67) + fres * 1.6));
  // Weighted toward the dominant process rather than toward the film. A rim
  // that is mostly iridescence averages to white, and two membranes stacking
  // their rims then blew the whole silhouette out -- which is the opposite of
  // the point, since the rim is meant to say which process is dominant.
  vec3 rim = mix(uDominant, irid, 0.20);

  vec3 col = inside * 0.26 + rim * fres * 0.55;
  float alpha = uShellAlpha * (0.30 + 0.70 * fres);

  gl_FragColor = vec4(col, alpha);
}
`;

/* PARTICLE SKIN -- an evenly spread point cloud just outside the shell.
 *
 * Points are placed by a fibonacci sphere on the CPU (even coverage, no
 * clumping at the poles the way a naive random sphere does) and displaced
 * here by the same field. Each point carries its own phase so the skin
 * shimmers without any two dots agreeing.
 */
export const SKIN_VERT = /* glsl */`
uniform float uSize;
uniform float uPixelRatio;
uniform float uSkinOffset;
// How much of the body's deformation the skin takes. Under one, because a
// skin that followed every lobe exactly flung its points past the silhouette
// in the intense state and read as debris around the orb rather than as a
// surface on it.
uniform float uSkinFollow;
attribute float aPhase;
varying float vTwinkle;
varying vec3 vSkinDir;

void main() {
  vec3 dir = normalize(position);
  float lobe;
  float d = displacement(dir, lobe);
  vSkinDir = dir;

  vec3 p = dir * (1.0 + d * uSkinFollow) * uSkinOffset + meltOffset(dir);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);

  // Per-point phase, so the skin glitters instead of pulsing as one sheet.
  vTwinkle = 0.55 + 0.45 * sin(uTime * 1.7 + aPhase * 6.2831);

  // The 6.0 is tuned to this camera, not copied from a snippet. The usual
  // form of this line uses 300.0, which assumes a camera a hundred-odd units
  // out; at the 6.4 this one sits at, it made every point 117 pixels across,
  // and fourteen thousand additively blended 117px sprites is a white disc
  // with an orb somewhere inside it.
  gl_PointSize = uSize * uPixelRatio * (6.0 / max(-mv.z, 0.001));
  gl_Position = projectionMatrix * mv;
}
`;

export const SKIN_FRAG = /* glsl */`
uniform vec3 uTint;
uniform float uOpacity;
varying float vTwinkle;
varying vec3 vSkinDir;

void main() {
  // Round points. Without this they are squares, which at this count reads as
  // static rather than as a dotted skin.
  vec2 uv = gl_PointCoord - 0.5;
  float r = dot(uv, uv);
  if (r > 0.25) discard;
  float soft = smoothstep(0.25, 0.02, r);

  gl_FragColor = vec4(uTint, uOpacity * soft * vTwinkle);
}
`;

/* The soft contact shadow under the orb. A radial falloff on a ground plane,
 * squashed and faded by how high the orb is floating -- a shadow that stayed
 * the same size while the thing above it bobbed would read as a sticker. */
export const SHADOW_FRAG = /* glsl */`
uniform float uOpacity;
uniform float uSpread;
varying vec2 vUv;

void main() {
  float d = length(vUv - 0.5) * 2.0;
  float a = smoothstep(1.0, 0.0, d / max(uSpread, 0.001));
  gl_FragColor = vec4(0.0, 0.0, 0.0, a * a * uOpacity);
}
`;

export const SHADOW_VERT = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

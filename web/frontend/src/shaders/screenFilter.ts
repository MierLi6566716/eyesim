/**
 * Live GPU screen-filter: a fullscreen WebGL2 fragment shader that
 * convolves an uploaded photo against the small PSF kernel returned by
 * POST /api/cases/simulate. Deliberately scoped to the uploaded image for
 * v1 (not an arbitrary live page/webcam snapshot); see
 * proposal (web build plan) for why. The kernel is capped at ~24x24 by the
 * backend (services/physics.py::_SHADER_KERNEL_MAX) specifically so this
 * per-fragment double loop stays real-time.
 */

export interface ScreenFilter {
  render: () => void;
  destroy: () => void;
}

const VERT_SRC = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

function fragSrc(kernelSize: number): string {
  const half = Math.floor(kernelSize / 2);
  return `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uImage;
uniform vec2 uTexel;
uniform float uKernel[${kernelSize * kernelSize}];

vec3 toLin(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }
vec3 toSrgb(vec3 c) { c = clamp(c, 0.0, 1.0); return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }

void main() {
  vec3 sum = vec3(0.0);
  for (int j = 0; j < ${kernelSize}; j++) {
    for (int i = 0; i < ${kernelSize}; i++) {
      vec2 offset = vec2(float(i - ${half}), float(j - ${half})) * uTexel;
      sum += toLin(texture(uImage, vUv + offset).rgb) * uKernel[j * ${kernelSize} + i];
    }
  }
  outColor = vec4(toSrgb(sum), 1.0);
}`;
}

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Could not create shader");
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error: ${log}`);
  }
  return shader;
}

export function createScreenFilter(
  canvas: HTMLCanvasElement,
  image: HTMLImageElement,
  kernel: number[][],
): ScreenFilter {
  const gl = canvas.getContext("webgl2");
  if (!gl) throw new Error("WebGL2 not supported in this browser");

  const kernelSize = kernel.length;
  const program = gl.createProgram();
  if (!program) throw new Error("Could not create GL program");
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERT_SRC));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragSrc(kernelSize)));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`Program link error: ${gl.getProgramInfoLog(program)}`);
  }
  gl.useProgram(program);

  const quad = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(program, "aPos");
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  gl.viewport(0, 0, canvas.width, canvas.height);

  const flatKernel = new Float32Array(kernelSize * kernelSize);
  for (let j = 0; j < kernelSize; j++) {
    for (let i = 0; i < kernelSize; i++) {
      flatKernel[j * kernelSize + i] = kernel[j][i];
    }
  }

  gl.uniform1i(gl.getUniformLocation(program, "uImage"), 0);
  gl.uniform2f(gl.getUniformLocation(program, "uTexel"), 1 / canvas.width, 1 / canvas.height);
  gl.uniform1fv(gl.getUniformLocation(program, "uKernel"), flatKernel);

  function render() {
    gl!.activeTexture(gl!.TEXTURE0);
    gl!.bindTexture(gl!.TEXTURE_2D, texture);
    gl!.drawArrays(gl!.TRIANGLE_STRIP, 0, 4);
  }

  function destroy() {
    gl!.deleteProgram(program);
    gl!.deleteTexture(texture);
    gl!.deleteBuffer(vbo);
  }

  render();
  return { render, destroy };
}

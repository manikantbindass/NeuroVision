import * as nifti from "nifti-reader-js";

export type Point = [number, number, number];
export type Volume = {
  shape: Point;
  spacing: Point;
  affine: number[][];
  values: Float32Array | Uint16Array;
};

export const indexAt = (shape: Point, x: number, y: number, z: number) =>
  x + shape[0] * (y + shape[1] * z);

export function sameGrid(a: Volume, b: Volume) {
  return (
    a.shape.every((size, axis) => size === b.shape[axis]) &&
    a.affine.every((row, i) =>
      row.every((value, j) => Math.abs(value - b.affine[i][j]) < 1e-3),
    )
  );
}

export function decodeVolume(buffer: ArrayBuffer): Volume {
  if (buffer.byteLength > 8_001_000 || !nifti.isNIFTI(buffer)) {
    throw new Error(
      "The server did not return a supported bounded NIfTI volume.",
    );
  }
  const header = nifti.readHeader(buffer);
  const shape: Point = [header.dims[1], header.dims[2], header.dims[3]];
  const count = shape[0] * shape[1] * shape[2];
  const type = header.datatypeCode;
  const bytes = type === 16 ? 4 : type === 512 ? 2 : 0;
  if (
    header.dims[0] !== 3 ||
    !shape.every((value) => Number.isSafeInteger(value) && value > 0) ||
    count > 2_000_000 ||
    !bytes ||
    header.numBitsPerVoxel !== bytes * 8 ||
    !Number.isSafeInteger(header.vox_offset) ||
    header.vox_offset < 352 ||
    header.vox_offset + count * bytes > buffer.byteLength
  ) {
    throw new Error("The exported NIfTI grid or datatype is unsupported.");
  }
  const view = new DataView(buffer, header.vox_offset, count * bytes);
  const values = type === 16 ? new Float32Array(count) : new Uint16Array(count);
  for (let i = 0; i < count; i++) {
    const value =
      type === 16
        ? view.getFloat32(i * bytes, header.littleEndian)
        : view.getUint16(i * bytes, header.littleEndian);
    if (!Number.isFinite(value))
      throw new Error("The exported volume contains invalid values.");
    values[i] = value;
  }
  const affine = header.affine;
  const spacing = [0, 1, 2].map((axis) =>
    Math.hypot(affine[0][axis], affine[1][axis], affine[2][axis]),
  ) as Point;
  if (
    !affine.every((row) => row.every(Number.isFinite)) ||
    !spacing.every((value) => value > 0)
  ) {
    throw new Error("The exported volume has invalid geometry.");
  }
  return { shape, spacing, affine, values };
}

export const PLANES = [
  {
    name: "Axial",
    axis: 2,
    horizontal: 0,
    vertical: 1,
    left: "L",
    right: "R",
    top: "A",
    bottom: "P",
  },
  {
    name: "Sagittal",
    axis: 0,
    horizontal: 1,
    vertical: 2,
    left: "P",
    right: "A",
    top: "S",
    bottom: "I",
  },
  {
    name: "Coronal",
    axis: 1,
    horizontal: 0,
    vertical: 2,
    left: "L",
    right: "R",
    top: "S",
    bottom: "I",
  },
] as const;

export type Plane = (typeof PLANES)[number];
export const midpoint = (volume: Volume): Point =>
  volume.shape.map((n) => Math.floor(n / 2)) as Point;

export function intensityWindow(volume: Volume): [number, number] {
  const stride = Math.max(1, Math.floor(volume.values.length / 20_000));
  const samples: number[] = [];
  for (let i = 0; i < volume.values.length; i += stride)
    samples.push(volume.values[i]);
  samples.sort((a, b) => a - b);
  const low = samples[Math.floor(samples.length * 0.01)];
  const high = samples[Math.floor(samples.length * 0.995)];
  return [low, high === low ? low + 1 : high];
}

export const COLORS = [
  [85, 225, 194],
  [221, 151, 255],
  [251, 185, 100],
  [116, 174, 255],
  [252, 134, 150],
] as const;
export const classColor = (id: number) => COLORS[(id - 1) % COLORS.length];
export const rgb = (color: readonly number[]) => `rgb(${color.join(",")})`;

// matmul.wgsl：行主序矩阵乘法 compute shader
// 每个 workgroup 处理 BLOCK_SIZE × BLOCK_SIZE 的 C 子块
// 行主序直接算，不需要 cuBLAS 的列主序 trick

@group(0) @binding(0) var<storage, read> a: array<f32>;     // [m, k]
@group(0) @binding(1) var<storage, read> b: array<f32>;     // [k, n]
@group(0) @binding(2) var<storage, read_write> c: array<f32>; // [m, n]

// 维度常量从 storage buffer 传入（wgpu 30 推荐用 STORAGE 而非 UNIFORM）
@group(0) @binding(3) var<storage, read> dims: vec4<u32>;  // x=m, y=n, z=k, w=unused

const BLOCK_SIZE: u32 = 16;

@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;  // C 的行
    let col = gid.x;  // C 的列
    let m = dims.x;
    let n = dims.y;
    let k = dims.z;

    // 边界检查
    if row >= m || col >= n { return; }

    // 点积：C[row, col] = Σ_{p=0}^{k-1} A[row, p] * B[p, col]
    var acc: f32 = 0.0;
    for (var p: u32 = 0u; p < k; p++) {
        acc += a[row * k + p] * b[p * n + col];
    }
    c[row * n + col] = acc;
}

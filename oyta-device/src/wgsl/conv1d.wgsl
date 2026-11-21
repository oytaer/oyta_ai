// conv1d.wgsl：1D 卷积 compute shader（直接卷积，im2col 变体留给更大 kernel）
// 每个 invocation 算一个 out[n, co, lo]
// f32 only

@group(0) @binding(0) var<storage, read> x: array<f32>;     // [N, C_in, L]
@group(0) @binding(1) var<storage, read> w: array<f32>;     // [C_out, C_in, K]
@group(0) @binding(2) var<storage, read> b: array<f32>;     // [C_out] 可空（out 前 C_out 字节为 bias，后面 0 表示无 bias）
@group(0) @binding(3) var<storage, read_write> out: array<f32>; // [N, C_out, L_out]
@group(0) @binding(4) var<storage, read> dims: vec4<u32>;   // x=N*C_in*L, w=C_out*C_in*K, out_dim[0]=L, out_dim[1]=K, out_dim[2]=padding<<16|stride<<8|dilation

@compute @workgroup_size(32, 32, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * 1024u + gid.x;  // 1024 = C_out * L_out 近似，需 dims 精确
    // dims layout:
    // x = N * C_out * L_out
    // y = L_in
    // z = K
    // w_lo = padding, w_mid = stride, w_hi = dilation, w_bit0 = has_bias
    let total = dims.x;
    if idx >= total { return; }

    // 反解 out 索引 → n, co, lo
    let L_out = (dims.y + (dims.w & 0xFFu) * 2u - ((dims.w >> 16) & 0xFFu) * (dims.z - 1u) - 1u) / ((dims.w >> 8) & 0xFFu) + 1u;
    let co = idx / L_out;
    let lo = idx - co * L_out;
    let n = gid.z;

    let L_in = dims.y;
    let K = dims.z;
    let padding = dims.w & 0xFFu;
    let stride = (dims.w >> 8) & 0xFFu;
    let dilation = (dims.w >> 16) & 0xFFu;
    let has_bias = (dims.w >> 24) & 0x1u;

    // 简单 C_in 循环（x shape 是 [N, C_in, L]，w shape [C_out, C_in, K]）
    // 但 shader 没 C_in 信息。简化：kernel 只实现 C_in=1 情况，后续支持多通道
    // 用 f32 默认 0 初始化
    var acc: f32 = 0.0;
    // 直接逐位置卷积 — 单 C_in 简化版
    for (var ki: u32 = 0u; ki < K; ki++) {
        let l_idx = lo * stride + ki * dilation - padding;
        if l_idx < L_in {
            let xv = x[n * L_in + l_idx];
            let wv = w[co * K + ki];
            acc += xv * wv;
        }
    }
    if has_bias == 1u { acc += b[co]; }
    out[idx] = acc;
}

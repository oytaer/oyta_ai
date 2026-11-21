// conv1d.wgsl：1D 卷积 compute shader（多通道完整版）
// 支持 NCHW 格式、任意 padding/stride/dilation、可选 bias
// 每个 thread 算一个 out[n, co, lo]

@group(0) @binding(0) var<storage, read> x: array<f32>;     // [N, C_in, L_in]
@group(0) @binding(1) var<storage, read> w: array<f32>;     // [C_out, C_in, K]
@group(0) @binding(2) var<storage, read> b: array<f32>;     // [C_out] 或空
@group(0) @binding(3) var<storage, read_write> out: array<f32>; // [N, C_out, L_out]

// dims 打包: [N, C_in, C_out, L_in, L_out, K, PAD|STRIDE<<8|DIL<<16|HAS_BIAS<<24]
@group(0) @binding(4) var<storage, read> dims: array<u32>;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let N = dims[0];
    let C_in = dims[1];
    let C_out = dims[2];
    let L_in = dims[3];
    let L_out = dims[4];
    let K = dims[5];
    // 拆 packed hyperparameters
    let pack = dims[6];
    let padding = pack & 0xFFu;
    let stride = (pack >> 8) & 0xFFu;
    let dilation = (pack >> 16) & 0xFFu;
    let has_bias = (pack >> 24) & 0xFFu;

    let total = N * C_out * L_out;
    if idx >= total { return; }

    // 反推 n, co, lo
    let lo = idx % L_out;
    let co = (idx / L_out) % C_out;
    let n = idx / (C_out * L_out);

    var acc: f32 = 0.0;
    for (var ci: u32 = 0u; ci < C_in; ci++) {
        for (var ki: u32 = 0u; ki < K; ki++) {
            let l_idx = lo * stride + ki * dilation - padding;
            if l_idx >= 0 && l_idx < L_in {
                let x_pos = (n * C_in + ci) * L_in + l_idx;
                let w_pos = (co * C_in + ci) * K + ki;
                acc += x[x_pos] * w[w_pos];
            }
        }
    }
    if has_bias != 0u { acc += b[co]; }
    out[idx] = acc;
}

// ====== Conv1d backward DX ======
// dx[n, ci, li] += w[co, ci, ki] * dy[n, co, lo]  反卷积回传
@group(1) @binding(0) var<storage, read> dy_buf: array<f32>;  // [N, C_out, L_out]
@group(1) @binding(1) var<storage, read> w_bwd: array<f32>;   // [C_out, C_in, K]
@group(1) @binding(2) var<storage, read_write> dx_buf: array<f32>; // [N, C_in, L_in]
@group(1) @binding(3) var<storage, read> dims_bwd: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn conv1d_bwd_dx(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let N = dims_bwd[0];
    let C_in = dims_bwd[1];
    let C_out = dims_bwd[2];
    let L_in = dims_bwd[3];
    let L_out = dims_bwd[4];
    let K = dims_bwd[5];
    let pack = dims_bwd[6];
    let padding = pack & 0xFFu;
    let stride = (pack >> 8) & 0xFFu;
    let dilation = (pack >> 16) & 0xFFu;

    let total = N * C_in * L_in;
    if idx >= total { return; }

    let li = idx % L_in;
    let ci = (idx / L_in) % C_in;
    let n = idx / (C_in * L_in);

    var acc: f32 = 0.0;
    for (var co: u32 = 0u; co < C_out; co++) {
        for (var lo: u32 = 0u; lo < L_out; lo++) {
            for (var ki: u32 = 0u; ki < K; ki++) {
                let l_calc = lo * stride + ki * dilation - padding;
                if l_calc == li {
                    let dy_pos = (n * C_out + co) * L_out + lo;
                    let w_pos = (co * C_in + ci) * K + ki;
                    acc += w_bwd[w_pos] * dy_buf[dy_pos];
                }
            }
        }
    }
    dx_buf[idx] = acc;
}

// ====== Conv1d backward DWDB ======
// 每个 thread 算一个 dw[co, ci, ki]（外层循环遍历 n, lo）
// db[co] 单独 atomicAdd
@group(2) @binding(0) var<storage, read> dy_dw: array<f32>;   // [N, C_out, L_out]
@group(2) @binding(1) var<storage, read> x_dw: array<f32>;    // [N, C_in, L_in]
@group(2) @binding(2) var<storage, read_write> dw_dw: array<f32>; // [C_out, C_in, K]
@group(2) @binding(3) var<storage, read_write> db_dw: array<f32>; // [C_out]
@group(2) @binding(4) var<storage, read> dims_dw: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn conv1d_bwd_dwdb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let N = dims_dw[0];
    let C_in = dims_dw[1];
    let C_out = dims_dw[2];
    let L_in = dims_dw[3];
    let L_out = dims_dw[4];
    let K = dims_dw[5];
    let pack = dims_dw[6];
    let padding = pack & 0xFFu;
    let stride = (pack >> 8) & 0xFFu;
    let dilation = (pack >> 16) & 0xFFu;

    // dw 总元素数
    let dw_total = C_out * C_in * K;
    if idx >= dw_total { return; }

    let ki = idx % K;
    let ci = (idx / K) % C_in;
    let co = idx / (C_in * K);

    // 累加 dw[co, ci, ki] = Σ_{n, lo} dy * x
    var acc: f32 = 0.0;
    for (var n: u32 = 0u; n < N; n++) {
        for (var lo: u32 = 0u; lo < L_out; lo++) {
            let l_idx = lo * stride + ki * dilation - padding;
            if l_idx >= 0 && l_idx < L_in {
                let dy_pos = (n * C_out + co) * L_out + lo;
                let x_pos = (n * C_in + ci) * L_in + l_idx;
                acc += dy_dw[dy_pos] * x_dw[x_pos];
            }
            // 同时累 db[co]
            if ci == 0u && ki == 0u {
                atomicAdd(&db_dw[co], dy_dw[dy_pos]);
            }
        }
    }
    dw_dw[idx] = acc;
}

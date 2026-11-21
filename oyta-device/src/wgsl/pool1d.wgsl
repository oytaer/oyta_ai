// pool1d.wgsl：MaxPool1d + AvgPool1d forward/backward
// 4 个独立 entry point

const WGS: u32 = 256u;

// ===== MaxPool1d forward =====
@group(0) @binding(0) var<storage, read> x: array<f32>;          // [N, C, L_in]
@group(0) @binding(1) var<storage, read_write> out: array<f32>;   // [N, C, L_out]
@group(0) @binding(2) var<storage, read_write> argmax: array<i32>; // [N, C, L_out]
@group(0) @binding(3) var<storage, read> dims: array<u32>;  // [NC, L_in, L_out, K, STRIDE, PAD]

@compute @workgroup_size(WGS, 1, 1)
fn maxpool_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let NC = dims[0];
    let total = NC * dims[2];  // NC * L_out
    if idx >= total { return; }
    let lo = idx % dims[2];
    let nc = idx / dims[2];
    let L_in = dims[1];
    let K = dims[3];
    let stride = dims[4];
    let pad = dims[5];

    let start = lo * stride - pad;
    let end = start + K;
    let s = max(start, 0u);
    let e = min(end, L_in);

    var mv: f32 = -3.4e+38;
    var mi: i32 = -1;
    for (var ki: u32 = s; ki < e; ki++) {
        let v = x[nc * L_in + ki];
        if v > mv { mv = v; mi = i32(ki); }
    }
    if mi < 0 { mi = 0; }
    out[idx] = mv;
    argmax[idx] = mi;
}

// ===== AvgPool1d forward =====
@group(1) @binding(0) var<storage, read> ax: array<f32>;
@group(1) @binding(1) var<storage, read_write> aout: array<f32>;
@group(1) @binding(2) var<storage, read> adims: array<u32>;

@compute @workgroup_size(WGS, 1, 1)
fn avgpool_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let NC = adims[0];
    let total = NC * adims[2];
    if idx >= total { return; }
    let lo = idx % adims[2];
    let nc = idx / adims[2];
    let L_in = adims[1];
    let K = adims[3];
    let stride = adims[4];
    let pad = adims[5];

    let start = lo * stride - pad;
    let end = start + K;
    let s = max(start, 0u);
    let e = min(end, L_in);
    let cnt = max(e - s, 1u);

    var s2: f32 = 0.0;
    for (var ki: u32 = s; ki < e; ki++) { s2 += ax[nc * L_in + ki]; }
    aout[idx] = s2 / f32(cnt);
}

// ===== MaxPool1d backward =====
@group(2) @binding(0) var<storage, read> dy: array<f32>;
@group(2) @binding(1) var<storage, read> bargmax: array<i32>;
@group(2) @binding(2) var<storage, read_write> dx: array<f32>;
@group(2) @binding(3) var<storage, read> bdims: array<u32>;  // [NC, L_out, L_in]

@compute @workgroup_size(WGS, 1, 1)
fn maxpool_bwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let NC = bdims[0];
    let L_out = bdims[1];
    let L_in = bdims[2];
    let total = NC * L_out;
    if idx >= total { return; }

    let nc = idx / L_out;
    let am = bargmax[idx];
    if am >= 0 {
        let am_u = u32(am);
        atomicAdd(&dx[nc * L_in + am_u], dy[idx]);
    }
}

// ===== AvgPool1d backward =====
@group(3) @binding(0) var<storage, read> ady: array<f32>;
@group(3) @binding(1) var<storage, read_write> adx: array<f32>;
@group(3) @binding(2) var<storage, read> abdims: array<u32>;  // [NC, L_out, L_in, K, STRIDE, PAD]

@compute @workgroup_size(WGS, 1, 1)
fn avgpool_bwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let NC = abdims[0];
    let L_out = abdims[1];
    let L_in = abdims[2];
    let K = abdims[3];
    let stride = abdims[4];
    let pad = abdims[5];
    let total = NC * L_out;
    if idx >= total { return; }

    let lo = idx % L_out;
    let nc = idx / L_out;
    let start = lo * stride - pad;
    let end = start + K;
    let s = max(start, 0u);
    let e = min(end, L_in);
    let cnt = max(e - s, 1u);
    let d = ady[idx] / f32(cnt);

    for (var ki: u32 = s; ki < e; ki++) {
        atomicAdd(&adx[nc * L_in + ki], d);
    }
}

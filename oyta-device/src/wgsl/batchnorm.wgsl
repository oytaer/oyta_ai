// batchnorm.wgsl：BatchNorm1d forward(train/eval) + backward
// 每个 entry point 独立 binding

const WGS: u32 = 256u;

// ===== BatchNorm forward (train) =====
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> x_norm: array<f32>;
@group(0) @binding(3) var<storage, read_write> rm: array<f32>;   // running_mean
@group(0) @binding(4) var<storage, read_write> rv: array<f32>;   // running_var
@group(0) @binding(5) var<storage, read> gamma: array<f32>;
@group(0) @binding(6) var<storage, read> beta: array<f32>;
@group(0) @binding(7) var<storage, read> dims: array<u32>;  // [outer, feat_dim, eps_bits, mom_bits]

@compute @workgroup_size(256, 1, 1)
fn bn_fwd_train(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = wg.x;
    let outer = dims[0];
    let feat_dim = dims[1];
    let eps = bitcast<u32, f32>(dims[2]);
    let mom = bitcast<u32, f32>(dims[3]);

    // mean (warp reduction)
    var s: f32 = 0.0;
    for (var i: u32 = lid.x; i < outer; i += WGS) { s += x[i * feat_dim + c]; }
    s = s + subgroupShuffleDown(s, 16u);
    s = s + subgroupShuffleDown(s, 8u);
    s = s + subgroupShuffleDown(s, 4u);
    s = s + subgroupShuffleDown(s, 2u);
    s = s + subgroupShuffleDown(s, 1u);

    group var smem: f32;
    if lid.x == 0 { smem = s / f32(outer); }
    workgroupBarrier();
    let mean_v = smem;
    workgroupBarrier();

    // variance
    var v: f32 = 0.0;
    for (var i: u32 = lid.x; i < outer; i += WGS) { let d = x[i * feat_dim + c] - mean_v; v += d * d; }
    v = v + subgroupShuffleDown(v, 16u) + subgroupShuffleDown(v, 8u) + subgroupShuffleDown(v, 4u) + subgroupShuffleDown(v, 2u) + subgroupShuffleDown(v, 1u);
    if lid.x == 0 { smem = v / f32(outer); }
    workgroupBarrier();
    let var_v = smem;

    let inv = 1.0 / sqrt(var_v + eps);
    let g = gamma[c];
    let b = beta[c];
    for (var i: u32 = lid.x; i < outer; i += WGS) {
        let xn = (x[i * feat_dim + c] - mean_v) * inv;
        x_norm[i * feat_dim + c] = xn;
        y[i * feat_dim + c] = xn * g + b;
    }
    if lid.x == 0 {
        rm[c] = (1.0 - mom) * rm[c] + mom * mean_v;
        rv[c] = (1.0 - mom) * rv[c] + mom * var_v;
    }
}

// ===== BatchNorm forward (eval) =====
@group(1) @binding(0) var<storage, read> ex: array<f32>;
@group(1) @binding(1) var<storage, read_write> ey: array<f32>;
@group(1) @binding(2) var<storage, read_write> exn: array<f32>;
@group(1) @binding(3) var<storage, read> erm: array<f32>;
@group(1) @binding(4) var<storage, read> erv: array<f32>;
@group(1) @binding(5) var<storage, read> egamma: array<f32>;
@group(1) @binding(6) var<storage, read> ebeta: array<f32>;
@group(1) @binding(7) var<storage, read> edims: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn bn_fwd_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let outer = edims[0];
    let feat_dim = edims[1];
    let total = outer * feat_dim;
    if idx >= total { return; }
    let c = idx % feat_dim;
    let mean_v = erm[c];
    let var_v = erv[c];
    let inv = 1.0 / sqrt(var_v + bitcast<u32, f32>(edims[2]));
    let xn = (ex[idx] - mean_v) * inv;
    exn[idx] = xn;
    ey[idx] = xn * egamma[c] + ebeta[c];
}

// ===== BatchNorm backward =====
@group(2) @binding(0) var<storage, read> dy: array<f32>;
@group(2) @binding(1) var<storage, read> bxn: array<f32>;
@group(2) @binding(2) var<storage, read_write> bdx: array<f32>;
@group(2) @binding(3) var<storage, read_write> bdgamma: array<f32>;
@group(2) @binding(4) var<storage, read_write> bdbeta: array<f32>;
@group(2) @binding(5) var<storage, read> bgamma: array<f32>;
@group(2) @binding(6) var<storage, read> bdims: array<u32>;  // [outer, feat_dim, ...]

@compute @workgroup_size(256, 1, 1)
fn bn_bwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = wg.x;
    let outer = bdims[0];
    let feat_dim = bdims[1];

    var db_sum: f32 = 0.0;
    var dg_sum: f32 = 0.0;
    for (var i: u32 = lid.x; i < outer; i += WGS) {
        let dyi = dy[i * feat_dim + c];
        db_sum += dyi;
        dg_sum += dyi * bxn[i * feat_dim + c];
    }
    db_sum = db_sum + subgroupShuffleDown(db_sum, 16u) + subgroupShuffleDown(db_sum, 8u) + subgroupShuffleDown(db_sum, 4u) + subgroupShuffleDown(db_sum, 2u) + subgroupShuffleDown(db_sum, 1u);
    dg_sum = dg_sum + subgroupShuffleDown(dg_sum, 16u) + subgroupShuffleDown(dg_sum, 8u) + subgroupShuffleDown(dg_sum, 4u) + subgroupShuffleDown(dg_sum, 2u) + subgroupShuffleDown(dg_sum, 1u);
    if lid.x == 0 { bdbeta[c] = db_sum; bdgamma[c] = dg_sum; }

    // dx = dy * gamma（简化版）
    let g = bgamma[c];
    for (var i: u32 = lid.x; i < outer; i += WGS) {
        bdx[i * feat_dim + c] = dy[i * feat_dim + c] * g;
    }
}

//! Profiling：算子耗时统计（§31.4c）
//!
//! 设计：
//!   - 固定大小数组 + 算子名→索引映射（避免 HashMap 动态分配）
//!   - 每个统计项独立 AtomicU64（无锁，比 Mutex 快 10-100x）
//!   - min_ns 用 CAS 循环更新（fetch_min 稳定版还没到）

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering}; // 原子操作（无锁）
use std::sync::OnceLock; // 线程安全延迟初始化

// ============================================================
// 全局开关 + 常量
// ============================================================

/// 全局 profiling 开关（默认关，零额外开销）
static PROFILING_ENABLED: AtomicBool = AtomicBool::new(false);

/// 最大算子数（固定 256，避免 HashMap 动态分配）
pub const MAX_OPS: usize = 256;

// ============================================================
// OpStats：单个算子的统计槽位
// ============================================================

/// 每个算子的统计槽位（全部 AtomicU64 无锁）
#[repr(C)] // 保证 C 布局（方便未来 FFI）
struct OpStats {
    name: &'static str,  // 静态算子名（'static 生命周期，编译期确定）
    count: AtomicU64,    // 调用次数
    total_ns: AtomicU64, // 累计耗时（ns）
    min_ns: AtomicU64,   // 最小耗时（ns）
    max_ns: AtomicU64,   // 最大耗时（ns）
}

impl OpStats {
    /// const fn 构造（编译期可用，OnceLock 里静态初始化）
    const fn new(name: &'static str) -> Self {
        Self {
            name,
            count: AtomicU64::new(0),
            total_ns: AtomicU64::new(0),
            min_ns: AtomicU64::new(u64::MAX), // min 初始为 MAX（第一个值自动成为 min）
            max_ns: AtomicU64::new(0),         // max 初始为 0
        }
    }

    /// 记录一次调用耗时（Atomic 无锁更新）
    fn record(&self, elapsed_ns: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);       // count++（Relaxed 足够）
        self.total_ns.fetch_add(elapsed_ns, Ordering::Relaxed); // total += elapsed
        // min = min(min, elapsed) — CAS 循环（fetch_min 稳定版还没到）
        loop {
            let cur_min = self.min_ns.load(Ordering::Relaxed);
            if elapsed_ns >= cur_min { break; } // 不是更小 → 跳过
            if self.min_ns.compare_exchange_weak(cur_min, elapsed_ns,
                Ordering::Relaxed, Ordering::Relaxed).is_ok() { break; } // CAS 成功
        }
        // max = max(max, elapsed) — 同上 CAS 循环
        loop {
            let cur_max = self.max_ns.load(Ordering::Relaxed);
            if elapsed_ns <= cur_max { break; }
            if self.max_ns.compare_exchange_weak(cur_max, elapsed_ns,
                Ordering::Relaxed, Ordering::Relaxed).is_ok() { break; }
        }
    }

    /// 快照当前值（非原子一致性，但统计允许微小偏差）
    fn snapshot(&self) -> (u64, u64, u64, u64) {
        (
            self.count.load(Ordering::Relaxed),
            self.total_ns.load(Ordering::Relaxed),
            self.min_ns.load(Ordering::Relaxed),
            self.max_ns.load(Ordering::Relaxed),
        )
    }
}

// ============================================================
// global_stats：OnceLock 延迟初始化的全局统计数组
// ============================================================

/// 全局统计数组（OnceLock 延迟初始化，固定大小 256）
fn global_stats() -> &'static [OpStats; MAX_OPS] {
    static STATS: OnceLock<[OpStats; MAX_OPS]> = OnceLock::new();
    STATS.get_or_init(|| {
        // 预分配常用算子名（前 49 个），其余全是 reserved（重复 207 次）
        const R: OpStats = OpStats::new("reserved");
        let make = |n: &'static str| OpStats::new(n); // 闭包工厂
        [
            // matmul 族（5）
            make("matmul"), make("matmul_t"), make("matmul_batched"),
            make("matmul_f16"), make("matmul_bf16"),
            // elwise 二元（4）
            make("elwise_add"), make("elwise_sub"), make("elwise_mul"), make("elwise_div"),
            // elwise 一元（12）
            make("elwise_neg"), make("elwise_abs"), make("elwise_exp"), make("elwise_log"),
            make("elwise_sqrt"), make("elwise_pow"), make("elwise_sin"), make("elwise_cos"),
            make("elwise_relu"), make("elwise_gelu"), make("elwise_gelu_fast"),
            make("elwise_sigmoid"), make("elwise_tanh"), make("elwise_silu"),
            // broadcast（1）
            make("elwise_add_broadcast"),
            // inplace（5）
            make("elwise_add_inplace"), make("elwise_mul_inplace"),
            make("elwise_scalar_add_inplace"), make("elwise_scalar_mul_inplace"),
            make("elwise_relu_inplace"),
            // reduce（4）
            make("reduce_sum"), make("reduce_mean"), make("reduce_max"), make("reduce_min"),
            // 归一化（4）
            make("softmax"), make("layernorm"), make("layernorm_inplace"), make("rmsnorm"),
            // attention（2）
            make("flash_attention"), make("flash_attention_batched"),
            // 融合算子（4）
            make("layernorm_gelu"), make("linear_bias_gelu"),
            make("rmsnorm_swiglu"), make("matmul_activation"),
            // 其他（3）
            make("fill"), make("copy"), make("where_bool"),
            // 形状变换（3）
            make("transpose"), make("reshape"), make("contiguous"),
            // reserved 补齐（256 - 49 = 207 个）
            R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R,
            R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R,
            R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R,
            R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R,
            R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R,
            R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R,
            R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R,
            R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R,
            R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R,
            R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R,
            R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R,
            R, R, R, R, R, R, R, R, R, R, R, R, R, R, R,
            R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R,
        ]
    })
}

// ============================================================
// op_index：算子名 → 预分配索引的 O(1) 查找
// ============================================================

/// 算子名 → 预分配索引（硬编码 match，O(1)，避免 HashMap）
pub fn op_index(name: &'static str) -> usize {
    match name {
        // matmul 族
        "matmul" => 0, "matmul_t" => 1, "matmul_batched" => 2,
        "matmul_f16" => 3, "matmul_bf16" => 4,
        // elwise 二元
        "elwise_add" => 5, "elwise_sub" => 6, "elwise_mul" => 7, "elwise_div" => 8,
        // elwise 一元
        "elwise_neg" => 9, "elwise_abs" => 10,
        "elwise_exp" => 11, "elwise_log" => 12, "elwise_sqrt" => 13,
        "elwise_pow" => 14, "elwise_sin" => 15, "elwise_cos" => 16,
        "elwise_relu" => 17, "elwise_gelu" => 18, "elwise_gelu_fast" => 19,
        "elwise_sigmoid" => 20, "elwise_tanh" => 21, "elwise_silu" => 22,
        // broadcast
        "elwise_add_broadcast" => 23,
        // inplace
        "elwise_add_inplace" => 24, "elwise_mul_inplace" => 25,
        "elwise_scalar_add_inplace" => 26, "elwise_scalar_mul_inplace" => 27,
        "elwise_relu_inplace" => 28,
        // reduce
        "reduce_sum" => 29, "reduce_mean" => 30,
        "reduce_max" => 31, "reduce_min" => 32,
        // 归一化
        "softmax" => 33, "layernorm" => 34, "layernorm_inplace" => 35, "rmsnorm" => 36,
        // attention
        "flash_attention" => 37, "flash_attention_batched" => 38,
        // 融合
        "layernorm_gelu" => 39, "linear_bias_gelu" => 40,
        "rmsnorm_swiglu" => 41, "matmul_activation" => 42,
        // 其他
        "fill" => 43, "copy" => 44, "where_bool" => 45,
        // 形状变换
        "transpose" => 46, "reshape" => 47, "contiguous" => 48,
        _ => MAX_OPS - 1, // 未知算子全映射到最后一个槽位（reserved）
    }
}

// ============================================================
// ProfileGuard：RAII guard，Drop 时自动 record
// ============================================================

/// RAII guard（创建时记 start time，Drop 时 AtomicU64 record）
pub struct ProfileGuard {
    idx: usize,              // 预分配索引
    start: std::time::Instant, // 开始时间点
    enabled: bool,           // 是否开启
}

impl Drop for ProfileGuard {
    fn drop(&mut self) {
        if self.enabled { // profiling 开启才 record
            // as_nanos() 返回 u128；这里 saturating 截断到 u64（292 年的 ns 才会溢出 u64，基本安全）
            let elapsed = self.start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            global_stats()[self.idx].record(elapsed); // 无锁 Atomic record
        }
    }
}

/// 创建 Profiling guard（RAII，Drop 时自动记录耗时）
///
/// 如果全局 profiling 关闭，guard 零额外开销（enabled=false）
pub fn profile(name: &'static str) -> ProfileGuard {
    let enabled = PROFILING_ENABLED.load(Ordering::Relaxed); // 只读开关
    ProfileGuard {
        idx: op_index(name),          // O(1) 查索引
        start: std::time::Instant::now(), // 记开始
        enabled,                       // 保存开关状态
    }
}

// ============================================================
// 全局控制 API（set / print / reset）
// ============================================================

/// 开启 / 关闭 profiling（全局 AtomicBool）
pub fn set_profiling_enabled(enabled: bool) {
    PROFILING_ENABLED.store(enabled, Ordering::Relaxed);
}

/// 打印 Profiling 报告（遍历固定数组，只读 AtomicU64）
pub fn print_profile_report() {
    let stats = global_stats();
    println!("\n========== Profiling Report (无锁 Atomic 版) ==========");
    println!("{:<24} {:>8} {:>12} {:>10} {:>10} {:>10}",
        "算子", "次数", "总耗时(ms)", "最小(us)", "最大(us)", "平均(us)");
    println!("{}", "-".repeat(78));
    let mut printed = 0usize;
    for s in stats.iter() {
        let (cnt, total, mn, mx) = s.snapshot(); // 非原子快照（统计允许微小偏差）
        if cnt > 0 && s.name != "reserved" && !s.name.starts_with("other_") {
            let avg = total / cnt;
            println!("{:<24} {:>8} {:>12.3} {:>10.3} {:>10.3} {:>10.3}",
                s.name,
                cnt,
                total as f64 / 1_000_000.0,         // ns → ms
                (if mn == u64::MAX { 0 } else { mn }) as f64 / 1_000.0, // ns → us（空槽位 max→0）
                mx as f64 / 1_000.0,                  // ns → us
                avg as f64 / 1_000.0,                 // ns → us
            );
            printed += 1;
        }
    }
    if printed == 0 {
        println!("(无 Profiling 数据 — set_profiling_enabled(true) 打开)");
    }
    println!("====================================================\n");
}

/// 重置所有统计（遍历数组，每个槽位清零）
pub fn reset_profile() {
    for s in global_stats().iter() {
        s.count.store(0, Ordering::Relaxed);
        s.total_ns.store(0, Ordering::Relaxed);
        s.min_ns.store(u64::MAX, Ordering::Relaxed); // min 复位为 MAX
        s.max_ns.store(0, Ordering::Relaxed);
    }
}

/// 查询 profiling 是否开启
pub fn is_profiling_enabled() -> bool {
    PROFILING_ENABLED.load(Ordering::Relaxed)
}

/// 49 个预分配算子名（和 global_stats 前 49 槽位一一对应）
pub const OP_NAMES: &[&str] = &[
    // matmul 族（5）
    "matmul", "matmul_t", "matmul_batched", "matmul_f16", "matmul_bf16",
    // elwise 二元（4）
    "elwise_add", "elwise_sub", "elwise_mul", "elwise_div",
    // elwise 一元（12）
    "elwise_neg", "elwise_abs", "elwise_exp", "elwise_log",
    "elwise_sqrt", "elwise_pow", "elwise_sin", "elwise_cos",
    "elwise_relu", "elwise_gelu", "elwise_gelu_fast",
    "elwise_sigmoid", "elwise_tanh", "elwise_silu",
    // broadcast（1）
    "elwise_add_broadcast",
    // inplace（5）
    "elwise_add_inplace", "elwise_mul_inplace",
    "elwise_scalar_add_inplace", "elwise_scalar_mul_inplace",
    "elwise_relu_inplace",
    // reduce（4）
    "reduce_sum", "reduce_mean", "reduce_max", "reduce_min",
    // 归一化（4）
    "softmax", "layernorm", "layernorm_inplace", "rmsnorm",
    // attention（2）
    "flash_attention", "flash_attention_batched",
    // 融合算子（4）
    "layernorm_gelu", "linear_bias_gelu",
    "rmsnorm_swiglu", "matmul_activation",
    // 其他（3）
    "fill", "copy", "where_bool",
    // 形状变换（3）
    "transpose", "reshape", "contiguous",
];

/// 算子统计快照（给 snapshot() 返回）
#[derive(Debug, Clone)]
pub struct OpStatsSnapshot {
    pub count: u64,
    pub total_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
}

impl OpStatsSnapshot {
    pub fn avg_ns(&self) -> u64 {
        if self.count == 0 { 0 } else { self.total_ns / self.count }
    }
}

/// 取当前 profiling 快照（非原子一致性，统计场景可接受）
///
/// 返回 (算子名, 统计快照)，过滤掉 reserved 和 count=0 的槽位
pub fn snapshot() -> Vec<(String, OpStatsSnapshot)> {
    let stats = global_stats();
    let mut out = Vec::new();
    for s in stats.iter() {
        let (cnt, total, mn, mx) = s.snapshot();
        if cnt > 0 && s.name != "reserved" {
            out.push((s.name.to_string(), OpStatsSnapshot {
                count: cnt, total_ns: total,
                min_ns: if mn == u64::MAX { 0 } else { mn },
                max_ns: mx,
            }));
        }
    }
    out
}

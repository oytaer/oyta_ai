//! bytes 级序列化模块
//!
//! 提供 Tensor ↔ 原始字节的通用转换（小端序）。
//! 支持 f32/f64/F16/BF16 四种 dtype。

use std::io::{Read, Write};
use oyta_device::{DType, F16, BF16};

/// 计算一个 Tensor 需要的原始字节数
///
/// # 参数
/// - numel: 元素总数
/// - dtype: dtype（决定每个元素的字节数）
///
/// # 溢出处理
/// checked_mul 防止 numel * byte_size 溢出 usize
pub fn tensor_byte_size(numel: usize, dtype: DType) -> usize {
    crate::checked::checked_mul(numel, dtype.byte_size())
        .unwrap_or(usize::MAX)
}

/// 把 f32 切片序列化成小端序 u8 字节
pub fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// 从小端序 u8 字节反序列化成 f32
pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let numel = bytes.len() / 4;
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        let b: [u8; 4] = bytes[i * 4 .. (i + 1) * 4].try_into().unwrap_or_else(|_| [0u8; 4]);
        out.push(f32::from_le_bytes(b));
    }
    out
}

/// 把 f64 切片序列化成小端序 u8 字节
pub fn f64_to_bytes(data: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 8);
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// 从小端序 u8 字节反序列化成 f64
pub fn bytes_to_f64(bytes: &[u8]) -> Vec<f64> {
    let numel = bytes.len() / 8;
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        let b: [u8; 8] = bytes[i * 8 .. (i + 1) * 8].try_into().unwrap_or_else(|_| [0u8; 8]);
        out.push(f64::from_le_bytes(b));
    }
    out
}

/// 把 F16 切片序列化成 u16 位模式的原始字节
pub fn f16_to_bytes(data: &[F16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &v in data {
        out.extend_from_slice(&v.0.to_le_bytes());
    }
    out
}

/// 从原始字节反序列化成 F16（把 2 字节当作 u16 位模式）
pub fn bytes_to_f16(bytes: &[u8]) -> Vec<F16> {
    let numel = bytes.len() / 2;
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        let b: [u8; 2] = bytes[i * 2 .. (i + 1) * 2].try_into().unwrap_or_else(|_| [0u8; 2]);
        out.push(F16(u16::from_le_bytes(b)));
    }
    out
}

/// 把 BF16 切片序列化成 u16 位模式的原始字节
pub fn bf16_to_bytes(data: &[BF16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &v in data {
        out.extend_from_slice(&v.0.to_le_bytes());
    }
    out
}

/// 从原始字节反序列化成 BF16（把 2 字节当作 u16 位模式）
pub fn bytes_to_bf16(bytes: &[u8]) -> Vec<BF16> {
    let numel = bytes.len() / 2;
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        let b: [u8; 2] = bytes[i * 2 .. (i + 1) * 2].try_into().unwrap_or_else(|_| [0u8; 2]);
        out.push(BF16(u16::from_le_bytes(b)));
    }
    out
}

/// 把原始字节写入文件（小端序序列化）
pub fn write_bytes(path: &str, bytes: &[u8]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(bytes)?;
    Ok(())
}

/// 从文件读取原始字节
pub fn read_bytes(path: &str) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(buf)
}

//! safetensors 文件格式读写
//!
//! 格式（小端序）:
//!   [8 字节 u64] = JSON header 字节数 N
//!   [N 字节]    JSON header（BTreeMap<tensor_name, TensorHeader> + optional __metadata__）
//!   [剩余]      所有 tensor 的 raw bytes（按 header 里的 data_offsets 顺序）
//!
//! TensorHeader { dtype: "F32"/"F16"/"BF16"/"F64", shape: [usize, ...], data_offsets: [u64, u64] }
//!
//! 这是 HuggingFace safetensors 格式的完整实现。

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};
use oyta_device::DType;

/// safetensors header 中单个 tensor 的描述
#[derive(Debug, Clone)]
pub struct TensorHeader {
    /// safetensors dtype 字符串（"F32"/"F16"/"BF16"/"F64"）
    pub dtype: String,
    /// 逻辑形状
    pub shape: Vec<usize>,
    /// 数据在文件中的绝对偏移量 [start, end)（相对于 header 之后的数据区起点）
    pub data_offsets: [u64; 2],
}

/// safetensors header（所有 tensor 的元数据）
#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    /// tensor 名 → header
    pub tensors: BTreeMap<String, TensorHeader>,
    /// 可选的元数据（键值对）
    pub metadata: Option<BTreeMap<String, String>>,
}

/// 保存一个 safetensors 文件
///
/// # 参数
/// - tensors: tensor 名 → (raw_bytes, shape, dtype)
/// - path: 输出文件路径
///
/// # 返回
/// io::Error 写入失败时返回
pub fn save_safetensors(
    tensors: &BTreeMap<String, (Vec<u8>, Vec<usize>, DType)>,
    path: &str,
) -> std::io::Result<()> {
    let mut json_map = BTreeMap::new();
    let mut current_offset: u64 = 0;
    let mut tensor_bytes_all: Vec<u8> = Vec::new();

    for (name, (bytes, shape, dtype)) in tensors {
        let len = bytes.len() as u64;
        let start = current_offset;
        let end = start + len;
        json_map.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype.safetensors_name(),
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
        tensor_bytes_all.extend_from_slice(bytes);
        current_offset = end;
    }

    let header_json = serde_json::to_string(&json_map)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let header_len = header_json.len() as u64;

    let mut file = std::fs::File::create(path)?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header_json.as_bytes())?;
    file.write_all(&tensor_bytes_all)?;
    Ok(())
}

/// 从 safetensors 文件读取
///
/// # 返回
/// - header: SafetensorsHeader（所有 tensor 元数据）
/// - 元组列表 (name, raw_bytes, dtype_str, shape)
pub fn load_safetensors(
    path: &str,
) -> std::io::Result<(
    SafetensorsHeader,
    Vec<(String, Vec<u8>, String, Vec<usize>)>,
)> {
    let mut file = std::fs::File::open(path)?;
    let file_size = file.metadata()?.len();

    // 1. 读取 8 字节 header 长度（小端序）
    let mut header_len_buf = [0u8; 8];
    file.read_exact(&mut header_len_buf)?;
    let header_len_u64 = u64::from_le_bytes(header_len_buf);

    // 安全边界：header_len 不能超过文件大小（不可能），也不能超过 128MiB（合理上限）
    const MAX_HEADER_LEN: u64 = 128 * 1024 * 1024;
    if header_len_u64 > file_size.saturating_sub(8) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("header_len {} 超过剩余文件大小 {}", header_len_u64, file_size.saturating_sub(8)),
        ));
    }
    if header_len_u64 > MAX_HEADER_LEN {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("header_len {} 超过安全上限 128MiB", header_len_u64),
        ));
    }
    let header_len = header_len_u64 as usize; // 现在安全了

    // 2. 读取 JSON header
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header_json = String::from_utf8(header_buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let header: serde_json::Value = serde_json::from_str(&header_json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    // 3. 解析 header
    let mut header_out = SafetensorsHeader {
        tensors: BTreeMap::new(),
        metadata: None,
    };
    let mut result = Vec::new();

    let data_start = 8 + header_len; // 文件中数据区起点

    if let Some(obj) = header.as_object() {
        for (name, info) in obj {
            // 跳过 __metadata__
            if name == "__metadata__" {
                let mut meta = BTreeMap::new();
                if let Some(meta_obj) = info.as_object() {
                    for (k, v) in meta_obj {
                        if let Some(s) = v.as_str() {
                            meta.insert(k.clone(), s.to_string());
                        }
                    }
                }
                header_out.metadata = Some(meta);
                continue;
            }

            let offsets = info["data_offsets"]
                .as_array()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "data_offsets 不是数组"))?;
            if offsets.len() != 2 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "data_offsets 必须有 [start, end] 两个元素",
                ));
            }
            let start = offsets[0]
                .as_u64()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "data_offsets[0] 不是 u64"))?;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "data_offsets[1] 不是 u64"))?;

            // === 安全校验 A：start ≤ end，且差值不溢出/不为负 ===
            if start > end {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("data_offsets start({}) > end({})", start, end),
                ));
            }
            let tensor_bytes = end - start; // u64，安全（start ≤ end）
            const MAX_TENSOR_BYTES: u64 = 16 * 1024 * 1024 * 1024; // 16 GiB 上限
            if tensor_bytes > MAX_TENSOR_BYTES {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("tensor 字节数 {} 超过安全上限 16GiB", tensor_bytes),
                ));
            }

            let shape: Vec<usize> = info["shape"]
                .as_array()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "shape 不是数组"))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();

            // === 安全校验 B：shape.numel 不溢出 + dtype.byte_size × numel == tensor_bytes ===
            let numel = crate::checked::checked_product(shape.iter().copied())
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("shape 乘积溢出: {}", e)))?;

            let dtype_str = info["dtype"]
                .as_str()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "dtype 不是字符串"))?
                .to_string();

            // 估算 dtype byte size（从 dtype_str 推断，避免依赖完整 DType 模块在 safetensors 里）
            let byte_size: usize = match dtype_str.as_str() {
                "F32" | "BF16" | "F16" | "F64" => {
                    let mapped = match dtype_str.as_str() {
                        "F32" => 4usize,
                        "F16" | "BF16" => 2usize,
                        "F64" => 8usize,
                        _ => unreachable!(),
                    };
                    let expected_bytes = crate::checked::checked_mul(numel, mapped)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("shape × dtype_size 溢出: {}", e)))?;
                    if expected_bytes as u64 != tensor_bytes {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!(
                                "tensor {} 预期 {} 字节 (numel={} × dtype={} 字节), 但 data_offsets 给出 {}",
                                name, numel as u64 * mapped as u64, numel, mapped, tensor_bytes
                            ),
                        ));
                    }
                    mapped
                }
                other => {
                    // 未知 dtype：跳过 byte_size 一致性校验，保守通过
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("未知 dtype: {}", other),
                    ));
                }
            };
            let _ = byte_size; // 已校验过了

            // === 安全校验 C：end（绝对偏移）≤ 文件实际大小 ===
            let abs_end = data_start as u64 + end;
            if abs_end > file_size {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("tensor {} 数据区终点 {} 超过文件大小 {}", name, abs_end, file_size),
                ));
            }

            header_out.tensors.insert(
                name.clone(),
                TensorHeader {
                    dtype: dtype_str.clone(),
                    shape: shape.clone(),
                    data_offsets: [start, end],
                },
            );

            // seek 到绝对位置
            let abs_pos = data_start as u64 + start;
            file.seek(SeekFrom::Start(abs_pos))?;
            let mut bytes = vec![0u8; (end - start) as usize];
            file.read_exact(&mut bytes)?;

            result.push((name.clone(), bytes, dtype_str, shape));
        }
    }

    Ok((header_out, result))
}

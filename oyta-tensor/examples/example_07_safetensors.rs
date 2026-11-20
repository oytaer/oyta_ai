//! example_07_safetensors.rs — HuggingFace safetensors 完整读写
//!
//! 展示：8 字节头部 + BTreeMap + data_offsets + extract_bytes

use oyta_tensor::*;
use std::collections::BTreeMap;

fn main() {
    println!("========================================");
    println!("  example_07: safetensors 完整读写");
    println!("========================================");

    // 1) 准备几个 Tensor 模拟模型权重
    println!("\n1. 准备模型权重 Tensor:");
    let w1 = Tensor::<f32, Cpu>::randn([768, 768]);
    let b1 = Tensor::<f32, Cpu>::zeros([768]);
    let w2 = Tensor::<f32, Cpu>::randn([768, 3072]);
    let b2 = Tensor::<f32, Cpu>::zeros([3072]);

    println!("   w1: shape={}, numel={}", w1.shape(), w1.numel());
    println!("   b1: shape={}, numel={}", b1.shape(), b1.numel());
    println!("   w2: shape={}, numel={}", w2.shape(), w2.numel());
    println!("   b2: shape={}, numel={}", b2.shape(), b2.numel());

    // 2) 保存为 safetensors
    // safetensors API 要求传入 BTreeMap<String, (data, shape, dtype)>
    // 这里用 from_raw_bytes / to_raw_bytes 桥接
    println!("\n2. 保存 safetensors:");
    let temp_path = "/tmp/oyta_example.safetensors";

    // 保存：用 save_safetensors（需要构造 BTreeMap）
    // 简化示例：直接用 Tensor 的 to_raw_bytes
    let mut tensors_map: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    tensors_map.insert("model.layers.0.self_attn.q_proj.weight".into(), w1.to_raw_bytes());
    tensors_map.insert("model.layers.0.self_attn.q_proj.bias".into(), b1.to_raw_bytes());
    tensors_map.insert("model.layers.0.mlp.gate_proj.weight".into(), w2.to_raw_bytes());
    tensors_map.insert("model.layers.0.mlp.gate_proj.bias".into(), b2.to_raw_bytes());

    // 手动写 safetensors 格式（8 字节头部 + JSON + raw bytes）
    let mut header_tensors: BTreeMap<String, SafetensorsHeader> = BTreeMap::new();
    let mut offset = 0u64;
    for (name, bytes) in &tensors_map {
        let byte_len = bytes.len() as u64;
        header_tensors.insert(name.clone(), SafetensorsHeader {
            dtype: "F32".into(),
            shape: vec![bytes.len() as usize / 4], // 简化示例
            data_offsets: [offset, offset + byte_len],
        });
        offset += byte_len;
    }

    let mut header_json = serde_json::json!({});
    for (name, h) in &header_tensors {
        header_json[name] = serde_json::json!({
            "dtype": h.dtype,
            "shape": h.shape,
            "data_offsets": h.data_offsets,
        });
    }

    let header_bytes = serde_json::to_vec(&header_json).unwrap();
    let header_len = header_bytes.len() as u64;

    use std::fs::File;
    use std::io::Write;
    let mut file = File::create(temp_path).unwrap();
    file.write_all(&header_len.to_le_bytes()).unwrap(); // 8 字节头部长度
    file.write_all(&header_bytes).unwrap();              // JSON header
    for bytes in tensors_map.values() {
        file.write_all(bytes).unwrap();                  // raw bytes
    }
    println!("   保存成功: {}", temp_path);
    println!("   文件大小: {} bytes", 8 + header_bytes.len() + offset as usize);

    // 3) 读回验证
    println!("\n3. 读回 safetensors:");
    use std::io::Read;
    let mut file = File::open(temp_path).unwrap();
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).unwrap();
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    println!("   头部长度: {} bytes", header_len);

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf).unwrap();
    let header: serde_json::Value = serde_json::from_slice(&header_buf).unwrap();
    println!("   JSON 头部 keys: {:?}", header.as_object().unwrap().keys().collect::<Vec<_>>());

    // 读取 Tensor 数据
    let file_start = 8 + header_len;
    let mut loaded_tensors: Vec<(String, Vec<u8>)> = Vec::new();
    for (name, value) in header.as_object().unwrap() {
        if name == "__metadata__" { continue; }
        let offsets = value["data_offsets"].as_array().unwrap();
        let start = offsets[0].as_u64().unwrap() as usize;
        let end = offsets[1].as_u64().unwrap() as usize;
        let byte_len = end - start;

        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start((file_start + start) as u64)).unwrap();
        let mut buf = vec![0u8; byte_len];
        file.read_exact(&mut buf).unwrap();
        loaded_tensors.push((name.clone(), buf));
        println!("   加载: {:<50} {} bytes", name, byte_len);
    }

    // 4) 精确验证 roundtrip
    println!("\n4. Roundtrip 精确验证:");
    let w1_loaded = loaded_tensors.iter().find(|(n, _)| n.contains("q_proj.weight")).unwrap();
    let w1_bytes_orig = w1.to_raw_bytes();
    assert_eq!(w1_loaded.1.len(), w1_bytes_orig.len());
    assert_eq!(w1_loaded.1, w1_bytes_orig);
    println!("   w1 roundtrip 精确匹配 ✅");

    let b1_loaded = loaded_tensors.iter().find(|(n, _)| n.contains("q_proj.bias")).unwrap();
    let b1_bytes_orig = b1.to_raw_bytes();
    assert_eq!(b1_loaded.1, b1_bytes_orig);
    println!("   b1 roundtrip 精确匹配 ✅");

    // 5) Tensor::from_raw_bytes 重建
    println!("\n5. from_raw_bytes 重建 Tensor:");
    let w1_reconstructed = Tensor::<f32, Cpu>::from_raw_bytes(
        &w1_loaded.1, w1.shape().clone()
    ).unwrap();
    assert_eq!(w1_reconstructed.data_slice(), w1.data_slice());
    println!("   ✅ from_raw_bytes 重建精确匹配");

    // 清理
    std::fs::remove_file(temp_path).unwrap();

    println!("\n✅ 示例 07 完成");
}

// safetensors header 辅助结构
#[derive(Debug, Clone)]
struct SafetensorsHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

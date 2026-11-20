//! memory_pool.rs：设备内存池（#[cfg(feature="memory_pool")]）

#[cfg(feature = "memory_pool")]
use std::sync::Mutex;

#[cfg(feature = "memory_pool")]
use crate::backend::aggregate::DeviceError;

// ============================================================
// DeviceMemoryPool：GPU 显存专用内存池（feature gate）
// ============================================================

#[cfg(feature = "memory_pool")]
#[derive(Debug, Clone)]
pub(crate) struct Chunk {
    pub offset: usize,
    pub len: usize,
}

/// DeviceMemoryPool：GPU 显存专用内存池
#[cfg(feature = "memory_pool")]
pub struct DeviceMemoryPool {
    base_ptr: usize,
    total_bytes: usize,
    allocated: Mutex<Vec<Chunk>>,
    freed: Mutex<Vec<Chunk>>,
    next_alloc: Mutex<usize>,
    alignment: usize,
    device_id: usize,
}

#[cfg(feature = "memory_pool")]
impl DeviceMemoryPool {
    /// 构造内存池（CPU 上 base_ptr=0 表示"走系统分配"）
    pub fn new(device_id: usize, total_bytes: usize, alignment: usize) -> Self {
        Self {
            base_ptr: 0,
            total_bytes,
            allocated: Mutex::new(Vec::new()),
            freed: Mutex::new(Vec::new()),
            next_alloc: Mutex::new(0),
            alignment: alignment.max(1),
            device_id,
        }
    }
    
    /// 计算对齐后的 size
    fn align_up(&self, bytes: usize) -> usize {
        (bytes + self.alignment - 1) / self.alignment * self.alignment
    }
    
    /// 分配：先查 freed first-fit，找不到从 arena 切新的
    ///
    /// 返回 Ok(offset) 或 Err(DeviceError)
    pub fn allocate(&self, bytes: usize) -> Result<usize, DeviceError> {
        let aligned = self.align_up(bytes);
        
        // 策略 1：freed first-fit
        let mut freed = self.freed.lock().unwrap_or_else(|e| e.into_inner());
        for (idx, chunk) in freed.iter().enumerate() {
            if chunk.len >= aligned {
                let offset = chunk.offset;
                let actual_len = chunk.len;
                freed.remove(idx);
                
                // 分裂（剩余 ≥ alignment 就放回 freed）
                if actual_len - aligned >= self.alignment {
                    freed.push(Chunk { offset: offset + aligned, len: actual_len - aligned });
                }
                self.allocated.lock().unwrap_or_else(|e| e.into_inner())
                    .push(Chunk { offset, len: aligned });
                return Ok(offset);
            }
        }
        
        // 策略 2：arena 切新的
        let mut next = self.next_alloc.lock().unwrap_or_else(|e| e.into_inner());
        let aligned_next = self.align_up(*next);
        if aligned_next.checked_add(aligned).map_or(true, |v| v > self.total_bytes) {
            return Err(DeviceError::InitFailed(format!(
                "DeviceMemoryPool 耗尽！已分配 {} / {} 字节，device={}",
                aligned_next, self.total_bytes, self.device_id
            )));
        }
        let offset = aligned_next;
        *next = aligned_next + aligned;
        self.allocated.lock().unwrap_or_else(|e| e.into_inner())
            .push(Chunk { offset, len: aligned });
        Ok(offset)
    }
    
    /// 释放（不真释放显存，只标记可复用）
    pub fn deallocate(&self, offset: usize) {
        let mut allocated = self.allocated.lock().unwrap_or_else(|e| e.into_inner());
        let idx = allocated.iter().position(|c| c.offset == offset);
        if let Some(i) = idx {
            let chunk = allocated.remove(i);
            self.freed.lock().unwrap_or_else(|e| e.into_inner()).push(chunk);
        }
    }
    
    /// offset → 实际设备指针
    pub fn to_device_ptr(&self, offset: usize) -> *mut u8 {
        (self.base_ptr + offset) as *mut u8
    }
    
    /// 训练 epoch 结束时重置整个池
    pub fn reset(&self) {
        self.allocated.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.freed.lock().unwrap_or_else(|e| e.into_inner()).clear();
        *self.next_alloc.lock().unwrap_or_else(|e| e.into_inner()) = 0;
    }
}

#[cfg(feature = "memory_pool")]
impl Default for DeviceMemoryPool {
    fn default() -> Self { Self::new(0, 1024 * 1024 * 1024, 256) } // 默认 1GB, 256 对齐
}

//! example_01_tensor_basics.rs — Tensor 基础使用
//!
//! 展示：Tensor 构造、属性访问、基本运算、工厂函数、形状变换
//! 这是所有用户的第一个入口点

use oyta_tensor::*;

fn main() {
    println!("========================================");
    println!("  example_01: Tensor 基础使用");
    println!("========================================");

    // 1) 从 Vec 构造 Tensor — 最常用的方式
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f32, Cpu>::from_data(data, [2, 3]).unwrap();
    println!("\n1. 构造 Tensor::from_data([2,3]):");
    println!("   shape={:?}, rank={}, numel={}", t.shape(), t.rank(), t.numel());
    println!("   strides={:?}", t.strides());
    println!("   data={:?}", t.data_slice());
    println!("   is_contiguous={}", t.is_contiguous());

    // 2) 工厂函数 — zeros / ones / full / arange / scalar / randn
    println!("\n2. 工厂函数:");
    let z = Tensor::<f32, Cpu>::zeros([2, 3]);
    println!("   zeros([2,3]): {:?}", z.data_slice());
    let o = Tensor::<f32, Cpu>::ones([3, 2]);
    println!("   ones([3,2]):  {:?}", o.data_slice());
    let f = Tensor::<f32, Cpu>::full([2, 2], 42.0);
    println!("   full([2,2], 42.0): {:?}", f.data_slice());
    let ar = Tensor::<f32, Cpu>::arange(0.0, 5.0, 1.0);
    println!("   arange(0..5 step 1): {:?}", ar.data_slice());
    let sc = Tensor::<f32, Cpu>::scalar(3.14);
    println!("   scalar(3.14): item={}, is_scalar={}", sc.item(), sc.is_scalar());

    // 3) 基本二元算子 — add / sub / mul / div
    println!("\n3. 二元算子:");
    let a = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    let b = Tensor::<f32, Cpu>::from_data(vec![10.0, 20.0, 30.0, 40.0], [2, 2]).unwrap();
    println!("   add: {:?}", a.add(&b).unwrap().data_slice());
    println!("   sub: {:?}", a.sub(&b).unwrap().data_slice());
    println!("   mul: {:?}", a.mul(&b).unwrap().data_slice());
    println!("   div: {:?}", a.div(&b).unwrap().data_slice());

    // 4) 一元激活函数 — 直接返回 Tensor，零 unwrap
    println!("\n4. 一元激活:");
    let x = Tensor::<f32, Cpu>::from_data(vec![-1.0, 0.0, 1.0, 2.0], [4]).unwrap();
    println!("   relu:   {:?}", x.relu().unwrap().data_slice());
    println!("   gelu:   {:?}", x.gelu().unwrap().data_slice());
    println!("   sigmoid(0)={:.4}", x.sigmoid().data_slice()[1]);
    println!("   tanh(0)={:.4}", x.tanh().data_slice()[1]);
    println!("   exp(1)={:.4}", x.exp().data_slice()[2]);
    println!("   sqrt(4)={:.4}", Tensor::<f32, Cpu>::from_data(vec![4.0], [1]).unwrap().sqrt().data_slice()[0]);

    // 5) 形状变换 — reshape(-1 推断) / transpose / squeeze
    println!("\n5. 形状变换:");
    let big = Tensor::<f32, Cpu>::arange(0.0, 12.0, 1.0);
    let r = big.reshape([3, 4]).unwrap();
    println!("   arange(0..12) reshape [3,4]: shape={:?}", r.shape());
    let _ = big.reshape([2usize, 6]).unwrap(); // -1 自动推断为 2 → 实际显式给出 2
    let tt = r.transpose(0, 1);
    println!("   transpose(0,1): shape={:?}, is_contiguous={}", tt.shape(), tt.is_contiguous());
    let s = tt.contiguous();
    println!("   .contiguous(): is_contiguous={}", s.is_contiguous());

    // 6) 规约算子 — sum / mean / max / min（沿 axis）
    println!("\n6. 规约算子:");
    let m = Tensor::<f32, Cpu>::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]
    ).unwrap();
    println!("   全局 sum:   {}", m.sum().unwrap());
    println!("   全局 mean:  {}", m.mean(None).unwrap().item());
    println!("   全局 max:   {}", m.max(None).unwrap().item());
    println!("   沿 axis=0 mean: {:?}", m.mean(Some(0)).unwrap().data_slice());
    println!("   沿 axis=1 max:  {:?}", m.max(Some(1)).unwrap().data_slice());

    println!("\n✅ 示例 01 完成");
}

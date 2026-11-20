//! tests.rs：Tensor 单元测试

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;
    use crate::shape::DynamicShape;
    use oyta_device::Cpu;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0, 4.0], DynamicShape::new([2, 2])).unwrap();
        assert_eq!(t.shape.dims, vec![2, 2]); assert_eq!(t.numel(), 4); assert!(t.is_contiguous());
    }

    #[test]
    fn test_zeros_ones() {
        let z = Tensor::<f32, Cpu>::zeros([3, 4]);
        assert_eq!(z.numel(), 12); assert!(z.data_slice().iter().all(|&v| v == 0.0));
        let o = Tensor::<f32, Cpu>::ones([3, 4]);
        assert!(o.data_slice().iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_tensor_transpose() {
        let t = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DynamicShape::new([2, 3])).unwrap();
        let tt = t.transpose(0, 1);
        assert_eq!(tt.shape.dims, vec![3, 2]); assert_eq!(tt.strides.0, vec![1, 3]); assert!(!tt.is_contiguous());
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::<f32, Cpu>::zeros([2, 3]);
        assert_eq!(t.reshape([6]).unwrap().shape.dims, vec![6]);
        assert_eq!(t.reshape([3, 2]).unwrap().shape.dims, vec![3, 2]);
    }

    #[test]
    fn test_matmul_small() {
        let a = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DynamicShape::new([2, 3])).unwrap();
        let b = Tensor::<f32, Cpu>::from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], DynamicShape::new([3, 2])).unwrap();
        let c = a.matmul(&b).unwrap(); let d = c.data_slice();
        assert!((d[0] - 58.0).abs() < 1e-5); assert!((d[1] - 64.0).abs() < 1e-5);
        assert!((d[2] - 139.0).abs() < 1e-5); assert!((d[3] - 154.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_identity() {
        let a = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], DynamicShape::new([3, 3])).unwrap();
        let eye = Tensor::<f32, Cpu>::from_data(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], DynamicShape::new([3, 3])).unwrap();
        let c = a.matmul(&eye).unwrap(); assert_eq!(c.data_slice(), a.data_slice());
    }

    #[test]
    fn test_softmax_row_sum() {
        let t = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], DynamicShape::new([2, 3])).unwrap();
        let s = t.softmax(-1).unwrap(); let d = s.data_slice();
        let r0: f32 = d[0..3].iter().sum(); let r1: f32 = d[3..6].iter().sum();
        assert!((r0 - 1.0).abs() < 1e-5); assert!((r1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_layernorm_mean_var() {
        let t = Tensor::<f32, Cpu>::randn([2, 4]);
        let gamma = Tensor::<f32, Cpu>::ones([4]); let beta = Tensor::<f32, Cpu>::zeros([4]);
        let ln = t.layernorm(&gamma, &beta, 1e-5).unwrap(); let row0: &[f32] = &ln.data_slice()[0..4];
        let mean: f32 = row0.iter().sum::<f32>() / 4.0; let var: f32 = row0.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-3); assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_add_mul() {
        let a = Tensor::<f32, Cpu>::zeros([2, 2]); let b = Tensor::<f32, Cpu>::ones([2, 2]);
        let c = a.add(&b).unwrap(); assert!(c.data_slice().iter().all(|&v| v == 1.0));
        let x = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0, 4.0], DynamicShape::new([2, 2])).unwrap();
        let y = Tensor::<f32, Cpu>::from_data(vec![2.0, 2.0, 2.0, 2.0], DynamicShape::new([2, 2])).unwrap();
        let z = x.mul(&y).unwrap(); assert_eq!(z.data_slice(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_relu_gelu() {
        let t = Tensor::<f32, Cpu>::from_data(vec![-1.0, 0.0, 1.0, 2.0], DynamicShape::new([4])).unwrap();
        let r = t.relu().unwrap(); assert_eq!(r.data_slice(), &[0.0, 0.0, 1.0, 2.0]);
        let t2 = Tensor::<f32, Cpu>::from_data(vec![-10.0, 0.0, 10.0], DynamicShape::new([3])).unwrap();
        let g = t2.gelu().unwrap(); let d = g.data_slice();
        assert!((d[0] - 0.0).abs() < 1e-3); assert!((d[2] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_sum() {
        let t = Tensor::<f32, Cpu>::ones([4]); assert!((t.sum().unwrap() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_contiguous() {
        let t = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DynamicShape::new([2, 3])).unwrap();
        assert!(t.is_contiguous());
        let tt = t.transpose(0, 1); assert!(!tt.is_contiguous());
        let c = tt.contiguous(); assert!(c.is_contiguous());
    }

    // ========== 补全：缺失的核心功能测试 ==========

    #[test]
    fn test_sub_div() {
        // sub: 5-2=3, 10-4=6, 15-6=9, 20-8=12
        let a = Tensor::<f32, Cpu>::from_data(vec![5.0, 10.0, 15.0, 20.0], DynamicShape::new([2, 2])).unwrap();
        let b = Tensor::<f32, Cpu>::from_data(vec![2.0, 4.0, 6.0, 8.0], DynamicShape::new([2, 2])).unwrap();
        let c = a.sub(&b).unwrap();
        assert_eq!(c.data_slice(), &[3.0, 6.0, 9.0, 12.0]);
        // div: 6/2=3, 8/4=2, 10/5=2, 12/6=2
        let x = Tensor::<f32, Cpu>::from_data(vec![6.0, 8.0, 10.0, 12.0], DynamicShape::new([2, 2])).unwrap();
        let y = Tensor::<f32, Cpu>::from_data(vec![2.0, 4.0, 5.0, 6.0], DynamicShape::new([2, 2])).unwrap();
        let z = x.div(&y).unwrap();
        assert_eq!(z.data_slice(), &[3.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_add_shape_mismatch_errors() {
        // Tensor.add 当前只支持 shape 精确匹配，不做广播（广播在 oyta-ops 层）
        let a = Tensor::<f32, Cpu>::from_data(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], DynamicShape::new([2, 3])).unwrap();
        let b = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0], DynamicShape::new([3])).unwrap();
        // shape 不匹配 → 应返回 Err
        let result = a.add(&b);
        assert!(result.is_err(), "shape 不匹配应返回错误");
    }

    #[test]
    fn test_mean_reduce() {
        // [2,3] 全局 mean: (1+2+3+4+5+6)/6 = 3.5
        let t = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DynamicShape::new([2, 3])).unwrap();
        let m = t.mean(None).unwrap();
        assert!((m.data_slice()[0] - 3.5).abs() < 1e-5);
    }

    #[test]
    fn test_max_min_reduce() {
        // [2,3] 全局 max = 6.0, min = 1.0
        let t = Tensor::<f32, Cpu>::from_data(vec![3.0, 1.0, 4.0, 2.0, 6.0, 5.0], DynamicShape::new([2, 3])).unwrap();
        let mx = t.max(None).unwrap();
        assert!((mx.data_slice()[0] - 6.0).abs() < 1e-5);
        let mn = t.min(None).unwrap();
        assert!((mn.data_slice()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_nonsquare() {
        // [1,3] * [3,2] = [1,2] (dot product 特化路径)
        let a = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0], DynamicShape::new([1, 3])).unwrap();
        let b = Tensor::<f32, Cpu>::from_data(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0], DynamicShape::new([3, 2])).unwrap();
        let c = a.matmul(&b).unwrap();
        // [1*4+2*6+3*8=4+12+24=40, 1*5+2*7+3*9=5+14+27=46]
        let d = c.data_slice();
        assert!((d[0] - 40.0).abs() < 1e-5);
        assert!((d[1] - 46.0).abs() < 1e-5);
    }

    #[test]
    fn test_unary_neg_abs_exp() {
        let t = Tensor::<f32, Cpu>::from_data(vec![-1.0, 0.0, 2.0], DynamicShape::new([3])).unwrap();
        // 一元方法直接返回 Tensor，不是 Result
        let n = t.neg();
        assert_eq!(n.data_slice(), &[1.0, 0.0, -2.0]);
        let a = t.abs();
        assert_eq!(a.data_slice(), &[1.0, 0.0, 2.0]);
        let e = t.exp();
        let d = e.data_slice();
        assert!((d[0] - (-1.0f32).exp()).abs() < 1e-5);
        assert!((d[2] - (2.0f32).exp()).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid_tanh() {
        let t = Tensor::<f32, Cpu>::from_data(vec![0.0], DynamicShape::new([1])).unwrap();
        let s = t.sigmoid();
        assert!((s.data_slice()[0] - 0.5).abs() < 1e-5); // sigmoid(0) = 0.5
        let t2 = Tensor::<f32, Cpu>::from_data(vec![0.0], DynamicShape::new([1])).unwrap();
        let th = t2.tanh();
        assert!((th.data_slice()[0] - 0.0).abs() < 1e-5); // tanh(0) = 0
    }

    #[test]
    fn test_log_sqrt() {
        let t = Tensor::<f32, Cpu>::from_data(vec![1.0, 4.0], DynamicShape::new([2])).unwrap();
        let l = t.log();
        assert!((l.data_slice()[0] - 0.0).abs() < 1e-5); // log(1) = 0
        let s = t.sqrt();
        assert!((s.data_slice()[1] - 2.0).abs() < 1e-5); // sqrt(4) = 2
    }
}

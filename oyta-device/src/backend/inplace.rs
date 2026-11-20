//! inplace.rs：泛型原地算子（InPlaceOps<T> 一套覆盖 f32/f16/bf16）

use crate::backend::base::BackendBase;
use crate::dtype::float_trait::Float;

pub trait InPlaceOps<T: Float>: BackendBase {
    fn elwise_add_inplace(a: *mut T, b: *const T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts_mut(a, n) };
        let b_s = unsafe { std::slice::from_raw_parts(b, n) };
        use rayon::prelude::*;
        a_s.par_iter_mut().zip(b_s.par_iter()).for_each(|(o, &x)| {
            *o = T::from_f32(o.to_f32() + x.to_f32());
        });
    }
    fn elwise_mul_inplace(a: *mut T, b: *const T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts_mut(a, n) };
        let b_s = unsafe { std::slice::from_raw_parts(b, n) };
        use rayon::prelude::*;
        a_s.par_iter_mut().zip(b_s.par_iter()).for_each(|(o, &x)| {
            *o = T::from_f32(o.to_f32() * x.to_f32());
        });
    }
    fn elwise_scalar_add_inplace(a: *mut T, n: usize, val: f32) {
        let a_s = unsafe { std::slice::from_raw_parts_mut(a, n) };
        use rayon::prelude::*;
        a_s.par_iter_mut().for_each(|o| { *o = T::from_f32(o.to_f32() + val); });
    }
    fn elwise_scalar_mul_inplace(a: *mut T, n: usize, val: f32) {
        let a_s = unsafe { std::slice::from_raw_parts_mut(a, n) };
        use rayon::prelude::*;
        a_s.par_iter_mut().for_each(|o| { *o = T::from_f32(o.to_f32() * val); });
    }
    fn elwise_relu_inplace(a: *mut T, n: usize) {
        let a_s = unsafe { std::slice::from_raw_parts_mut(a, n) };
        use rayon::prelude::*;
        a_s.par_iter_mut().for_each(|o| {
            let v = o.to_f32();
            *o = if v > 0.0 { T::from_f32(v) } else { T::from_f32(0.0) };
        });
    }
}

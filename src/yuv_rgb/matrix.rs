// const matrix stuff

use nalgebra::{Matrix1x3, Matrix3};

pub struct RowVector(f32, f32, f32);

impl RowVector {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self(x, y, z)
    }

    pub const fn scalar_mul(self, x: f32) -> Self {
        Self(self.0 * x, self.1 * x, self.2 * x)
    }

    pub fn as_nalgebra(&self) -> Matrix1x3<f32> {
        Matrix1x3::new(self.0, self.1, self.2)
    }
}

// pub struct ColVector(f32, f32, f32);

pub struct Matrix(RowVector, RowVector, RowVector);

impl Matrix {
    pub const fn new(r1: RowVector, r2: RowVector, r3: RowVector) -> Self {
        Self(r1, r2, r3)
    }

    pub fn as_nalgebra(&self) -> Matrix3<f32> {
        Matrix3::from_rows(&[
            self.0.as_nalgebra(),
            self.1.as_nalgebra(),
            self.2.as_nalgebra(),
        ])
    }
}

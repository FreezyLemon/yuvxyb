// const matrix stuff

use nalgebra::{Matrix1x3, Matrix3};

pub struct RowVector(f32, f32, f32);

impl RowVector {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self(x, y, z)
    }

    pub const fn from_array(arr: [f32; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }

    pub const fn x(&self) -> f32 { self.0 }
    pub const fn y(&self) -> f32 { self.1 }
    pub const fn z(&self) -> f32 { self.2 }

    pub const fn cross(&self, other: &Self) -> Self {
        let (sx, sy, sz) = (self.0, self.1, self.2);
        let (ox, oy, oz) = (other.0, other.1, other.2);

        Self::new(
            sy * oz - sz * oy,
            sz * ox - sx * oz,
            sx * oy - sy * ox,
        )
    }

    pub const fn dot(&self, other: &Self) -> f32 {
        self.0 * other.0 + self.1 * other.1 + self.2 * other.2
    }

    pub const fn scalar_mul(&self, x: f32) -> Self {
        Self(self.0 * x, self.1 * x, self.2 * x)
    }

    pub const fn scalar_div(&self, x: f32) -> Self {
        Self(self.0 / x, self.1 / x, self.2 / x)
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

    pub const fn scalar_div(&self, x: f32) -> Self {
        Self(
            self.0.scalar_div(x),
            self.1.scalar_div(x),
            self.2.scalar_div(x),
        )
    }

    pub const fn transpose(&self) -> Self {
        let Matrix(r1, r2, r3) = self;

        let RowVector(s11, s12, s13) = *r1;
        let RowVector(s21, s22, s23) = *r2;
        let RowVector(s31, s32, s33) = *r3;

        Self::new(
            RowVector::new(s11, s21, s31),
            RowVector::new(s12, s22, s32),
            RowVector::new(s13, s23, s33),
        )
    }

    pub fn as_nalgebra(&self) -> Matrix3<f32> {
        Matrix3::from_rows(&[
            self.0.as_nalgebra(),
            self.1.as_nalgebra(),
            self.2.as_nalgebra(),
        ])
    }
}

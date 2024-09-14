use std::ops::Mul;

#[derive(Clone, Copy)]
pub struct RowVector(f32, f32, f32);

impl RowVector {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self(x, y, z)
    }

    pub const fn from_array(arr: [f32; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }

    pub const fn x(self) -> f32 {
        self.0
    }
    pub const fn y(self) -> f32 {
        self.1
    }
    pub const fn z(self) -> f32 {
        self.2
    }

    pub const fn cross(self, other: Self) -> Self {
        let RowVector(sx, sy, sz) = self;
        let RowVector(ox, oy, oz) = other;

        Self::new(sy * oz - sz * oy, sz * ox - sx * oz, sx * oy - sy * ox)
    }

    pub const fn dot(self, other: Self) -> f32 {
        self.0 * other.0 + self.1 * other.1 + self.2 * other.2
    }

    pub const fn scalar_mul(self, x: f32) -> Self {
        Self(self.0 * x, self.1 * x, self.2 * x)
    }

    pub const fn scalar_div(self, x: f32) -> Self {
        Self(self.0 / x, self.1 / x, self.2 / x)
    }

    pub const fn component_mul(self, other: Self) -> Self {
        Self(self.0 * other.0, self.1 * other.1, self.2 * other.2)
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct ColVector(f32, f32, f32);

impl ColVector {
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self(r, g, b)
    }

    pub const fn from_array(arr: [f32; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }

    pub const fn r(self) -> f32 {
        self.0
    }
    pub const fn g(self) -> f32 {
        self.1
    }
    pub const fn b(self) -> f32 {
        self.2
    }

    pub const fn transpose(self) -> RowVector {
        RowVector::new(self.0, self.1, self.2)
    }
}

#[derive(Clone, Copy)]
pub struct Matrix(RowVector, RowVector, RowVector);

impl Matrix {
    pub const fn new(r1: RowVector, r2: RowVector, r3: RowVector) -> Self {
        Self(r1, r2, r3)
    }

    pub const fn r1(self) -> RowVector {
        self.0
    }
    pub const fn r2(self) -> RowVector {
        self.1
    }
    pub const fn r3(self) -> RowVector {
        self.2
    }

    pub const fn identity() -> Self {
        Self::new(
            RowVector::new(1.0, 0.0, 0.0),
            RowVector::new(0.0, 1.0, 0.0),
            RowVector::new(0.0, 0.0, 1.0),
        )
    }

    pub const fn scalar_div(self, x: f32) -> Self {
        Self(
            self.0.scalar_div(x),
            self.1.scalar_div(x),
            self.2.scalar_div(x),
        )
    }

    pub const fn transpose(self) -> Self {
        let Matrix(r1, r2, r3) = self;

        let RowVector(s11, s12, s13) = r1;
        let RowVector(s21, s22, s23) = r2;
        let RowVector(s31, s32, s33) = r3;

        Self::new(
            RowVector::new(s11, s21, s31),
            RowVector::new(s12, s22, s32),
            RowVector::new(s13, s23, s33),
        )
    }

    pub const fn invert(self) -> Self {
        // Cramer's rule
        let Matrix(r1, r2, r3) = self;

        let RowVector(s11, s12, s13) = r1;
        let RowVector(s21, s22, s23) = r2;
        let RowVector(s31, s32, s33) = r3;

        let minor_11 = s22 * s33 - s32 * s23;
        let minor_12 = s21 * s33 - s31 * s23;
        let minor_13 = s21 * s32 - s31 * s22;

        let minor_21 = s12 * s33 - s32 * s13;
        let minor_22 = s11 * s33 - s31 * s13;
        let minor_23 = s11 * s32 - s31 * s12;

        let minor_31 = s12 * s23 - s22 * s13;
        let minor_32 = s11 * s23 - s21 * s13;
        let minor_33 = s11 * s22 - s21 * s12;

        let determinant = s11 * minor_11 - s12 * minor_12 + s13 * minor_13;

        assert!(determinant != 0.0);

        Self::new(
            RowVector::new(minor_11, -minor_12, minor_13),
            RowVector::new(-minor_21, minor_22, -minor_23),
            RowVector::new(minor_31, -minor_32, minor_33),
        )
        .transpose()
        .scalar_div(determinant)
    }
}

impl Mul<ColVector> for Matrix {
    type Output = ColVector;

    fn mul(self, rhs: ColVector) -> Self::Output {
        let Matrix(r1, r2, r3) = self;

        ColVector::new(
            r1.0 * rhs.0 + r1.1 * rhs.1 + r1.2 * rhs.2,
            r2.0 * rhs.0 + r2.1 * rhs.1 + r2.2 * rhs.2,
            r3.0 * rhs.0 + r3.1 * rhs.1 + r3.2 * rhs.2,
        )
    }
}

impl Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Self::Output {
        let Matrix(r1, r2, r3) = self;
        let Matrix(o1, o2, o3) = rhs;

        Matrix::new(
            RowVector::new(
                r1.0 * o1.0 + r1.1 * o2.0 + r1.2 * o3.0,
                r1.0 * o1.1 + r1.1 * o2.1 + r1.2 * o3.1,
                r1.0 * o1.2 + r1.1 * o2.2 + r1.2 * o3.2,
            ),
            RowVector::new(
                r2.0 * o1.0 + r2.1 * o2.0 + r2.2 * o3.0,
                r2.0 * o1.1 + r2.1 * o2.1 + r2.2 * o3.1,
                r2.0 * o1.2 + r2.1 * o2.2 + r2.2 * o3.2,
            ),
            RowVector::new(
                r3.0 * o1.0 + r3.1 * o2.0 + r3.2 * o3.0,
                r3.0 * o1.1 + r3.1 * o2.1 + r3.2 * o3.1,
                r3.0 * o1.2 + r3.1 * o2.2 + r3.2 * o3.2,
            ),
        )
    }
}

impl Mul<[f32; 3]> for Matrix {
    type Output = [f32; 3];

    fn mul(self, rhs: [f32; 3]) -> Self::Output {
        let Matrix(r1, r2, r3) = self;

        [
            r1.0 * rhs[0] + r1.1 * rhs[1] + r1.2 * rhs[2],
            r2.0 * rhs[0] + r2.1 * rhs[1] + r2.2 * rhs[2],
            r3.0 * rhs[0] + r3.1 * rhs[1] + r3.2 * rhs[2],
        ]
    }
}

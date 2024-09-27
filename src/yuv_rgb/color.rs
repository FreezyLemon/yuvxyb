use av_data::pixel::{ColorPrimaries, MatrixCoefficients};

use super::matrix::{ColVector, Matrix, RowVector};

use super::{ycbcr_to_ypbpr, ypbpr_to_ycbcr};
use crate::{ConversionError, Pixel, Yuv, YuvConfig};

const fn m_to_idx(m: MatrixCoefficients) -> usize {
    match m {
        MatrixCoefficients::Identity => 0,
        MatrixCoefficients::BT709 => 1,
        MatrixCoefficients::Unspecified => 2,
        MatrixCoefficients::Reserved => 3,
        MatrixCoefficients::BT470M => 4,
        MatrixCoefficients::BT470BG => 5,
        MatrixCoefficients::ST170M => 6,
        MatrixCoefficients::ST240M => 7,
        MatrixCoefficients::YCgCo => 8,
        MatrixCoefficients::BT2020NonConstantLuminance => 9,
        MatrixCoefficients::BT2020ConstantLuminance => 10,
        MatrixCoefficients::ST2085 => 11,
        MatrixCoefficients::ChromaticityDerivedNonConstantLuminance => 12,
        MatrixCoefficients::ChromaticityDerivedConstantLuminance => 13,
        MatrixCoefficients::ICtCp => 14,
    }
}

const fn idx_to_m(idx: usize) -> Option<MatrixCoefficients> {
    match idx {
        0 => Some(MatrixCoefficients::Identity),
        1 => Some(MatrixCoefficients::BT709),
        2 => Some(MatrixCoefficients::Unspecified),
        3 => Some(MatrixCoefficients::Reserved),
        4 => Some(MatrixCoefficients::BT470M),
        5 => Some(MatrixCoefficients::BT470BG),
        6 => Some(MatrixCoefficients::ST170M),
        7 => Some(MatrixCoefficients::ST240M),
        8 => Some(MatrixCoefficients::YCgCo),
        9 => Some(MatrixCoefficients::BT2020NonConstantLuminance),
        10 => Some(MatrixCoefficients::BT2020ConstantLuminance),
        11 => Some(MatrixCoefficients::ST2085),
        12 => Some(MatrixCoefficients::ChromaticityDerivedNonConstantLuminance),
        13 => Some(MatrixCoefficients::ChromaticityDerivedConstantLuminance),
        14 => Some(MatrixCoefficients::ICtCp),
        _ => None,
    }
}

const fn c_to_idx(c: ColorPrimaries) -> usize {
    match c {
        ColorPrimaries::Reserved0 => 0,
        ColorPrimaries::BT709 => 1,
        ColorPrimaries::Unspecified => 2,
        ColorPrimaries::Reserved => 3,
        ColorPrimaries::BT470M => 4,
        ColorPrimaries::BT470BG => 5,
        ColorPrimaries::ST170M => 6,
        ColorPrimaries::ST240M => 7,
        ColorPrimaries::Film => 8,
        ColorPrimaries::BT2020 => 9,
        ColorPrimaries::ST428 => 10,
        ColorPrimaries::P3DCI => 11,
        ColorPrimaries::P3Display => 12,
        ColorPrimaries::Tech3213 => 13,
    }
}

const fn idx_to_c(idx: usize) -> Option<ColorPrimaries> {
    match idx {
        0 => Some(ColorPrimaries::Reserved0),
        1 => Some(ColorPrimaries::BT709),
        2 => Some(ColorPrimaries::Unspecified),
        3 => Some(ColorPrimaries::Reserved),
        4 => Some(ColorPrimaries::BT470M),
        5 => Some(ColorPrimaries::BT470BG),
        6 => Some(ColorPrimaries::ST170M),
        7 => Some(ColorPrimaries::ST240M),
        8 => Some(ColorPrimaries::Film),
        9 => Some(ColorPrimaries::BT2020),
        10 => Some(ColorPrimaries::ST428),
        11 => Some(ColorPrimaries::P3DCI),
        12 => Some(ColorPrimaries::P3Display),
        13 => Some(ColorPrimaries::Tech3213),
        _ => None,
    }
}

const NUM_MATRIX_COEFFICIENTS: usize = {
    let mut idx = 0;
    while let Some(_) = idx_to_m(idx) {
        idx += 1;
    }

    idx
};
const NUM_COLOR_PRIMARIES: usize = {
    let mut idx = 0;
    while let Some(_) = idx_to_c(idx) {
        idx += 1;
    }

    idx
};

static RGB_TO_YUV_MATS: [[Result<Matrix, ConversionError>; NUM_MATRIX_COEFFICIENTS];
    NUM_COLOR_PRIMARIES] = {
    let mut result = [[Err(ConversionError::UnsupportedColorPrimaries); NUM_MATRIX_COEFFICIENTS];
        NUM_COLOR_PRIMARIES];

    let mut c_idx = 0;
    while c_idx < NUM_COLOR_PRIMARIES {
        let Some(c) = idx_to_c(c_idx) else {
            panic!("couldn't convert index to ColorPrimaries");
        };

        let mut m_idx = 0;
        while m_idx < NUM_MATRIX_COEFFICIENTS {
            let Some(m) = idx_to_m(m_idx) else {
                panic!("couldn't convert index to MatrixCoefficients");
            };
            result[c_idx][m_idx] = get_rgb_to_yuv_matrix(m, c);

            m_idx += 1;
        }

        c_idx += 1;
    }

    result
};

static PRIMARY_TRANSFORM_MATS: [[Result<Matrix, ConversionError>; NUM_COLOR_PRIMARIES];
    NUM_COLOR_PRIMARIES] = {
    let mut result = [[Err(ConversionError::UnsupportedColorPrimaries); NUM_COLOR_PRIMARIES];
        NUM_COLOR_PRIMARIES];

    let mut in_c_idx = 0;
    while in_c_idx < NUM_COLOR_PRIMARIES {
        let Some(in_c) = idx_to_c(in_c_idx) else {
            panic!("couldn't convert index to ColorPrimaries");
        };

        let mut out_c_idx = 0;
        while out_c_idx < NUM_COLOR_PRIMARIES {
            let Some(out_c) = idx_to_c(out_c_idx) else {
                panic!("couldn't convert index to ColorPrimaries");
            };

            let x_to_r = match gamut_xyz_to_rgb_matrix(out_c) {
                Ok(m) => m,
                Err(e) => {
                    result[in_c_idx][out_c_idx] = Err(e);
                    out_c_idx += 1;
                    continue;
                }
            };

            let r_to_x = match gamut_rgb_to_xyz_matrix(in_c) {
                Ok(m) => m,
                Err(e) => {
                    result[in_c_idx][out_c_idx] = Err(e);
                    out_c_idx += 1;
                    continue;
                }
            };

            let white_point = white_point_adaptation_matrix(in_c, out_c);

            result[in_c_idx][out_c_idx] = Ok(x_to_r.mul_mat(white_point).mul_mat(r_to_x));
            out_c_idx += 1;
        }

        in_c_idx += 1;
    }

    result
};

const fn get_rgb_to_yuv_matrix(
    matrix: MatrixCoefficients,
    primaries: ColorPrimaries,
) -> Result<Matrix, ConversionError> {
    match matrix {
        MatrixCoefficients::Identity
        | MatrixCoefficients::BT2020ConstantLuminance
        | MatrixCoefficients::ChromaticityDerivedConstantLuminance
        | MatrixCoefficients::ST2085
        | MatrixCoefficients::ICtCp => ncl_rgb_to_yuv_matrix_from_primaries(primaries),
        MatrixCoefficients::BT709
        | MatrixCoefficients::BT470M
        | MatrixCoefficients::BT470BG
        | MatrixCoefficients::ST170M
        | MatrixCoefficients::ST240M
        | MatrixCoefficients::YCgCo
        | MatrixCoefficients::ChromaticityDerivedNonConstantLuminance
        | MatrixCoefficients::BT2020NonConstantLuminance => ncl_rgb_to_yuv_matrix(matrix),
        MatrixCoefficients::Reserved => Err(ConversionError::UnsupportedMatrixCoefficients),
        MatrixCoefficients::Unspecified => Err(ConversionError::UnspecifiedMatrixCoefficients),
    }
}

const fn ncl_rgb_to_yuv_matrix_from_primaries(
    primaries: ColorPrimaries,
) -> Result<Matrix, ConversionError> {
    match primaries {
        ColorPrimaries::BT709 => ncl_rgb_to_yuv_matrix(MatrixCoefficients::BT709),
        ColorPrimaries::BT2020 => {
            ncl_rgb_to_yuv_matrix(MatrixCoefficients::BT2020NonConstantLuminance)
        }
        p => match get_yuv_constants_from_primaries(p) {
            Ok((kr, kb)) => Ok(ncl_rgb_to_yuv_matrix_from_kr_kb(kr, kb)),
            Err(e) => Err(e),
        },
    }
}

const fn ncl_rgb_to_yuv_matrix(matrix: MatrixCoefficients) -> Result<Matrix, ConversionError> {
    Ok(match matrix {
        MatrixCoefficients::YCgCo => Matrix::new(
            RowVector::new(0.25, 0.5, 0.25),
            RowVector::new(-0.25, 0.5, -0.25),
            RowVector::new(0.5, 0.0, -0.5),
        ),
        MatrixCoefficients::ST2085 => Matrix::new(
            RowVector::new(1688.0, 2146.0, 262.0),
            RowVector::new(683.0, 2951.0, 462.0),
            RowVector::new(99.0, 309.0, 3688.0),
        )
        .scalar_div(4096.0),
        m => match get_yuv_constants(m) {
            Ok((kr, kb)) => ncl_rgb_to_yuv_matrix_from_kr_kb(kr, kb),
            Err(e) => return Err(e),
        },
    })
}

const fn get_yuv_constants_from_primaries(
    primaries: ColorPrimaries,
) -> Result<(f32, f32), ConversionError> {
    // ITU-T H.265 Annex E, Eq (E-22) to (E-27).
    let [primaries_r, primaries_g, primaries_b] = match get_primaries_xy(primaries) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };

    let r_xyz = RowVector::from_array(xy_to_xyz(primaries_r));
    let g_xyz = RowVector::from_array(xy_to_xyz(primaries_g));
    let b_xyz = RowVector::from_array(xy_to_xyz(primaries_b));
    let white_xyz = RowVector::from_array(get_white_point(primaries));

    let x_rgb = RowVector::new(r_xyz.x(), g_xyz.x(), b_xyz.x());
    let y_rgb = RowVector::new(r_xyz.y(), g_xyz.y(), b_xyz.y());
    let z_rgb = RowVector::new(r_xyz.z(), g_xyz.z(), b_xyz.z());

    let denom = x_rgb.dot(y_rgb.cross(z_rgb));
    let kr = white_xyz.dot(g_xyz.cross(b_xyz)) / denom;
    let kb = white_xyz.dot(r_xyz.cross(g_xyz)) / denom;

    Ok((kr, kb))
}

const fn get_yuv_constants(matrix: MatrixCoefficients) -> Result<(f32, f32), ConversionError> {
    Ok(match matrix {
        MatrixCoefficients::Identity => (0.0, 0.0),
        MatrixCoefficients::BT470M => (0.3, 0.11),
        MatrixCoefficients::ST240M => (0.212, 0.087),
        MatrixCoefficients::BT470BG | MatrixCoefficients::ST170M => (0.299, 0.114),
        MatrixCoefficients::BT709 => (0.2126, 0.0722),
        MatrixCoefficients::BT2020NonConstantLuminance
        | MatrixCoefficients::BT2020ConstantLuminance => (0.2627, 0.0593),
        // Unusable
        MatrixCoefficients::Reserved
        | MatrixCoefficients::YCgCo
        | MatrixCoefficients::ST2085
        | MatrixCoefficients::ChromaticityDerivedNonConstantLuminance
        | MatrixCoefficients::ChromaticityDerivedConstantLuminance
        | MatrixCoefficients::ICtCp => return Err(ConversionError::UnsupportedMatrixCoefficients),
        MatrixCoefficients::Unspecified => {
            return Err(ConversionError::UnspecifiedMatrixCoefficients)
        }
    })
}

const fn ncl_rgb_to_yuv_matrix_from_kr_kb(kr: f32, kb: f32) -> Matrix {
    let kg = 1.0 - kr - kb;
    let uscale = -2.0 * kb + 2.0;
    let vscale = -2.0 * kr + 2.0;

    Matrix::new(
        RowVector::new(kr, kg, kb),
        RowVector::new(-kr, -kg, 1.0 - kb).scalar_div(uscale),
        RowVector::new(1.0 - kr, -kg, -kb).scalar_div(vscale),
    )
}

const fn get_primaries_xy(primaries: ColorPrimaries) -> Result<[[f32; 2]; 3], ConversionError> {
    Ok(match primaries {
        ColorPrimaries::BT470M => [[0.670, 0.330], [0.210, 0.710], [0.140, 0.080]],
        ColorPrimaries::BT470BG => [[0.640, 0.330], [0.290, 0.600], [0.150, 0.060]],
        ColorPrimaries::ST170M | ColorPrimaries::ST240M => {
            [[0.630, 0.340], [0.310, 0.595], [0.155, 0.070]]
        }
        ColorPrimaries::BT709 => [[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]],
        ColorPrimaries::Film => [[0.681, 0.319], [0.243, 0.692], [0.145, 0.049]],
        ColorPrimaries::BT2020 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        ColorPrimaries::P3DCI | ColorPrimaries::P3Display => {
            [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]]
        }
        ColorPrimaries::Tech3213 => [[0.630, 0.340], [0.295, 0.605], [0.155, 0.077]],
        ColorPrimaries::Reserved0 | ColorPrimaries::Reserved | ColorPrimaries::ST428 => {
            return Err(ConversionError::UnsupportedColorPrimaries)
        }
        // SAFETY: We guess any unspecified data when beginning conversion
        ColorPrimaries::Unspecified => return Err(ConversionError::UnspecifiedColorPrimaries),
    })
}

const fn get_white_point(primaries: ColorPrimaries) -> [f32; 3] {
    // White points in XY.
    const ILLUMINANT_C: [f32; 2] = [0.31, 0.316];
    const ILLUMINANT_DCI: [f32; 2] = [0.314, 0.351];
    const ILLUMINANT_D65: [f32; 2] = [0.3127, 0.3290];
    const ILLUMINANT_E: [f32; 2] = [1.0 / 3.0, 1.0 / 3.0];

    match primaries {
        ColorPrimaries::BT470M | ColorPrimaries::Film => xy_to_xyz(ILLUMINANT_C),
        ColorPrimaries::ST428 => xy_to_xyz(ILLUMINANT_E),
        ColorPrimaries::P3DCI => xy_to_xyz(ILLUMINANT_DCI),
        _ => xy_to_xyz(ILLUMINANT_D65),
    }
}

const fn xy_to_xyz(xy: [f32; 2]) -> [f32; 3] {
    let [x, y] = xy;
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// Converts 8..=16-bit YUV data to 32-bit floating point gamma-corrected RGB
/// in a range of 0.0..=1.0;
pub fn yuv_to_rgb<T: Pixel>(input: &Yuv<T>) -> Result<Vec<[f32; 3]>, ConversionError> {
    let m = m_to_idx(input.config().matrix_coefficients);
    let c = c_to_idx(input.config().color_primaries);
    let transform = RGB_TO_YUV_MATS[c][m]?.invert();
    let mut data = ycbcr_to_ypbpr(input);

    for pix in &mut data {
        *pix = transform.mul_arr(*pix);
    }

    Ok(data)
}

/// Converts 32-bit floating point gamma-corrected RGB in a range of 0.0..=1.0
/// to 8..=16-bit YUV.
///
/// # Errors
/// - If the `YuvConfig` would produce an invalid image
pub fn rgb_to_yuv<T: Pixel>(
    input: &[[f32; 3]],
    width: usize,
    height: usize,
    config: YuvConfig,
) -> Result<Yuv<T>, ConversionError> {
    let m = m_to_idx(config.matrix_coefficients);
    let c = c_to_idx(config.color_primaries);
    let transform = RGB_TO_YUV_MATS[c][m]?;
    let yuv: Vec<_> = input.iter().map(|pix| transform.mul_arr(*pix)).collect();
    Ok(ypbpr_to_ycbcr(&yuv, width, height, config))
}

pub fn transform_primaries(
    mut input: Vec<[f32; 3]>,
    in_primaries: ColorPrimaries,
    out_primaries: ColorPrimaries,
) -> Result<Vec<[f32; 3]>, ConversionError> {
    if in_primaries == out_primaries {
        return Ok(input);
    }

    let in_c = c_to_idx(in_primaries);
    let out_c = c_to_idx(out_primaries);
    let transform = PRIMARY_TRANSFORM_MATS[in_c][out_c]?;

    for pix in &mut input {
        *pix = transform.mul_arr(*pix);
    }

    Ok(input)
}

const fn gamut_rgb_to_xyz_matrix(primaries: ColorPrimaries) -> Result<Matrix, ConversionError> {
    if matches!(primaries, ColorPrimaries::ST428) {
        return Ok(Matrix::identity());
    }

    let xyz_matrix = match get_primaries_xyz(primaries) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let white_xyz = ColVector::from_array(get_white_point(primaries));

    let s = xyz_matrix.invert().mul_vec(white_xyz).transpose();
    Ok(Matrix::new(
        xyz_matrix.r1().component_mul(s),
        xyz_matrix.r2().component_mul(s),
        xyz_matrix.r3().component_mul(s),
    ))
}

const fn gamut_xyz_to_rgb_matrix(primaries: ColorPrimaries) -> Result<Matrix, ConversionError> {
    if matches!(primaries, ColorPrimaries::ST428) {
        return Ok(Matrix::identity());
    }

    match gamut_rgb_to_xyz_matrix(primaries) {
        Ok(m) => Ok(m.invert()),
        Err(e) => Err(e),
    }
}

const fn get_primaries_xyz(primaries: ColorPrimaries) -> Result<Matrix, ConversionError> {
    // Columns: R G B
    // Rows: X Y Z
    let [primaries_r, primaries_g, primaries_b] = match get_primaries_xy(primaries) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };

    let m = Matrix::new(
        RowVector::from_array(xy_to_xyz(primaries_r)),
        RowVector::from_array(xy_to_xyz(primaries_g)),
        RowVector::from_array(xy_to_xyz(primaries_b)),
    );

    Ok(m.transpose())
}

const fn white_point_adaptation_matrix(
    in_primaries: ColorPrimaries,
    out_primaries: ColorPrimaries,
) -> Matrix {
    let bradford = Matrix::new(
        RowVector::new(0.8951f32, 0.2664f32, -0.1614f32),
        RowVector::new(-0.7502f32, 1.7135f32, 0.0367f32),
        RowVector::new(0.0389f32, -0.0685f32, 1.0296f32),
    );

    let white_in = ColVector::from_array(get_white_point(in_primaries));
    let white_out = ColVector::from_array(get_white_point(out_primaries));

    if white_in.eq(white_out) {
        return Matrix::identity();
    }

    let rgb_in = bradford.mul_vec(white_in);
    let rgb_out = bradford.mul_vec(white_out);

    let m = Matrix::new(
        RowVector::new(rgb_out.r() / rgb_in.r(), 0.0, 0.0),
        RowVector::new(0.0, rgb_out.g() / rgb_in.g(), 0.0),
        RowVector::new(0.0, 0.0, rgb_out.b() / rgb_in.b()),
    );

    bradford.invert().mul_mat(m).mul_mat(bradford)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn size_of() {
        use std::mem::size_of;

        let s = size_of::<Result<Matrix, ConversionError>>();
        println!("Result<Matrix, ConversionError>: {s}");

        let s = size_of::<Matrix>();
        println!("Matrix: {s}");

        let s = size_of::<ConversionError>();
        println!("ConversionError: {s}");
    }
}

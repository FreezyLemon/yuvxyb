use anyhow::{bail, Result};
use av_data::pixel::{ColorPrimaries, MatrixCoefficients};
use debug_unreachable::debug_unreachable;
use nalgebra::{Matrix1x3, Matrix3, Matrix3x1};

use super::{from_yuv444f32, to_yuv444f32};
use crate::{Yuv, YuvConfig, YuvPixel};

pub fn get_yuv_to_rgb_matrix(config: YuvConfig) -> Result<Matrix3<f32>> {
    Ok(get_rgb_to_yuv_matrix(config)?
        .try_inverse()
        .expect("Matrix can be inverted"))
}

pub fn get_rgb_to_yuv_matrix(config: YuvConfig) -> Result<Matrix3<f32>> {
    match config.matrix_coefficients {
        MatrixCoefficients::Identity
        | MatrixCoefficients::BT2020ConstantLuminance
        | MatrixCoefficients::ChromaticityDerivedConstantLuminance
        | MatrixCoefficients::ST2085
        | MatrixCoefficients::ICtCp => ncl_rgb_to_yuv_matrix_from_primaries(config.color_primaries),
        MatrixCoefficients::BT709
        | MatrixCoefficients::BT470M
        | MatrixCoefficients::BT470BG
        | MatrixCoefficients::ST170M
        | MatrixCoefficients::ST240M
        | MatrixCoefficients::YCgCo
        | MatrixCoefficients::ChromaticityDerivedNonConstantLuminance
        | MatrixCoefficients::BT2020NonConstantLuminance => {
            ncl_rgb_to_yuv_matrix(config.matrix_coefficients)
        }
        // Unusable
        MatrixCoefficients::Reserved => {
            bail!("Cannot convert YUV<->RGB using this transfer function")
        }
        // SAFETY: We guess any unspecified data when beginning conversion
        MatrixCoefficients::Unspecified => unsafe { debug_unreachable!() },
    }
}

pub fn ncl_rgb_to_yuv_matrix_from_primaries(primaries: ColorPrimaries) -> Result<Matrix3<f32>> {
    match primaries {
        ColorPrimaries::BT709 => ncl_rgb_to_yuv_matrix(MatrixCoefficients::BT709),
        ColorPrimaries::BT2020 => {
            ncl_rgb_to_yuv_matrix(MatrixCoefficients::BT2020NonConstantLuminance)
        }
        _ => {
            let (kr, kb) = get_yuv_constants_from_primaries(primaries)?;
            Ok(ncl_rgb_to_yuv_matrix_from_kr_kb(kr, kb))
        }
    }
}

pub fn ncl_rgb_to_yuv_matrix(matrix: MatrixCoefficients) -> Result<Matrix3<f32>> {
    Ok(match matrix {
        MatrixCoefficients::YCgCo => {
            Matrix3::from_row_slice(&[0.25, 0.5, 0.25, -0.25, 0.5, -0.25, 0.5, 0.0, -0.5])
        }
        MatrixCoefficients::ST2085 => Matrix3::from_row_slice(&[
            1688.0 / 4096.0,
            2146.0 / 4096.0,
            262.0 / 4096.0,
            683.0 / 4096.0,
            2951.0 / 4096.0,
            462.0 / 4096.0,
            99.0 / 4096.0,
            309.0 / 4096.0,
            3688.0 / 4096.0,
        ]),
        _ => {
            let (kr, kb) = get_yuv_constants(matrix)?;
            ncl_rgb_to_yuv_matrix_from_kr_kb(kr, kb)
        }
    })
}

pub fn get_yuv_constants_from_primaries(primaries: ColorPrimaries) -> Result<(f32, f32)> {
    // ITU-T H.265 Annex E, Eq (E-22) to (E-27).
    let primaries_xy = get_primaries_xy(primaries)?;

    let r_xyz = Matrix1x3::from_row_slice(&xy_to_xyz(primaries_xy[0][0], primaries_xy[0][1]));
    let g_xyz = Matrix1x3::from_row_slice(&xy_to_xyz(primaries_xy[1][0], primaries_xy[1][1]));
    let b_xyz = Matrix1x3::from_row_slice(&xy_to_xyz(primaries_xy[2][0], primaries_xy[2][1]));
    let white_xyz = Matrix1x3::from_row_slice(&get_white_point(primaries));

    let x_rgb = Matrix1x3::from_row_slice(&[r_xyz[0], g_xyz[0], b_xyz[0]]);
    let y_rgb = Matrix1x3::from_row_slice(&[r_xyz[1], g_xyz[1], b_xyz[1]]);
    let z_rgb = Matrix1x3::from_row_slice(&[r_xyz[2], g_xyz[2], b_xyz[2]]);

    let denom = x_rgb.dot(&y_rgb.cross(&z_rgb));
    let kr = white_xyz.dot(&g_xyz.cross(&b_xyz)) / denom;
    let kb = white_xyz.dot(&r_xyz.cross(&g_xyz)) / denom;

    Ok((kr, kb))
}

pub fn get_yuv_constants(matrix: MatrixCoefficients) -> Result<(f32, f32)> {
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
        | MatrixCoefficients::ICtCp => {
            bail!("Cannot convert YUV<->RGB using these matrix coefficients")
        }
        // SAFETY: We guess any unspecified data when beginning conversion
        MatrixCoefficients::Unspecified => unsafe { debug_unreachable!() },
    })
}

pub fn ncl_rgb_to_yuv_matrix_from_kr_kb(kr: f32, kb: f32) -> Matrix3<f32> {
    let mut ret = [0.0; 9];
    let kg = 1.0 - kr - kb;
    let uscale = 1.0 / (2.0 - 2.0 * kb);
    let vscale = 1.0 / (2.0 - 2.0 * kr);

    ret[0] = kr;
    ret[1] = kg;
    ret[2] = kb;

    ret[3] = -kr * uscale;
    ret[4] = -kg * uscale;
    ret[5] = (1.0 - kb) * uscale;

    ret[6] = (1.0 - kr) * vscale;
    ret[7] = -kg * vscale;
    ret[8] = -kb * vscale;

    Matrix3::from_row_slice(&ret)
}

pub fn get_primaries_xy(primaries: ColorPrimaries) -> Result<[[f32; 2]; 3]> {
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
            bail!("Cannot convert YUV<->RGB using these primaries")
        }
        // SAFETY: We guess any unspecified data when beginning conversion
        ColorPrimaries::Unspecified => unsafe { debug_unreachable!() },
    })
}

pub fn get_white_point(primaries: ColorPrimaries) -> [f32; 3] {
    // White points in XY.
    const ILLUMINANT_C: [f32; 2] = [0.31, 0.316];
    const ILLUMINANT_DCI: [f32; 2] = [0.314, 0.351];
    const ILLUMINANT_D65: [f32; 2] = [0.3127, 0.3290];
    const ILLUMINANT_E: [f32; 2] = [1.0 / 3.0, 1.0 / 3.0];

    match primaries {
        ColorPrimaries::BT470M | ColorPrimaries::Film => {
            xy_to_xyz(ILLUMINANT_C[0], ILLUMINANT_C[1])
        }
        ColorPrimaries::ST428 => xy_to_xyz(ILLUMINANT_E[0], ILLUMINANT_E[1]),
        ColorPrimaries::P3DCI => xy_to_xyz(ILLUMINANT_DCI[0], ILLUMINANT_DCI[1]),
        _ => xy_to_xyz(ILLUMINANT_D65[0], ILLUMINANT_D65[1]),
    }
}

fn xy_to_xyz(x: f32, y: f32) -> [f32; 3] {
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// Converts 8..=16-bit YUV data to 32-bit floating point gamma-corrected RGB
/// in a range of 0.0..=1.0;
pub fn yuv_to_rgb<T: YuvPixel>(input: &Yuv<T>) -> Result<Vec<[f32; 3]>> {
    let transform = get_yuv_to_rgb_matrix(input.config())?;
    let data = to_yuv444f32(input);
    Ok(data
        .into_iter()
        .map(|pix| {
            let pix = Matrix3x1::from_column_slice(&pix);
            let res = transform * pix;
            [res[0], res[1], res[2]]
        })
        .collect::<Vec<_>>())
}

/// Converts 32-bit floating point gamma-corrected RGB in a range of 0.0..=1.0
/// to 8..=16-bit YUV.
///
/// # Errors
/// - If the `YuvConfig` would produce an invalid image
pub fn rgb_to_yuv<T: YuvPixel>(
    input: &[[f32; 3]],
    width: u32,
    height: u32,
    config: YuvConfig,
) -> Result<Yuv<T>> {
    let transform = get_rgb_to_yuv_matrix(config)?;
    let yuv = input
        .iter()
        .map(|pix| {
            let pix = Matrix3x1::from_column_slice(pix);
            let res = transform * pix;
            [res[0], res[1], res[2]]
        })
        .collect::<Vec<_>>();
    Ok(from_yuv444f32(
        &yuv,
        width as usize,
        height as usize,
        config,
    ))
}

#[cfg(test)]
mod tests {
    use av_data::pixel::{ColorPrimaries, MatrixCoefficients, TransferCharacteristic};

    use super::*;
    use crate::Yuv;

    #[test]
    fn bt601_full_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> =
            vec![(168, 152, 92), (71, 57, 230), (122, 122, 79), (133, 96, 39)];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: true,
            matrix_coefficients: MatrixCoefficients::ST170M,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::ST170M,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = yuv_to_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = rgb_to_yuv(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt601_limited_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> = vec![
            (160, 149, 97),
            (77, 67, 215),
            (121, 123, 86),
            (130, 101, 52),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: false,
            matrix_coefficients: MatrixCoefficients::ST170M,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::ST170M,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = yuv_to_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = rgb_to_yuv(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt601_full_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (674, 609, 367),
            (286, 226, 920),
            (491, 487, 315),
            (532, 385, 155),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 10,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: true,
            matrix_coefficients: MatrixCoefficients::ST170M,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::ST170M,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = yuv_to_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = rgb_to_yuv(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt601_limited_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (641, 595, 388),
            (309, 267, 861),
            (484, 490, 344),
            (520, 403, 206),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 10,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: false,
            matrix_coefficients: MatrixCoefficients::ST170M,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::ST170M,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = yuv_to_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = rgb_to_yuv(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt709_full_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> = vec![
            (174, 131, 91),
            (73, 109, 232),
            (114, 162, 109),
            (112, 205, 109),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: true,
            matrix_coefficients: MatrixCoefficients::BT709,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::BT709,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = yuv_to_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = rgb_to_yuv(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt709_limited_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> = vec![
            (165, 131, 95),
            (79, 112, 220),
            (114, 158, 111),
            (112, 195, 111),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: false,
            matrix_coefficients: MatrixCoefficients::BT709,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::BT709,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = yuv_to_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = rgb_to_yuv(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt709_full_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (698, 525, 362),
            (295, 438, 931),
            (459, 650, 435),
            (448, 820, 437),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 10,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: true,
            matrix_coefficients: MatrixCoefficients::BT709,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::BT709,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = yuv_to_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = rgb_to_yuv(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt709_limited_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (662, 523, 380),
            (316, 447, 879),
            (457, 632, 444),
            (447, 782, 446),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 10,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: false,
            matrix_coefficients: MatrixCoefficients::BT709,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::BT709,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = yuv_to_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = rgb_to_yuv(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt2020_full_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> = vec![
            (170, 133, 90),
            (84, 104, 233),
            (112, 163, 109),
            (108, 205, 110),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: true,
            matrix_coefficients: MatrixCoefficients::BT2020NonConstantLuminance,
            transfer_characteristics: TransferCharacteristic::BT2020Ten,
            color_primaries: ColorPrimaries::BT2020,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = yuv_to_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = rgb_to_yuv(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt2020_limited_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> = vec![
            (162, 132, 95),
            (88, 107, 220),
            (112, 159, 111),
            (109, 196, 112),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: false,
            matrix_coefficients: MatrixCoefficients::BT2020NonConstantLuminance,
            transfer_characteristics: TransferCharacteristic::BT2020Ten,
            color_primaries: ColorPrimaries::BT2020,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = yuv_to_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = rgb_to_yuv(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt2020_full_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (684, 533, 361),
            (336, 416, 931),
            (449, 653, 436),
            (435, 822, 440),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 10,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: true,
            matrix_coefficients: MatrixCoefficients::BT2020NonConstantLuminance,
            transfer_characteristics: TransferCharacteristic::BT2020Ten,
            color_primaries: ColorPrimaries::BT2020,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = yuv_to_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = rgb_to_yuv(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt2020_limited_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (649, 530, 380),
            (352, 428, 879),
            (449, 635, 445),
            (437, 784, 449),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 10,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: false,
            matrix_coefficients: MatrixCoefficients::BT2020NonConstantLuminance,
            transfer_characteristics: TransferCharacteristic::BT2020Ten,
            color_primaries: ColorPrimaries::BT2020,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = yuv_to_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = rgb_to_yuv(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }
}
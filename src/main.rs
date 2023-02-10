use std::{cmp::Ordering, env::args};

use image::{self, GenericImage, GenericImageView};
use nalgebra::Vector3;

type Vec3 = Vector3<f32>;

const COLOR_SCHEME: [Vec3; 26] = [
    Vec3::new(244_f32, 219_f32, 214_f32),
    Vec3::new(240_f32, 198_f32, 198_f32),
    Vec3::new(245_f32, 189_f32, 230_f32),
    Vec3::new(198_f32, 160_f32, 246_f32),
    Vec3::new(237_f32, 135_f32, 150_f32),
    Vec3::new(238_f32, 153_f32, 160_f32),
    Vec3::new(245_f32, 169_f32, 127_f32),
    Vec3::new(238_f32, 212_f32, 159_f32),
    Vec3::new(166_f32, 218_f32, 149_f32),
    Vec3::new(139_f32, 213_f32, 202_f32),
    Vec3::new(145_f32, 215_f32, 227_f32),
    Vec3::new(125_f32, 196_f32, 228_f32),
    Vec3::new(138_f32, 173_f32, 244_f32),
    Vec3::new(183_f32, 189_f32, 248_f32),
    Vec3::new(202_f32, 211_f32, 245_f32),
    Vec3::new(184_f32, 192_f32, 224_f32),
    Vec3::new(165_f32, 173_f32, 203_f32),
    Vec3::new(147_f32, 154_f32, 183_f32),
    Vec3::new(128_f32, 135_f32, 162_f32),
    Vec3::new(110_f32, 115_f32, 141_f32),
    Vec3::new(91_f32, 96_f32, 120_f32),
    Vec3::new(73_f32, 77_f32, 100_f32),
    Vec3::new(54_f32, 58_f32, 79_f32),
    Vec3::new(36_f32, 39_f32, 58_f32),
    Vec3::new(30_f32, 32_f32, 48_f32),
    Vec3::new(24_f32, 25_f32, 38_f32),
];

// we also need to normalize vectors
fn find_simular(original: Vec3) -> Vec3 {
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let sim_fn = |a: &Vec3, b: &Vec3| {
        let a = nalgebra::Point3::from(*a);
        let b = nalgebra::Point3::from(*b);
        sigmoid(nalgebra::distance(&a, &b))
    };
    let norm = |v: Vec3| {
        let (x, y, z) = (v.x, v.y, v.z);
        let (x, y, z) = (
            x / 255.0 * 2.0 - 1.0,
            y / 255.0 * 2.0 - 1.0,
            z / 255.0 * 2.0 - 1.0,
        );
        Vec3::new(x, y, z)
    };

    let idx = COLOR_SCHEME
        .iter()
        .map(|&color| sim_fn(&norm(original), &norm(color)))
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            if a > b {
                Ordering::Greater
            } else if a < b {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
        .unwrap()
        .0;

    COLOR_SCHEME[idx]
}

fn find_closest_to_color(img: &mut image::DynamicImage) {
    let (width, height) = img.dimensions();

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let pixel = Vec3::new(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
            let new_pixel = find_simular(pixel);
            img.put_pixel(
                x,
                y,
                image::Rgba([new_pixel.x as u8, new_pixel.y as u8, new_pixel.z as u8, 255]),
            );
        }
    }
}
// Floyd–Steinberg dithering
fn floyd_steinberg_dithering(img: &mut image::DynamicImage) {
    let (width, height) = img.dimensions();

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let pixel = Vec3::new(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
            let new_pixel = find_simular(pixel);
            img.put_pixel(
                x,
                y,
                image::Rgba([new_pixel.x as u8, new_pixel.y as u8, new_pixel.z as u8, 255]),
            );

            let error = pixel - new_pixel;
            let error = Vec3::new(error.x / 16.0, error.y / 16.0, error.z / 16.0);

            if x + 1 < width {
                let pixel = img.get_pixel(x + 1, y);
                let pixel = Vec3::new(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
                let new_pixel = pixel + error * 7.0;
                img.put_pixel(
                    x + 1,
                    y,
                    image::Rgba([new_pixel.x as u8, new_pixel.y as u8, new_pixel.z as u8, 255]),
                );
            }

            if y + 1 < height {
                if x > 0 {
                    let pixel = img.get_pixel(x - 1, y + 1);
                    let pixel = Vec3::new(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
                    let new_pixel = pixel + error * 3.0;
                    img.put_pixel(
                        x - 1,
                        y + 1,
                        image::Rgba([new_pixel.x as u8, new_pixel.y as u8, new_pixel.z as u8, 255]),
                    );
                }

                let pixel = img.get_pixel(x, y + 1);
                let pixel = Vec3::new(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
                let new_pixel = pixel + error * 5.0;
                img.put_pixel(
                    x,
                    y + 1,
                    image::Rgba([new_pixel.x as u8, new_pixel.y as u8, new_pixel.z as u8, 255]),
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Algorithm {
    FloydSteinberg,
    ClosestToColor,
}
impl Algorithm {
    fn from_str(s: &str) -> Self {
        match s {
            "floyd-steinberg"  => Self::FloydSteinberg,
            "closest-to-color" => Self::ClosestToColor,
            _                  => panic!("Unknown algorithm"),
        }
    }
}

#[derive(Debug)]
struct App {
    output: String,
    algorithm: Algorithm,
    image: image::DynamicImage,
}

impl App {
    fn parse() -> Self {
        let args: Vec<_> = args().collect();

        let output = if let Some(pos) = args.iter().position(|s| s == "-o") {
            args[pos + 1].to_string()
        } else {
            format!("dih_{}", args.get(1).expect("provide input file!") )
        };

        let algorithm = if let Some(pos) = args.iter().position(|s| s == "-a") {
            &args[pos + 1]
        } else {
            "floyd-steinberg"
        };

        Self {
            image: image::open(args.get(1).expect("Provide input file")).expect("Could not open image"),
            output: output.to_string(),
            algorithm: Algorithm::from_str(&algorithm),
        }
    }

    fn save_image(&self) {
        self.image.save(&self.output).expect("Could not save image");
    }

    fn run_algorithm(mut self) -> Self {
        match self.algorithm {
            Algorithm::FloydSteinberg => floyd_steinberg_dithering(&mut self.image),
            Algorithm::ClosestToColor => find_closest_to_color    (&mut self.image),
        }
        
        self
    }
}

fn main() {
    App::parse()
        .run_algorithm()
        .save_image();
}

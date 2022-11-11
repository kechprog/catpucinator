use std::cmp::Ordering;

use image::{self, GenericImageView, GenericImage};
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
    Vec3::new(91_f32,  96_f32,  120_f32),
    Vec3::new(73_f32,  77_f32,  100_f32),
    Vec3::new(54_f32,  58_f32,  79_f32),
    Vec3::new(36_f32,  39_f32,  58_f32),
    Vec3::new(30_f32,  32_f32,  48_f32),
    Vec3::new(24_f32,  25_f32,  38_f32),
];

/*
 * Normalization breaks everything
 * and rough edges
 * too less colors
*/

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
    // x.tanh()
}

fn sim_fn(a: &Vec3, b: &Vec3) -> f32 {
    let a = nalgebra::Point3::from(*a);
    let b = nalgebra::Point3::from(*b);
    sigmoid(nalgebra::distance(&a, &b))
    // cos sim
    // sigmoid(a.dot(b) / (a.norm() * b.norm()))
}

fn norm(v: Vec3) -> Vec3 {
    let (x, y, z) = (v.x, v.y, v.z);
    let (x, y, z) = (x / 255.0 * 2.0 - 1.0, y / 255.0 * 2.0 - 1.0, z / 255.0 * 2.0 - 1.0);
    Vec3::new(x, y, z)
}

// we also need to normalize vectors
fn find_simular(original: Vec3) -> Vec3 {
    let mut sim = vec![];

    for &color in &COLOR_SCHEME {
        sim.push((
            sim_fn(&norm(original), &norm(color)),
            color
        ));
    }

    sim.sort_by(|a,b| 
        if       a>b {Ordering::Greater}
        else if  a<b {Ordering::Less   }
        else         {Ordering::Equal  });

    sim[0].1
}

fn main() {

    // init
    let mut img = image::open("test.jpeg")
        .expect("Could not open image");
    let (width, height) = img.dimensions();

    // loop
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let pixel = Vec3::new(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
            let new_pixel = find_simular(pixel);
            img.put_pixel(x, y, image::Rgba([new_pixel.x as u8, new_pixel.y as u8, new_pixel.z as u8, 255]));
        }
    }

    // end
    img.save("output.png").unwrap();
}

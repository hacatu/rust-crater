use std::cmp::Ordering;

use num_traits::NumRef;
use rand::{distributions::{uniform::SampleUniform, Distribution, Uniform}, Rng};

use crate::{KdPoint, KdRegion};


#[derive(Clone, Debug)]
pub struct CuPoint<T: Ord + Clone + NumRef, const N: usize> {
    buf: [T; N]
}

#[derive(Clone, Debug)]
pub struct CuRegion<T: Ord + Clone + NumRef, const N: usize> {
    pub start: CuPoint<T, N>,
    pub end: CuPoint<T, N>
}

impl<T: Ord + Clone + NumRef, const N: usize> KdPoint for CuPoint<T, N> {
    type Distance = T;

    fn cmp(&self, other: &Self, layer: usize) -> Ordering {
        let idx = layer%N;
        if N == 0 {
            Ordering::Equal
        } else {
            self.buf[idx].cmp(&other.buf[idx])
        }
    }

    fn sqdist(&self, other: &Self) -> Self::Distance {
        let mut a = T::zero();
        for i in 0..N {
            let d = self.buf[i].clone() - &other.buf[i];
            a = a + d.clone()*&d;
        }
        a
    }
}

impl<T: Ord + Clone + NumRef + SampleUniform, const N: usize> Distribution<CuPoint<T, N>> for Uniform<T> {
    fn sample<R>(&self, rng: &mut R) -> CuPoint<T, N> where R: Rng + ?Sized {
        CuPoint{buf: match self.sample_iter(rng).take(N).collect::<Vec<T>>().try_into() {
            Ok(buf) => buf, Err(_) => panic!("Couldn't get N values from SampleUniform!")}}
    }
}

impl<T: Ord + Copy + NumRef, const N: usize> Copy for CuPoint<T, N> {}

impl<T: Ord + Clone + NumRef, const N: usize> CuPoint<T, N> {
	pub fn view(&self) -> &[T; N] {
		&self.buf
	}
	
	pub fn extract(self) -> [T; N] {
		self.buf
	}
}

impl<T: Ord + Clone + NumRef, const N: usize> KdRegion for CuRegion<T, N> {
    type Point = CuPoint<T, N>;

    fn split(&self, point: &Self::Point, layer: usize) -> (Self, Self) {
        let mut sub0 = self.clone();
        let mut sub1 = self.clone();
        let idx = layer%N;
        let split_coord = &point.buf[idx];
        sub0.end.buf[idx].clone_from(split_coord);
        sub1.start.buf[idx].clone_from(split_coord);
        (sub0, sub1)
    }

    fn min_sqdist(&self, point: &Self::Point) -> T {
        let mut a = T::zero();
        for i in 0..N {
            let l = &self.start.buf[i];
            let r = &self.end.buf[i];
            let x = &point.buf[i];
            if x < l {
                a = a + l - x;
            } else if r < x {
                a = a + x - r;
            }
        }
        a
    }

    fn extend(&mut self, point: &Self::Point) {
        for (i, x) in (&point.buf).into_iter().enumerate() {
            if x < &self.start.buf[i] {
                self.start.buf[i].clone_from(x);
            } else if x > &self.end.buf[i] {
                self.end.buf[i].clone_from(x);
            }
        }
    }
    
    fn single_point(point: &Self::Point) -> Self {
        Self{start: point.clone(), end: point.clone()}
    }
}

impl<T: Ord + Copy + NumRef, const N: usize> Copy for CuRegion<T, N> {}

#[cfg(test)]
mod tests {
    use rand::distributions::{Distribution, Uniform};

    use crate::kdree::KdTree;

    use super::*;

    const NUM_POINTS: usize = 1000;
    const BOX_SIZE: i64 = 2000;
    const KCS_SIZE: i64 = 2200;
    const KCS_COUNT: usize = 50;
    const KCS_TRIALS: usize = 50;
    const KD_TRIALS: usize = 5;

	fn get_bounds<const N: usize>(kdt: &KdTree::<CuRegion<i64, N>>) -> Option<CuRegion<i64, N>> {
		let mut it = kdt.into_iter();
		let mut res = it.next().map(|pt|CuRegion{start:pt.clone(), end:pt.clone()})?;
		for point in it {
			for i in 0..N {
				let x = &point.buf[i];
				if x < &res.start.buf[i] {
					res.start.buf[i].clone_from(x)
				} else if x > &res.end.buf[i] {
					res.end.buf[i].clone_from(x)
				}
			}
		}
		res.into()
	}
	
    #[test]
    fn pointcloud() {
        let mut rng = rand::thread_rng();
        let box_dist = Uniform::new_inclusive(-BOX_SIZE/2, BOX_SIZE/2);
        let kcs_dist = Uniform::new_inclusive(-KCS_SIZE/2, KCS_SIZE/2);
        for _ in 0..KD_TRIALS {
            eprintln!("Generating {0} random lattice points in [-{1}, {1}]^3", NUM_POINTS, BOX_SIZE/2);
            let mut points: Vec<CuPoint<i64, 3>> = Vec::new();
            for _ in 0..NUM_POINTS {
                points.push(box_dist.sample(&mut rng))
            }
            eprintln!("Checking bounds of points");
            let kdt = KdTree::<CuRegion<i64, 3>>::make(points);
            let bounds = get_bounds(&kdt);
            match (&bounds, &kdt.bounds) {
                (Some(CuRegion{start: a, end: b}), Some(CuRegion{start: c, end: d})) =>
                    if a.view() != c.view() || b.view() != d.view() {
                        panic!("Bounds did not match!")},
                _ => panic!("Failed to get bounds!")
            }
            if !kdt.check_tree() {
                panic!("KD Tree built wrong!")
            }
            for _ in 0..KCS_TRIALS {
                let point: CuPoint<i64, 3> = kcs_dist.sample(&mut rng);
                eprintln!("Getting {} closest points to {:?}", KCS_COUNT, &point);
                let mut res = kdt.k_closest(&point, KCS_COUNT);
                let mut res_naive = kdt.k_closest_naive(&point, KCS_COUNT);
                if res.len() != KCS_COUNT || res_naive.len() != KCS_COUNT {
                    panic!("K Closest and/or K Closest naive failed to get {} points!", KCS_COUNT)
                }
                res.sort_unstable_by_key(|pt|point.sqdist(pt));
                res_naive.sort_unstable_by_key(|pt|point.sqdist(pt));
                if res.into_iter().zip(res_naive).any(|(o, e)|point.sqdist(o) != point.sqdist(e)) {
                    panic!("K Closest and K Closest naive did not get the same sets of points!")
                }
            }
        }
    }
}


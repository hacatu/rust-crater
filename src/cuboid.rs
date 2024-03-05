use std::{array, cmp::{max, Ordering}};

use num_traits::NumRef;
use rand::{distributions::{uniform::SampleUniform, Distribution, Uniform}, Rng};

use crate::{KdPoint, KdRegion};

/// Represents a point in N-dimensional Euclidean space whose coordinates are numeric type T
/// Wrapping CuPoint<i64, 3> or CuPoint<f64, 3> is a good way to get started quickly if you
/// don't need a topologically exotic implementation
#[derive(Clone, Debug, PartialEq)]
pub struct CuPoint<T, const N: usize>
where T: Ord + Clone + NumRef {
    buf: [T; N]
}

/// Represents an axis aligned cuboid region in N-dimensional Euclidean space whose
/// coordinates are numeric type T
#[derive(Clone, Debug)]
pub struct CuRegion<T, const N: usize>
where T: Ord + Clone + NumRef {
    pub start: CuPoint<T, N>,
    pub end: CuPoint<T, N>
}

impl<T, const N: usize> KdPoint for CuPoint<T, N>
where T: Ord + Clone + NumRef {
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
            let (x, y) = (&self.buf[i], &other.buf[i]);
            // compute absolute difference between x and y in a really annoying way because generic
            // math is annoying and we can't just call x.abs_diff(y)
            let d = if x > y { x.clone() - y } else { y.clone() - x };
            a = a + d.clone()*&d;
        }
        a
    }
}

/// Generate a random point in a square/cube/etc.
/// Given a Uniform distribution sampling from a range, this adds the ability to
/// randomly generate CuPoints whose coordinates are iid (independent and identically distributed)
/// from that range
impl<T, const N: usize> Distribution<CuPoint<T, N>> for Uniform<T>
where T: Ord + Clone + NumRef + SampleUniform {
    fn sample<R>(&self, rng: &mut R) -> CuPoint<T, N> where R: Rng + ?Sized {
        CuPoint{buf: array::from_fn(|_|self.sample(rng))}
    }
}

/// Generate a default point (all coordinates zero)
impl<T, const N: usize> Default for CuPoint<T, N>
where T: Ord + Clone + NumRef {
    fn default() -> Self {
        Self{buf: array::from_fn(|_|T::zero())}
    }
}

impl<T: Ord + Copy + NumRef, const N: usize> Copy for CuPoint<T, N> {}
impl<T: Ord + Clone + NumRef, const N: usize> Eq for CuPoint<T, N> {}

impl<T, const N: usize> CuPoint<T, N>
where T: Ord + Clone + NumRef {
    /// make a point with a given value
    pub fn make(buf: [T; N]) -> Self {
        Self{buf}
    }

	/// get readonly access to the buffer
	pub fn view(&self) -> &[T; N] {
		&self.buf
	}
	
	/// consume the point to mutably access the buffer
	pub fn extract(self) -> [T; N] {
		self.buf
	}
}

impl<T, const N: usize> From<[T; N]> for CuPoint<T, N>
where T: Ord + Clone + NumRef {
    fn from(buf: [T; N]) -> Self {
        Self{buf}
    }
}

impl<T, const N: usize> KdRegion for CuRegion<T, N>
where T: Ord + Clone + NumRef {
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
        (&self.start.buf).into_iter().zip(&self.end.buf).zip(&point.buf).fold(T::zero(), |a,((l, r), x)|{
            let d = if x < l {
                l.clone() - x
            } else if r < x {
                x.clone() - r
            } else {
                return a
            };
            a + d.clone()*d
        })
    }

    fn max_sqdist(&self, point: &Self::Point) -> Option<T> {
        Some((&self.start.buf).into_iter().zip(&self.end.buf).zip(&point.buf).fold(T::zero(), |a,((l, r), x)|{
            let d = if x < l {
                r.clone() - x
            } else if r < x {
                x.clone() - l
            } else {
                max(r.clone() - x, x.clone() - l)
            };
            a + d.clone()*d
        }))
    }

    fn might_overlap(&self, other: &Self) -> bool {
        (&self.start.buf).into_iter().zip(&self.end.buf).zip((&other.start.buf).into_iter().zip(&other.end.buf)).all(|((a,b),(c,d))|{
            !(b < c || d < a)
        })
    }

    fn is_superset(&self, other: &Self) -> bool {
        (&self.start.buf).into_iter().zip(&self.end.buf).zip((&other.start.buf).into_iter().zip(&other.end.buf)).all(|((a,b),(c,d))|{
            a <= c && d <= b
        })
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
    use crate::kdtree::{KdTree, QueryOptions};

    use super::*;

    const NUM_POINTS: usize = 1000;
    const BOX_SIZE: i64 = 2000;
    const KCS_SIZE: i64 = 2200;
    const KCS_COUNT: usize = 50;
    const KCS_TRIALS: usize = 50;
    const KD_TRIALS: usize = 5;

	fn get_bounds<const N: usize>(kdt: &KdTree::<CuRegion<i64, N>>) -> Option<CuRegion<i64, N>> {
		let mut it = kdt.iter_points();
		let mut res = it.next().map(|p|CuRegion{start:p.clone(), end:p.clone()})?;
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
            let mut points: Vec<(CuPoint<i64, 3>, ())> = Vec::new();
            for _ in 0..NUM_POINTS {
                points.push((box_dist.sample(&mut rng), ()))
            }
            eprintln!("Checking bounds of points");
            let kdt: KdTree<_> = points.into_iter().collect();
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
                let mut res: Vec<_> = kdt.k_closest(&point, KCS_COUNT, QueryOptions::ALL_NO_TIES).into();
                let mut res_naive: Vec<_> = kdt.k_closest_naive(&point, KCS_COUNT).into();
                if res.len() != KCS_COUNT || res_naive.len() != KCS_COUNT {
                    panic!("K Closest and/or K Closest naive failed to get {} points!", KCS_COUNT)
                }
                res.sort_unstable_by_key(|(p,_)|point.sqdist(p));
                res_naive.sort_unstable_by_key(|(p,_)|point.sqdist(p));
                if res.into_iter().zip(res_naive).any(|((o,_), (e,_))|point.sqdist(o) != point.sqdist(e)) {
                    panic!("K Closest and K Closest naive did not get the same sets of points!")
                }
            }
        }
    }
}


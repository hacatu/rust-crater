use std::cmp::{max, Ordering};

use num_traits::NumRef;

use rand::{distributions::{uniform::SampleUniform, Distribution, Uniform}, Rng};

pub trait KdPoint: Sized {
    type Distance: Ord;
    fn sqdist(&self, other: &Self) -> Self::Distance;
    fn cmp(&self, other: &Self, layer: usize) -> Ordering;
}

pub trait KdRegion: Sized {
    type Point: KdPoint;
    fn split(&self, point: &Self::Point, layer: usize) -> (Self, Self);
    fn extend(&mut self, point: &Self::Point);
    fn single_point(point: &Self::Point) -> Self;
    fn min_sqdist(&self, point: &Self::Point) -> <Self::Point as KdPoint>::Distance;
}

pub struct KdTree<R: KdRegion> {
    pub bounds: Option<R>,
    points: Vec<R::Point>
}

pub enum WalkDecision {
    Continue,
    SkipChildren,
    Stop
}

pub fn get_bounds<'a, R: KdRegion>(points: impl IntoIterator<Item = &'a R::Point>) -> Option<R> where R: 'a {
    let mut it = points.into_iter();
    let mut res = R::single_point(it.next()?);
    for p in it {
        res.extend(p);
    }
    Some(res)
}

pub struct MmHeap<T> {
    buf: Vec<T>
}

impl<T> MmHeap<T> {
    pub fn new() -> Self {
        Self{buf: Vec::new()}
    }

    pub fn make(buf: Vec<T>, cmp: &impl Fn(&T, &T) -> Ordering) -> Self {
        let mut res = Self{buf};
        res.ify_by(cmp);
        res
    }

    pub fn ify_by(&mut self, cmp: &impl Fn(&T, &T) -> Ordering) {
        let nonleaf_idx_upper_bound = (usize::MAX >> 1) >> self.buf.len().leading_zeros();
        for i in (0..nonleaf_idx_upper_bound).rev() {
            self.sift_down_by(i, cmp);
        }
    }

    pub fn peak_min(&self) -> Option<&T> {
        self.buf.first()
    }

    pub fn peak_max_by(&self, cmp: &impl Fn(&T, &T) -> Ordering) -> Option<&T> {
        match self.buf.get(1..3) {
            Some(slice_ref) => slice_ref.into_iter().max_by(|a, b|cmp(a, b)),
            None => self.buf.get(max(self.buf.len(), 1) - 1)
        }
    }

    pub fn sift_up_by(&mut self, mut i: usize, cmp: &impl Fn(&T, &T) -> Ordering) {
        if i == 0 || i >= self.buf.len() {
            return;
        }
        let mut ord = if (i + 1).leading_zeros()&1 == 1 {
            Ordering::Less
        } else {
            Ordering::Greater
        };
        let mut i1 = (i - 1) >> 1;
        if cmp(self.buf.get(i1).unwrap(), self.buf.get(i).unwrap()) == ord {
            self.buf.swap(i, i1);
            i = i1;
            ord = ord.reverse()
        }
        while i > 2 {
            i1 = (i - 3) >> 2;
            if cmp(self.buf.get(i).unwrap(), self.buf.get(i1).unwrap()) == ord {
                self.buf.swap(i, i1);
                i = i1;
            } else {
                break
            }
        }
    }

    pub fn sift_down_by(&mut self, mut i: usize, cmp: &impl Fn(&T, &T) -> Ordering) {
        let ord = if (i + 1).leading_zeros()&1 == 1 {
            Ordering::Less
        } else {
            Ordering::Greater
        };
        while 2*i + 1 < self.buf.len() {
            // Find m, the index of the extremal element among the children and grandchildren
            // of the element at index i. For min layers, extremal means
            // minimal, and for max layers it means maximal
            let mut m = 2*i + 1;
            for ii in [2*i + 2, 4*i + 3, 4*i + 4, 4*i + 5, 4*i + 6].into_iter().take_while(|&j|j<self.buf.len()) {
                if cmp(self.buf.get(ii).unwrap(), self.buf.get(m).unwrap()) == ord {
                    m = ii;
                }
            }
            // If m is a grandchild of i (as should be the case most of the time)
            // we may have to sift down farther after fixing up here
            if m > 2*i + 2 {
                if cmp(self.buf.get(m).unwrap(), self.buf.get(i).unwrap()) == ord {
                    self.buf.swap(m, i);
                    let p = (m - 1) >> 1;
                    if cmp(self.buf.get(p).unwrap(), self.buf.get(m).unwrap()) == ord {
                        self.buf.swap(m, p);
                    }
                    i = m;
                }else{
                    break
                }
            } else {// otherwise em is a direct child so it must be a leaf or its invariant would be wrong
                if cmp(self.buf.get(m).unwrap(), self.buf.get(i).unwrap()) == ord {
                    self.buf.swap(m, i);
                }
                break
            }
        }
    }

    pub fn push_by(&mut self, e: T, cmp: &impl Fn(&T, &T) -> Ordering) {
        self.buf.push(e);
        self.sift_up_by(self.buf.len() - 1, cmp)
    }

    pub fn pop_min_by(&mut self, cmp: &impl Fn(&T, &T) -> Ordering) -> Option<T> {
        self.pop_idx_by(0, cmp)
    }

    pub fn pop_max_by(&mut self, cmp: &impl Fn(&T, &T) -> Ordering) -> Option<T> {
        match self.buf.get(1..3) {
            Some(slice_ref) => self.pop_idx_by(1 + slice_ref.into_iter().enumerate().max_by(|(_i,a),(_j,b)|cmp(a,b)).unwrap().0, cmp),
            None => self.buf.pop()
        }
    }

    pub fn pop_idx_by(&mut self, i: usize, cmp: &impl Fn(&T, &T) -> Ordering) -> Option<T> {
        let l = self.buf.len();
        if i + 1 >= l {
            return self.buf.pop()
        }
        self.buf.swap(i, l - 1);
        let res = self.buf.pop();
        self.sift_down_by(i, cmp);
        res
    }

    pub fn pushpop_min_by(&mut self, e: T, cmp: &impl Fn(&T, &T) -> Ordering) -> T {
        if self.buf.is_empty() {
            return e
        }
        self.push_by(e, cmp);
        self.pop_min_by(cmp).unwrap()
    }

    pub fn pushpop_max_by(&mut self, e: T, cmp: &impl Fn(&T, &T) -> Ordering) -> T {
        if self.buf.is_empty() {
            return e
        }
        self.push_by(e, cmp);
        self.pop_max_by(cmp).unwrap()
        
    }
}

impl<'a, T> IntoIterator for &'a MmHeap<T> {
    type Item = &'a T;
    type IntoIter = <&'a Vec<T> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        (&self.buf).into_iter()
    }
}

impl<T> Into<Vec<T>> for MmHeap<T> {
    fn into(self) -> Vec<T> {
        self.buf
    }
}

impl<R: KdRegion> KdTree<R> {
    pub fn make(points: Vec<R::Point>) -> Self {
        let mut res = Self{bounds: get_bounds(&points), points};
        res.ify();
        res
    }
    pub fn ify(&mut self) {
        self.ify_r(0, self.points.len(), 0)
    }
    pub fn walk<'a>(&'a self, visitor: &mut impl FnMut(&R, &'a R::Point) -> WalkDecision) {
        if let Some(bounds) = self.bounds.as_ref() {
            self.walk_r(visitor, bounds, 0, self.points.len(), 0);
        }
    }
    pub fn k_closest<'a>(&'a self, point: &R::Point, k: usize) -> Vec<&'a R::Point> {
        let mut res = MmHeap::new();
        let mut max_sqdist = None;
        let cmp_fn = &|a: &&R::Point, b: &&R::Point|point.sqdist(a).cmp(&point.sqdist(b));
        self.walk(&mut |bounds: &R, pt: &'a R::Point|{
            if res.buf.len() < k {
                res.push_by(pt, cmp_fn);
            } else {
                max_sqdist = Some(point.sqdist(res.pushpop_max_by(pt, cmp_fn)))
            }
            if max_sqdist.as_ref().is_some_and(|d|d <= &bounds.min_sqdist(pt)) {
                return WalkDecision::SkipChildren;
            }
            return WalkDecision::Continue;
        });
        res.into()
    }
    fn ify_r(&mut self, a: usize, mut b: usize, mut layer: usize) {
        while a < b {
            let med_idx = (a + b)/2;
            self.points.get_mut(a..b).unwrap().select_nth_unstable_by(med_idx - a, |p, q| p.cmp(q, layer));
            layer += 1;
            self.ify_r(med_idx + 1, b, layer);
            b = med_idx;
        }
    }
    fn walk_r<'a>(&'a self, visitor: &mut impl FnMut(&R, &'a R::Point) -> WalkDecision, bounds: &R, a: usize, mut b: usize, mut layer: usize) -> WalkDecision {
        let mut sub0_holder;
        let mut sub0 = bounds;
        while a < b {
            let mid_idx = (a + b)/2;
            // safe because a < b <= self.points.len() so mid_idx < self.points.len()
            let point: &'a R::Point = self.points.get(mid_idx).unwrap();
            match visitor(sub0, point) {
                WalkDecision::Stop => return WalkDecision::Stop,
                WalkDecision::SkipChildren => return WalkDecision::Continue,
                WalkDecision::Continue => ()
            }
            layer += 1;
            let tmp = sub0.split(point, layer);
            sub0_holder = tmp.0; // We can't make a structured bindee outlive its scope
            sub0 = &sub0_holder;
            match self.walk_r(visitor, &tmp.1, mid_idx + 1, b, layer) {
                WalkDecision::Stop => return WalkDecision::Stop,
                _ => ()
            }
            b = mid_idx
        }
        WalkDecision::Continue
    }

    #[cfg(test)]
    fn check_layer(&self, a: usize, b: usize, layer: usize) -> bool {
        if b > self.points.len() || a > b {
            return false
        } else if a == b {
            return true
        }
        let mid_idx = (a + b)/2;
        let m = self.points.get(mid_idx).unwrap();
        for e in self.points.get(a..mid_idx).unwrap_or(&[]) {
            if e.cmp(m, layer) == Ordering::Greater {
                return false;
            }
        }
        for e in self.points.get(mid_idx+1..b).unwrap_or(&[]) {
            if e.cmp(m, layer) == Ordering::Less {
                return false;
            }
        }
        true
    }

    #[cfg(test)]
    fn check_tree_r(&self, a: usize, mut b: usize, mut layer: usize) -> bool {
        while b > a {
            if !self.check_layer(a, b, layer) {
                return false;
            }
            let mid_idx = (a + b)/2;
            layer += 1;
            if !self.check_tree_r(mid_idx + 1, b, layer) {
                return false;
            }
            b = mid_idx;
        }
        true
    }

    #[cfg(test)]
    fn check_tree(&self) -> bool {
        self.check_tree_r(0, self.points.len(), 0)
    }

    #[cfg(test)]
    fn k_closest_naive<'a>(&'a self, point: &R::Point, k: usize) -> Vec<&'a R::Point> {
        let mut res = MmHeap::new();
        let cmp_fn = &|a: &&R::Point, b: &&R::Point|point.sqdist(a).cmp(&point.sqdist(b));
        (&self.points).into_iter().for_each(&mut |pt: &'a R::Point|{
            if res.buf.len() < k {
                res.push_by(pt, cmp_fn)
            } else {
                res.pushpop_max_by(pt, cmp_fn);
            }
        });
        res.into()
    }
}

#[derive(Clone, Debug)]
pub struct CuPoint<T: Ord + Clone + NumRef, const N: usize> {
    buf: [T; N]
}

impl<T: Ord + Clone + NumRef, const N: usize> KdPoint for CuPoint<T, N> {
    type Distance = T;
    fn cmp(&self, other: &Self, layer: usize) -> Ordering {
        let idx = layer%N;
        if N == 0 {
            Ordering::Equal
        } else {
            self.buf.get(idx).unwrap().cmp(other.buf.get(idx).unwrap())
        }
    }
    fn sqdist(&self, other: &Self) -> Self::Distance {
        let mut a = T::zero();
        for i in 0..N {
            let d = self.buf.get(i).unwrap().clone() - other.buf.get(i).unwrap();
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

#[derive(Clone, Debug)]
pub struct CuRegion<T: Ord + Clone + NumRef, const N: usize> {
    pub start: CuPoint<T, N>,
    pub end: CuPoint<T, N>
}

impl<T: Ord + Clone + NumRef, const N: usize> KdRegion for CuRegion<T, N> {
    type Point = CuPoint<T, N>;
    fn split(&self, point: &Self::Point, layer: usize) -> (Self, Self) {
        let mut sub0 = self.clone();
        let mut sub1 = self.clone();
        let idx = layer%N;
        let split_coord = point.buf.get(idx).unwrap();
        sub0.end.buf.get_mut(idx).unwrap().clone_from(split_coord);
        sub1.start.buf.get_mut(idx).unwrap().clone_from(split_coord);
        (sub0, sub1)
    }
    fn min_sqdist(&self, point: &Self::Point) -> T {
        let mut a = T::zero();
        for i in 0..N {
            let l = self.start.buf.get(i).unwrap();
            let r = self.end.buf.get(i).unwrap();
            let x = point.buf.get(i).unwrap();
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
            if x < self.start.buf.get(i).unwrap() {
                self.start.buf.get_mut(i).unwrap().clone_from(x);
            } else if x > self.end.buf.get(i).unwrap() {
                self.end.buf.get_mut(i).unwrap().clone_from(x);
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

    use super::*;

    const NUM_POINTS: usize = 1000;
    const BOX_SIZE: i64 = 2000;
    const KCS_SIZE: i64 = 2200;
    const KCS_COUNT: usize = 50;
    const KCS_TRIALS: usize = 50;
    const KD_TRIALS: usize = 5;

    fn get_bounds<const N: usize>(kdt: &KdTree::<CuRegion<i64, N>>) -> Option<CuRegion<i64, N>> {
        let mut res = kdt.points.first().map(|pt|CuRegion{start:pt.clone(), end:pt.clone()})?;
        for i in 0..N {
            for point in kdt.points.get(1..).unwrap() {
                let x = point.buf.get(i).unwrap();
                if x < res.start.buf.get(i).unwrap() {
                    res.start.buf.get_mut(i).unwrap().clone_from(x)
                } else if x > res.end.buf.get(i).unwrap() {
                    res.end.buf.get_mut(i).unwrap().clone_from(x)
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
            eprintln!("Got {:?}", &points);
            eprintln!("Checking bounds of points");
            let kdt = KdTree::<CuRegion<i64, 3>>::make(points);
            let bounds = get_bounds(&kdt);
            match (&bounds, &kdt.bounds) {
                (Some(CuRegion{start: a, end: b}), Some(CuRegion{start: c, end: d})) =>
                    if a.buf != c.buf || b.buf != d.buf {
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
                eprintln!("K Closest got {:?}", res);
                eprintln!("Naive got {:?}", res);
                res.sort_unstable_by_key(|pt|point.sqdist(pt));
                res_naive.sort_unstable_by_key(|pt|point.sqdist(pt));
                if res.into_iter().zip(res_naive).any(|(o, e)|point.sqdist(o) != point.sqdist(e)) {
                    panic!("K Closest and K Closest naive did not get the same sets of points!")
                }
            }
        }
    }
}

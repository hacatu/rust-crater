use crate::{get_bounds, mmheap::MmHeap, KdPoint, KdRegion, WalkDecision};

/// A KD tree represents a collection of points in space, with the ability to answer distance related queries, mainly:
/// - What are the k closest points to a given point?  (The first k points in the tree, ordered by increasing squared
///   distance from the given point, including the given point if it is in the tree).  (Sometimes called a nearest
///   neighbors query)
/// - What are all the points within a distance d of a given point?  (All points whose squared distance from a given
///   point is at most d^2).  (Sometimes called a ball query)
/// - What are all the points within a given region?  (Sometimes called a cuboid query)
/// This implementation uses an implicit tree, meaning all points are stored in one contiguous buffer and no dereferences
/// are needed to traverse the tree.  This is good for lookup performance, but unfortunately it means that adding/removing
/// points can't be done currently without rebuilding the tree.
/// Besides the three basic distance related queries, KD trees can be used to some extent to help with more complicated
/// distance related queries, like finding the closest pairs of points.
pub struct KdTree<R: KdRegion> {
    pub bounds: Option<R>,
    points: Vec<R::Point>
}

impl<R: KdRegion> KdTree<R> {
	/// Construct a KD tree out of a vector of points, moving the vector into the output
	/// Calculates the bounds and builds the implict tree
    pub fn make(points: Vec<R::Point>) -> Self {
        let mut res = Self{bounds: get_bounds(&points), points};
        res.ify();
        res
    }

	/// Iterate over all points in the tree in depth first order,
	/// calling a visitor function on each point.  The visitor function
	/// is also given the bounds of the subtree corresponding to the point,
	/// and may return a WalkDecision to instruct the traversal to skip the
	/// subtree or to stop the traversal entirely.
    pub fn walk<'a>(&'a self, visitor: &mut impl FnMut(&R, &'a R::Point) -> WalkDecision) {
        if let Some(bounds) = self.bounds.as_ref() {
            self.walk_r(visitor, bounds, 0, self.points.len(), 0);
        }
    }

	/// Return the k points in the tree which are the closest to a given point.
	/// Ties are broken arbitrarily.  If there are fewer than k points in the tree,
	/// returns all the points.
    pub fn k_closest<'a>(&'a self, point: &R::Point, k: usize) -> Vec<&'a R::Point> {
        let mut res = MmHeap::new();
        let mut max_sqdist = None;
        let cmp_fn = &|a: &&R::Point, b: &&R::Point|point.sqdist(a).cmp(&point.sqdist(b));
        self.walk(&mut |bounds: &R, pt: &'a R::Point|{
            if res.len() < k {
                res.push_by(pt, cmp_fn);
            } else {
                max_sqdist = Some(point.sqdist(res.pushpop_max_by(pt, cmp_fn)))
            }
            if max_sqdist.as_ref().is_some_and(|d|d <= &bounds.min_sqdist(pt)) {
                return WalkDecision::SkipChildren;
            }
            WalkDecision::Continue
        });
        res.into()
    }

    fn ify(&mut self) {
        self.ify_r(0, self.points.len(), 0)
    }

    fn ify_r(&mut self, a: usize, mut b: usize, mut layer: usize) {
        while a < b {
            let med_idx = (a + b)/2;
            self.points[a..b].select_nth_unstable_by(med_idx - a, |p, q| p.cmp(q, layer));
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
            let point: &'a R::Point = &self.points[mid_idx];
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
        use std::cmp::Ordering;

        if b > self.points.len() || a > b {
            return false
        } if a == b {
            return true
        }
        let mid_idx = (a + b)/2;
        let m = &self.points[mid_idx];
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
    pub(crate) fn check_tree(&self) -> bool {
        self.check_tree_r(0, self.points.len(), 0)
    }

    #[cfg(test)]
    pub(crate) fn k_closest_naive<'a>(&'a self, point: &R::Point, k: usize) -> Vec<&'a R::Point> {
        let mut res = MmHeap::new();
        let cmp_fn = &|a: &&R::Point, b: &&R::Point|point.sqdist(a).cmp(&point.sqdist(b));
        (&self.points).into_iter().for_each(&mut |pt: &'a R::Point|{
            if res.len() < k {
                res.push_by(pt, cmp_fn)
            } else {
                res.pushpop_max_by(pt, cmp_fn);
            }
        });
        res.into()
    }
}

impl<'a, R: KdRegion> IntoIterator for &'a KdTree<R> {
	type Item = &'a R::Point;
	type IntoIter = <&'a Vec<R::Point> as IntoIterator>::IntoIter;
	fn into_iter(self) -> Self::IntoIter {
		(&self.points).into_iter()
	}
}


use std::{cmp::Ordering, mem, ops::{Index, IndexMut}};

use num_traits::Zero;

use crate::{get_bounds, mmheap::MmHeap, KdPoint, KdRegion, WalkDecision};

/// Represents an inner / outer limit for a query within the tree (`KdTree::k_closest`).
/// See `QueryOptions` for more info
pub enum QueryBound<'a, R: KdRegion> {
    SqDist(<R::Point as KdPoint>::Distance),
    Region(&'a R),
    None
}

/// Fine grained control over a query within the tree (`KdTree::k_closest`).
pub struct QueryOptions<'a, R: KdRegion> {
    /// Points within this bound will be EXCLUDED from the query result.
    /// Points ON the inner boundary will also be EXCLUDED (ie the excluded region is a closed set).
    /// Useful to exclude points that are "too close" to the query point,
    /// for example when iteratively increasing a sampling radius in a shortest path algorithm
    /// or some other situation where the number of k_closest points needed is not know a prior.
    /// For `QueryBound::SqDist(d)`, this means points whose sqdist from the query point is <= d will be excluded.
    /// Since the query point itself is included by default if it is in the tree, passing `QueryBound::SqDist(Distance::zero())`
    /// here can be used to exclude it.
    /// For `QueryBound::Region(&r)`, this means points within r (where r.min_sqdist is Distance::zero()) will excluded.
    /// NB this cannot be conservative like an unbounded search.
    /// That is, if `R::min_sqdist` can be an underestimate, then `QueryBound::Region` should not be used.
    /// For `QueryBound::None`, no points will be excluded for being "too close" to the query point.
    pub inner_bound: QueryBound<'a, R>,
    /// Points outside this bound will be EXCLUDED from the query result.
    /// Points ON the outer boundary will be INCLUDED (ie the excluded region is an open set).
    /// This makes the overall included region a half open set, excluding the inner bound but including
    /// the outer bound.  This is to make it easy to have successive queries cover additional area without duplicate results.
    /// For `QueryBound::SqDist(d)`, points whose sqdist from the query point is > d will be excluded.
    /// For `QueryBount::Region(&r)`, points outside r (where r.min_sqdist is not Distance::zero()) will be excluded.
    /// Unlike the inner bound, this will work correctly with a conservative `R::min_sqdist` (ie if min_sqdist is an underestimate
    /// it can only cause the result to include extra points, not miss points that should be included).
    /// For `QueryBound::None`, no points will be excluded for being "too far" from the query point.
    pub outer_bound: QueryBound<'a, R>,
    /// If false, if multiple points are tied for being the kth closest, which one will be returned is arbitrary,
    /// but exactly k points will be returned.
    /// If true, if multiple points are tied for being the kth closest, all of them will be returned,
    /// and more than k points will be returned in this case.
    /// Setting this to true is necessary to correctly call `k_closest` with an iteratively increasing sampling
    /// radius.
    pub keep_ties: bool
}

impl<'a, R: KdRegion> QueryOptions<'a, R> {
    /// Default QueryOptions to have the included region be the entire tree,
    /// and return all points which are tied for being the kth-closest
    pub const ALL_KEEP_TIES: Self = Self{inner_bound: QueryBound::None, outer_bound: QueryBound::None, keep_ties: true};
    /// Default QueryOptions to have the included region be the entire tree,
    /// and arbitrarily break ties for the kth-closest point so no more than k points are ever returned.
    pub const ALL_NO_TIES: Self = Self{inner_bound: QueryBound::None, outer_bound: QueryBound::None, keep_ties: false};
    /// Returns true if the included region contains the point `pt`, where `point` is the center of the query,
    /// or false if `pt` is in the excluded region.
    pub fn contains(&self, point: &R::Point, pt: &R::Point) -> bool {
        match &self.outer_bound {
            QueryBound::SqDist(max_sqdist) => if point.sqdist(pt) > *max_sqdist { return false },
            QueryBound::Region(r) => if r.min_sqdist(pt) > <R::Point as KdPoint>::Distance::zero() { return false },
            QueryBound::None => ()
        }
        match &self.inner_bound {
            QueryBound::SqDist(lb_sqdist) => point.sqdist(pt) > *lb_sqdist,
            QueryBound::Region(r) => r.min_sqdist(pt) > <R::Point as KdPoint>::Distance::zero(),
            QueryBound::None => true
        }
    }
    /// Returns true if the included region might overlap with `bounds` and
    /// might contain additional points within `max_sqdist` of the center of the query, `point`.
    /// When some of the required functions of `<R as KdRegion>` are implemented conservatively,
    /// this function will return true in cases where the query region and `bounds` can't
    /// actually contain any additional points in the result set.
    /// However, when this function returns false, the query region will always be disjoint from `bounds`,
    /// UNLESS self.inner_bound is not `QueryBound::None` and a conservative implementation is used for KdRegion.
    pub fn might_overlap(&self, point: &R::Point, max_sqdist: Option<&<R::Point as KdPoint>::Distance>, bounds: &R) -> bool {
        if let Some(d) = max_sqdist {
            match bounds.min_sqdist(point).cmp(d) {
                Ordering::Greater => return false,
                Ordering::Equal => if !self.keep_ties { return false },
                Ordering::Less => ()
            }
        }
        match &self.outer_bound {
            QueryBound::SqDist(outer_sqdist) => if bounds.min_sqdist(point) > *outer_sqdist { return false },
            QueryBound::Region(r) => if !r.might_overlap(bounds) { return false },
            QueryBound::None => ()
        }
        match &self.inner_bound {
            QueryBound::SqDist(inner_sqdist) => !bounds.max_sqdist(point).is_some_and(|d|d <= *inner_sqdist),
            QueryBound::Region(r) => !r.is_superset(bounds),
            QueryBound::None => true
        }
    }
}

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
#[derive(Debug)]
pub struct KdTree<R: KdRegion, V = ()> {
    pub bounds: Option<R>,
    points: Box<[(R::Point, V)]>
}

impl<R: KdRegion, V> KdTree<R, V> {
    /// Get the number of points in the kdtree
    pub fn len(&self) -> usize {
        self.points.len()
    }

	/// Iterate over all point, value pairs in the tree in depth first order,
	/// calling a visitor function on each.  The visitor function gets a const reference
	/// to the point, the value, and the bounds of the subtree corresponding to the point,
	/// and may return a WalkDecision to instruct the traversal to skip the
	/// subtree or to stop the traversal entirely.
    pub fn walk<'a>(&'a self, visitor: &mut impl FnMut(&R, &'a R::Point, &'a V) -> WalkDecision) {
        let Some(bounds) = self.bounds.clone() else { return };
        let mut todo = vec![(bounds, 0, self.len(), 0)];
        while let Some((bounds, a, b, layer)) = todo.pop() {
            let mid_idx = (a + b)/2;
            // safe because a < b <= self.points.len() so mid_idx < self.points.len()
            let point = &self.points[mid_idx].0;
            let value = &self.points[mid_idx].1;
            match visitor(&bounds, point, value) {
                WalkDecision::Stop => return,
                WalkDecision::SkipChildren => continue,
                WalkDecision::Continue => ()
            }
            let (sub0, sub1) = bounds.split(point, layer);
            if a < mid_idx { todo.push((sub0, a, mid_idx, layer + 1)) }
            if mid_idx + 1 < b { todo.push((sub1, mid_idx + 1, b, layer + 1)) }
        }
    }

    /// Iterate over all point, value pairs in the tree in depth first order,
    /// calling a visitor function on each.  The visitor function gets a mutable reference
    /// to the value, but a const reference
	/// to the point and the bounds of the subtree corresponding to the point,
	/// and may return a WalkDecision to instruct the traversal to skip the
	/// subtree or to stop the traversal entirely.
    pub fn walk_mut<'a>(&'a mut self, visitor: &mut impl FnMut(&R, &'a R::Point, &mut V) -> WalkDecision) {
        let Some(bounds) = self.bounds.clone() else { return };
        let mut todo = vec![(bounds, 0, self.len(), 0)];
        while let Some((bounds, a, b, layer)) = todo.pop() {
            let mid_idx = (a + b)/2;
            // safe because a < b <= self.points.len() so mid_idx < self.points.len()
            let point = &self.points[mid_idx].0;
            let value = &mut self.points[mid_idx].1;
            match visitor(&bounds, point, value) {
                WalkDecision::Stop => return,
                WalkDecision::SkipChildren => continue,
                WalkDecision::Continue => ()
            }
            let (sub0, sub1) = bounds.split(point, layer);
            if a < mid_idx { todo.push((sub0, a, mid_idx, layer + 1)) }
            if mid_idx + 1 < b { todo.push((sub1, mid_idx + 1, b, layer + 1)) }
        }
    }

	/// Return the k points in the tree which are the closest to a given point.
    /// Behavior can be fine-tuned using `cfg`:
    /// - Can restrict the result set to only include points within a certain sqdist of the query point
    ///   or withn a certain region, as well as to exclude points within another smaller sqdist or subregion.
    /// - Can specify how any points tied for kth closest should be handled (keep all or keep one arbitrarily).
	/// If there are fewer than k points in the tree, returns all the points.
    /// If k is 0, returns an empty minmax heap without looking at the tree
    pub fn k_closest<'a>(&'a self, point: &R::Point, k: usize, cfg: QueryOptions<'a, R>) -> MmHeap<(&'a R::Point, &'a V)> {
        let mut res = MmHeap::new();
        if k == 0 { return res }
        let mut tied_points = Vec::new();
        let mut max_sqdist = if let QueryBound::SqDist(max_sqdist) = &cfg.outer_bound { Some(max_sqdist.clone()) } else { None };
        let cmp_fn = &|&(a, _): &_, &(b, _): &_|point.sqdist(a).cmp(&point.sqdist(b));
        self.walk(&mut |bounds, pt, v|{
            if cfg.contains(point, pt) {
                if res.len() + tied_points.len() < k {
                    res.push_by((pt, v), cmp_fn);
                } else if !cfg.keep_ties {
                    max_sqdist = Some(point.sqdist(res.pushpop_max_by((pt, v), cmp_fn).0))
                } else {
                    if max_sqdist.is_none() {
                        max_sqdist = Some(point.sqdist(res.peek_max_by(cmp_fn).unwrap().0))
                    }
                    match point.sqdist(pt).cmp(max_sqdist.as_ref().unwrap()) {
                        Ordering::Greater => (),
                        Ordering::Equal => tied_points.push((pt, v)),
                        Ordering::Less => if res.len() + 1 == k {
                            tied_points.clear();
                            tied_points.push(res.pushpop_max_by((pt, v), cmp_fn));
                            max_sqdist = Some(point.sqdist(tied_points[0].0));
                            while res.peek_max_by(cmp_fn).is_some_and(
                                |&(p, _)|point.sqdist(p) == *max_sqdist.as_ref().unwrap()
                            ) {
                                tied_points.push(res.pop_max_by(cmp_fn).unwrap())
                            }
                        }
                    }
                }
            } else if !cfg.might_overlap(point, max_sqdist.as_ref(), bounds) {
                return WalkDecision::SkipChildren;
            }
            WalkDecision::Continue
        });
        res.extend_by(tied_points, cmp_fn);
        res
    }

    /// Borrowing iterator over only references to points.
    /// The order of the result is arbitrary, but all points will be visited exactly once.
    pub fn iter_points(&self) -> impl Iterator<Item=&R::Point> + '_ {
        self.into_iter().map(|(p,_)|p)
    }

    /// Consuming iterator over only points, which are moved out of self.
    /// The order of the result is arbitrarily, but all points will be visited exactly once.
    pub fn into_points(self) -> impl Iterator<Item=R::Point> {
        self.into_iter().map(|(p,_)|p)
    }

    /// Borrowing iterator over only references to values.
    /// The order of the result is arbitrary, but the value for each point will be visited exactly once.
    pub fn iter_values(&self) -> impl Iterator<Item=&V> + '_ {
        self.into_iter().map(|(_,v)|v)
    }

    /// Mutable borrowing iterator over only references to values.
    /// The order of the result is arbitrary, but the value for each point will be visited exactly once.
    /// This is a splitting borrow, so it is safe to store a reference to some value and mutate it even
    /// after subsequent values have been visited.
    pub fn mut_values(&mut self) -> impl Iterator<Item=&mut V> + '_ {
        self.into_iter().map(|(_,v)|v)
    }

    /// Consuming iterator over only values, which are moved out of self.
    /// The order of the result is arbitrary, but the value for each point will be visited exactly once.
    pub fn into_values(self) -> impl Iterator<Item=V> {
        self.into_iter().map(|(_,v)|v)
    }

    /// Return a reference to the value for some point in the tree, or None if the point is not found
    pub fn get(&self, point: &R::Point) -> Option<&V> {
        self.points.get(self.find(point)).map(|(_,v)|v)
    }

    /// Return a reference to a point in the tree and the corresponding value, or None if the point is not found
    pub fn get_point_value(&self, point: &R::Point) -> Option<(&R::Point, &V)> {
        self.points.get(self.find(point)).map(|(p,v)|(p,v))
    }

    /// Return true if the tree contains some point or false otherwise
    pub fn contains_point(&self, point: &R::Point) -> bool {
        self.find(point) != self.len()
    }

    /// Return a mutable reference to the value for some point in the tree, or None if the point is not found
    pub fn get_mut(&mut self, point: &R::Point) -> Option<&mut V> {
        self.points.get_mut(self.find(point)).map(|(_,v)|v)
    }

    /// Convert a const reference to a point in the tree into an internal index.
    /// This function is unsafe because it can't be used productively without
    /// `launder_idx`.  `ent` MUST be a reference to one of the points actually stored in
    /// the tree, NOT an identical point elsewhere, or this function invokes undefined behavior.
    pub unsafe fn launder_point_ref(&self, ent: &R::Point) -> usize {
        (ent as *const R::Point as *const (R::Point, V))
            .offset_from(self.points.as_ptr()) as usize
    }

    /// Convert a const reference to a value in the tree into an internal index.
    /// This function is unsafe because it can't be used productively without
    /// `launder_idx`.  `ent` MUST be a reference to one of the values actually stored in
    /// the tree, NOT an identical point elsewhere, or this function invokes undefined behavior.
    pub unsafe fn launder_value_ref(&self, ent: &V) -> usize {
        ((ent as *const V).byte_sub(mem::offset_of!((R::Point, V), 1))
            as *const (R::Point, V)
        ).offset_from(self.points.as_ptr()) as usize
    }

    /// Convert an internal index into a mutable reference to a value in the tree.
    /// The internal index must have come from `launder_point_ref` or `launder_value_ref`
    /// called on the same tree.
    /// The intent of this function is to allow mutating the values of the points in the
    /// result set of `k_closest` etc.
    pub unsafe fn launder_idx(&mut self, idx: usize) -> &mut V {
        &mut self.points[idx].1
    }

    fn find(&self, point: &R::Point) -> usize {
        self.find_r(point, 0, self.len(), 0)
    }

    fn find_r(&self, point: &R::Point, mut a: usize, mut b: usize, mut layer: usize) -> usize {
        while a < b {
            let mid_idx = (a + b)/2;
            let p = &self.points[mid_idx].0;
            match point.cmp(p, layer) {
                Ordering::Less => b = mid_idx,
                Ordering::Greater => a = mid_idx + 1,
                Ordering::Equal => {
                    if point == p { return mid_idx }
                    a = self.find_r(point, a, mid_idx, layer + a);
                    if a != self.len() { return a }
                    a = mid_idx + 1
                }
            }
            layer += 1
        }
        self.len()
    }

    fn ify(&mut self) {
        self.ify_r(0, self.points.len(), 0)
    }

    fn ify_r(&mut self, a: usize, mut b: usize, mut layer: usize) {
        while a < b {
            let med_idx = (a + b)/2;
            self.points[a..b].select_nth_unstable_by(med_idx - a, |(p, _), (q, _)| p.cmp(q, layer)); // rust picks up Ord::cmp if we don't handhold it
            layer += 1;
            self.ify_r(med_idx + 1, b, layer);
            b = med_idx;
        }
    }

    #[cfg(test)]
    fn check_layer(&self, a: usize, b: usize, layer: usize) -> bool {
        if b > self.points.len() || a > b {
            return false
        } if a == b {
            return true
        }
        let mid_idx = (a + b)/2;
        let m = &self.points[mid_idx].0;
        for (e, _) in self.points.get(a..mid_idx).unwrap_or(&[]) {
            if KdPoint::cmp(e, m, layer) == Ordering::Greater {
                return false;
            }
        }
        for (e, _) in self.points.get(mid_idx+1..b).unwrap_or(&[]) {
            if KdPoint::cmp(e, m, layer) == Ordering::Less {
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
    pub(crate) fn k_closest_naive<'a>(&'a self, point: &R::Point, k: usize) -> Vec<(&'a R::Point, &'a V)> {
        let mut res = MmHeap::new();
        let cmp_fn = &|&(a, _): &_, &(b, _): &_|point.sqdist(a).cmp(&point.sqdist(b));
        (&self).into_iter().for_each(&mut |(p, v)|{
            if res.len() < k {
                res.push_by((p, v), cmp_fn)
            } else {
                res.pushpop_max_by((p, v), cmp_fn);
            }
        });
        res.into()
    }
}

pub struct Iter<'a, P: KdPoint, V> {
    buf: &'a [(P, V)],
    idx: usize
}

impl<'a, P: KdPoint, V> Iterator for Iter<'a, P, V> {
    type Item = (&'a P, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        self.buf.get(self.idx).inspect(|_|self.idx += 1).map(|(a,b)|(a,b))
    }
}

pub struct IterMut<'a, P: KdPoint, V> {
    tail: &'a mut [(P, V)],
}

impl<'a, P, V> Iterator for IterMut<'a, P, V>
where P: KdPoint {
    type Item = (&'a P, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        let Some(([(a, b)], tail)) = mem::take(&mut self.tail).split_at_mut_checked(1) else { return None };
        self.tail = tail;
        Some((a, b))
    }
}

impl<'a, R: KdRegion, V> IntoIterator for &'a KdTree<R, V> {
	type Item = (&'a R::Point, &'a V);
	type IntoIter = Iter<'a, R::Point, V>;
	fn into_iter(self) -> Self::IntoIter {
		Iter{buf: &self.points, idx: 0}
	}
}

impl<'a, R: KdRegion, V> IntoIterator for &'a mut KdTree<R, V> {
    type Item = (&'a R::Point, &'a mut V);
    type IntoIter = IterMut<'a, R::Point, V>;
    fn into_iter(self) -> Self::IntoIter {
        IterMut{tail: &mut self.points}
    }
}

impl<R: KdRegion, V> IntoIterator for KdTree<R, V> {
    type Item = (R::Point, V);
    type IntoIter = <Vec<(R::Point, V)> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.points.into_vec().into_iter()
    }
}

impl<R: KdRegion, V, const N: usize> From<[(R::Point, V); N]> for KdTree<R, V> {
    fn from(ents: [(R::Point, V); N]) -> Self {
        let bounds = get_bounds((&ents).into_iter().map(|(p,_)|p));
        let points = ents.into_iter().collect();
        let mut res = Self{points, bounds};
        res.ify();
        res
    }
}

impl<R: KdRegion, V> FromIterator<(R::Point, V)> for KdTree<R, V> {
    fn from_iter<T: IntoIterator<Item = (R::Point, V)>>(iter: T) -> Self {
        let points: Box<[_]> = iter.into_iter().collect();
        let bounds = get_bounds((&points).into_iter().map(|(p,_)|p));
        let mut res = Self{points, bounds};
        res.ify();
        res
    }
}

impl<R: KdRegion, V> Index<&R::Point> for KdTree<R, V> {
    type Output = V;
    fn index(&self, point: &R::Point) -> &V {
        self.get(point).unwrap()
    }
}

impl<R: KdRegion, V> IndexMut<&R::Point> for KdTree<R, V> {
    fn index_mut(&mut self, point: &R::Point) -> &mut Self::Output {
        self.get_mut(point).unwrap()
    }
}


//! Very Generic Data Structures
//!
//! Crater provides KD Trees ([`kdtree::KdTree`]), minmax heaps ([`mmheap::MmHeap`]), and intrusive Fibonacci heaps ([`fheap::FibHeap`]) with as much flexibility as possible.
//! For KD Trees, no restrictions are placed on the data besides that points
//! have well defined distance to regions/other points, and regions can be expanded/split.
//! Similarly, minmax heaps accept any comparison function.
//!
//! To get started quickly, look at [`cuboid::CuPoint`] and [`cuboid::CuRegion`] and the tests in [`cuboid::tests::pointcloud`].
//! This will provide an overview of how to use KD trees with the prebuilt Cartesian points/regions.
//! To associate custom data to points, create structs wrapping [`cuboid::CuPoint`] and [`cuboid::CuRegion`] which delegate to their [`KdPoint`]
//! and [`KdRegion`] implementations.  To implement [`KdPoint`] and [`KdRegion`] for arbitrary custom types,
//! continue reading their trait documentation here.
#![feature(split_at_checked, allocator_api, non_null_convenience, slice_ptr_get)]

pub mod mmheap;
pub mod cuboid;
pub mod kdtree;
pub mod fheap;

use std::cmp::Ordering;

use num_traits::Zero;


/// A point in the tree.  Types implementing this can contain arbitrary data,
/// to be accessed when the tree is queried by a point + distance, point + count,
/// region, etc.
/// NB: If `T: KdPoint`, `<T as Eq>::eq` can be implemented by comparing `T::sqdist` to `T::Distance::zero()` 
pub trait KdPoint: Sized + Eq {
    /// The type used by sqdist and related KdRegion functions to represent (squared) distance.
    /// Must be totally ordered, where greater distances mean points are farther apart.
    /// Also must be Clone and have a meaningful minimum value (0)
    type Distance: Ord + Clone + Zero;
    /// The squared distance is more computationally convenient than the proper distance
    /// in many cases.  The distance function only has to be topologically consistent
    /// and totally ordered.  See KdRegion::min_sqdist for more info.  The return value should be >= Distance::zero().
    fn sqdist(&self, other: &Self) -> Self::Distance;
    /// Compare two points in some layer of the tree.  This generalizes splitting
    /// different layers of the tree in different dimensions and is tied to KdRegion::split.
    /// Traditionally, points in a KD tree are compared by their 1st coordinate in layer 0,
    /// their 2nd in layer 1, ..., onto their Kth in layer k - 1, and then in layer K it wraps
    /// around to 0 and repeats.  So for a 3D KD tree, layer 0 would compare x coordinates
    /// with the root having the median x coordinate, all points in the first subtree having
    /// x coordinate <=, and all points in the second subtree having x coordinate >=.
    /// This method allows for more general strategies, but pushes some of the burden of
    /// layer-dependent behavior onto the point type.  It's still possible to just implement
    /// a by-coordinate cmp function of course, and this is what the prebuilt CuPoint does.
    fn cmp(&self, other: &Self, layer: usize) -> Ordering;
}

/// A region in the tree, or in space in general.  A type implementing this will be tightly
/// coupled to some type implementing [`KdPoint`].  For example, the prebuilt [`cuboid::CuPoint`] struct
/// represents a point in Cartesian space with Euclidean distance, and the prebuilt [`cuboid::CuRegion`]
/// struct represents cuboid regions of space (rectangles in 2D, rectangular prisms in 3D, etc).
/// Regions often represent infinitely many points in space (how many points are in your
/// typical rectangle?).  Regions should be able to represent a single point, but the ability
/// to represent an empty region isn't necessary.
pub trait KdRegion: Sized + Clone {
    type Point: KdPoint;
    /// Given a point `p` in this region `A` and a layer `l`, split `A` into two subregions `B` and `C` so that:
    /// - Any point `q` in `A` is in `B` or `C`, that is, if `A.min_sqdist(q) == 0`, then either `B.min_sqdist(q) == 0`
    ///   or `C.min_sqdist(q) == 0` (possibly both).  Note that strictly speaking this only *needs* to be
    ///   true over all points in the tree, not all points in space, but in practice it is almost always
    ///   much easier to make this true for all points in space, not least because this trait and
    ///   method have no reference to any particular KD tree.
    /// - Any point `q` in `B` is `<= p` in layer `l`, using [`KdPoint::cmp`], and any point `q` in `C` is `>= p`
    /// - Any point not in `A` should hopefully be farther from `B` than from `C` (or visa versa) in at least
    ///   some layers (note that distance doesn't depend on layer, but split direction does).
    ///   That is, if `A.min_dist(q) == d where d > 0`, then `B.min_sqdist(q), C.min_sqdist(q) >= d` (are both at least `d`),
    ///   and ideally one should be significantly more in at least some layers.
    /// Note how the basic instance of regions, cuboid regions, obviously obey all these properties
    fn split(&self, point: &Self::Point, layer: usize) -> (Self, Self);
    /// Given a region and a point possibly not in the region, extend the region to include the point if
    /// necessary.  The concrete requirements this places on the implementation are that
    /// `self.min_sqdist(q)` can only decrease or remain the same for any fixed `q`, and in particular `self.extend(q)`
    /// should cause `self.min_sqdist(q)` to be `0` if it wasn't already
    fn extend(&mut self, point: &Self::Point);
    /// Create a region consisting of a single point.  For cuboid regions for example, this is represented as
    /// a cuboid whose inclusive "start" and "end" points are both the same.  Types implementing this trait
    /// should be able to represent single points fairly well, but because of the conservative nature of
    /// everything, it is acceptable to fudge it by having a very small region containing the point.
    /// It's not necessary for types to be able to represent an empty region well or even at all.
    fn single_point(point: &Self::Point) -> Self;
    /// Return the minimal squared distance any point in this region could have to a given point.
    /// The return value must be `<= KdPoint::sqdist` between the given point and any point within this region.
    /// It's safe to return a smaller value, or even always return `Distance::zero()`, but this degrades performance because
    /// we can't prune subtrees from the search.
    /// If `B` is a subregion of `A` and `p` is a point not in `A`, then `B.min_sqdist(p) >= A.min_sqdist(p)`
    fn min_sqdist(&self, point: &Self::Point) -> <Self::Point as KdPoint>::Distance;
    /// Return the maximal squared distance any point in this region could have to a given point, or `None` if infinite.
    /// The return value must be `>= KdPoint::sqdist` between the given point and any point within this region.
    /// `None` is considered infinitely far away.  It's safe to return a larger value, or even always return `None`,
    /// but this may degrade performace for some queries that cull based on minimal distance.
    /// Currently, this only happens for [`kdtree::KdTree::k_closest`] where [`kdtree::QueryOptions::lower_bound`] is [`kdtree::QueryBound::SqDist`].
    /// If `B` is a subregion of `A` and `p` is a point not in `A`, then `B.max_sqdist(p) <= A.max_sqdist(p)`.
    fn max_sqdist(&self, point: &Self::Point) -> Option<<Self::Point as KdPoint>::Distance>;
    /// Return true if this region and another region might overlap, or false if they are definitely disjoint.
    /// Conservative implementors can always return true.
    /// Currently only used by [`kdtree::KdTree::k_closest`] if [`kdtree::QueryOptions::outer_bound`] is [`kdtree::QueryBound::Region`].
    fn might_overlap(&self, other: &Self) -> bool;
    /// Return true if this region is DEFINITELY a superset of another region, or false if it is not.
    /// `A` may be a superset of `B` even if `B` is internally tangent to `A` or `B` is `A`.
    /// May return false even if self is a superset of other, if it would be expensive or difficult to compute correctly.
    /// Currently only used by [`kdtree::KdTree::k_closest`] if [`kdtree::QueryOptions::inner_bound`] is [`kdtree::QueryBound::Region`]
    fn is_superset(&self, other: &Self) -> bool;
}


/// Tree traversal control flow, similar to Rust's builtin [`std::ops::ControlFlow`] enum.
pub enum WalkDecision {
    Continue,
    SkipChildren,
    Stop
}


/// Get the bounding box of a set of points.  For [`cuboid::CuRegion`] and [`cuboid::CuPoint`], this will be an
/// AABB (Axis Aligned Bounding Box).  For general [`KdRegion`], the bounds are not required to be tight
pub fn get_bounds<'a, R: KdRegion>(points: impl IntoIterator<Item = &'a R::Point>) -> Option<R> where R: 'a {
    let mut it = points.into_iter();
    let mut res = R::single_point(it.next()?);
    it.for_each(|p|res.extend(p));
    Some(res)
}


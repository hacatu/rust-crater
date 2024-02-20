//! Very Generic Data Structures
//!
//! Rust-Crater provides KD Trees and minmax heaps with as much flexibility as possible.
//! For KD Trees, no restrictions are placed on the data besides that points
//! have well defined distance to regions/other points, and regions can be expanded/split.
//! Similarly, minmax heaps accept any comparison function.
//!
//! To get started quickly, look at CuPoint and CuRegion and the tests in cuboid::pointcloud.
//! This will provide an overview of how to use KD trees with the prebuilt Cartesian points/regions.
//! To associate custom data to points, create structs wrapping CuPoint and CuRegion which delegate to their KdPoint
//! and KdRegion implementations.  To implement KdPoint and KdRegion for arbitrary custom types,
//! continue reading their trait documentation here.

pub mod mmheap;
pub mod cuboid;
pub mod kdtree;

use std::cmp::Ordering;


/// A point in the tree.  Types implementing this can contain arbitrary data,
/// to be accessed when the tree is queried by a point + distance, point + count,
/// region, etc
pub trait KdPoint: Sized {
    type Distance: Ord;
    /// The squared distance is more computationally convenient than the proper distance
    /// in many cases.  The distance function only has to be topologically consistent
    /// and totally ordered.  See kdRegion::min_sqdist for more info
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
/// coupled to some type implementing KdPoint.  For example, the prebuilt CuPoint struct
/// represents a point in Cartesian space with Euclidean distance, and the prebuilt CuRegion
/// struct represents cuboid regions of space (rectangles in 2D, rectangular prisms in 3D, etc).
/// Regions often represent infinitely many points in space (how many points are in your
/// typical rectangle?).  Regions should be able to represent a single point, but the ability
/// to represent an empty region isn't necessary.
pub trait KdRegion: Sized {
    type Point: KdPoint;
    /// Given a point p in this region A and a layer l, split A into two subregions B and C so that:
    /// - Any point q in A is in B or C, that is, if A.min_sqdist(q) = 0, then either B.min_sqdist(q) = 0
    ///   or C.min_sqdist(q) = 0 (possibly both).  Note that strictly speaking this only *needs* to be
    ///   true over all points in the tree, not all points in space, but in practice it is almost always
    ///   much easier to make this true for all points in space, not least because this trait and
    ///   method have no reference to any particular KD tree.
    /// - Any point q in B is <= than p in layer l, using KdPoint::cmp, and any point in C is >=
    /// - Any point not in A should hopefully be farther from B than from C (or visa versa) in at least
    ///   some layers (note that distance doesn't depend on layer, but split direction does).
    ///   That is, if A.min_dist(q) = d > 0, then B.min_sqdist(q), C.min_sqdist(q) >= d (are both at least d),
    ///   and ideally one should be significantly more in at least some layers.
    /// Note how the basic instance of regions, cuboid regions, obviously obey all these properties
    fn split(&self, point: &Self::Point, layer: usize) -> (Self, Self);
    /// Given a region and a point possibly not in the region, extend the region to include the point if
    /// necessary.  The concrete requirements this places on the implementation are that
    /// self.min_sqdist(q) can only decrease or remain the same for any fixed q, and in particular self.extend(q)
    /// should cause self.min_sqdist(q) to be 0 if it wasn't already
    fn extend(&mut self, point: &Self::Point);
    /// Create a region consisting of a single point.  For cuboid regions for example, this is represented as
    /// a cuboid whose inclusive "start" and "end" points are both the same.  Types implementing this trait
    /// should be able to represent single points fairly well, but because of the conservative nature of
    /// everything, it is acceptable to fudge it by having a very small region containing the point.
    /// It's not necessary for types to be able to represent an empty region well or even at all.
    fn single_point(point: &Self::Point) -> Self;
    /// Return the minimal squared distance any point in this region could have to a given point.
    /// The return value must be <= KdPoint::sqdist between the given point and any point within this region.
    /// It's safe to return a smaller value, or even always return 0, but this degrades performance because
    /// we can't prune subtrees from the search.
    /// If B is a subregion of A and p is a point not in A, then B.min_sqdist(p) >= A.min_sqdist(p)
    fn min_sqdist(&self, point: &Self::Point) -> <Self::Point as KdPoint>::Distance;
}


/// Tree traversal control flow, similar to Rust's builtin ControlFlow enum.
pub enum WalkDecision {
    Continue,
    SkipChildren,
    Stop
}


/// Get the bounding box of a set of points.  For CuRegion and CuPoint, this will be an
/// AABB (Axis Aligned Bounding Box).  For general KdRegion, the bounds are not required to be tight
pub fn get_bounds<'a, R: KdRegion>(points: impl IntoIterator<Item = &'a R::Point>) -> Option<R> where R: 'a {
    let mut it = points.into_iter();
    let mut res = R::single_point(it.next()?);
    it.for_each(|p|res.extend(p));
    Some(res)
}


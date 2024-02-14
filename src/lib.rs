pub mod mmheap;
pub mod cuboid;
pub mod kdree;

use std::cmp::Ordering;



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



pub enum WalkDecision {
    Continue,
    SkipChildren,
    Stop
}



pub fn get_bounds<'a, R: KdRegion>(points: impl IntoIterator<Item = &'a R::Point>) -> Option<R> where R: 'a {
    let mut it = points.into_iter();
    let mut res = R::single_point(it.next()?);
    it.for_each(|p|res.extend(p));
    Some(res)
}


# Rust-Crater

[![GitHub CI](https://github.com/hacatu/rust-crater/actions/workflows/cov_and_docs.yml/badge.svg)](https://github.com/hacatu/rust-crater/actions/workflows/cov_and_docs.yml)
[![Documentation](https://img.shields.io/badge/-documentation-gray)](https://hacatu.github.io/rust-crater/doc/crater)
[![Coverage](https://hacatu.github.io/rust-crater/cov/badges/plastic.svg)](https://hacatu.github.io/rust-crater/cov)

This is a data structures library, named after my [C data structures library](https://github.com/hacatu/Crater).

Rust has a lot more builtin data structures compared to C, so much of the stuff is unnecessary.
Vectors, hashtables, binary heaps, and argument parsing are already provided.

There are even some very rare vector functions like `select_nth_unstable`, nearly identical to my
`vec_ith` + `vec_partition_with_median`: both of these provide a way to find the nth element based on some
comparison function in an unsorted vector in linear time, and then partition the vector so the nth element
is at the nth index, every element preceding it is `<=`, and every element following it is `>=`.
The difference is my `vec_partition_with_median` is a three way partitioning function, meaning it separates
the vector into `<`, `=`, and `>` regions, but this is actually overkill for KD Trees, which were the
primary focus here.

Currently, this library implements VERY generic KD Trees and a minmax heap.

There are some good KD Tree libraries already available for rust, but none of them are sufficiently generic
for my needs.  For simple 2D/3D KD Trees with ints/floats, other libraries currently offer more
functionality.  For KD Trees that don't live in Cartesian space or have exotic scalar types, this library
is likely a good choice.

## Features

KD Trees can be made with an arbitrary topology (2D, 3D+, Cartesian, embedded on the surface of a sphere/torus/etc).

This is facillitated by having several traits associated with the `KdTree` struct:

- `KdPoint`: Represents a point in the tree.  Types implementing this trait only need to have two methods
  - `KdPoint::cmp`: Given a layer in the tree, compare two points in that layer.  For cartesian points,
    this is as simple as getting the `layer%dim` coordinate, but if the points are embedded in the surface of
	a sphere for example it could be more natural to compare based on angle if `layer` is odd and based on
	longitude if `layer` is even.
  - `KdPoint::sqdist`: Given two points, compute the squared distance between them.  `KdPoint` does not have
    any restrictions on the type of the coordinates of the points it represents; it doesn't even assume
	points are represented with coordinates.  Instead, the only associated type is `KdPoint::Distance`,
	which is returned by `KdPoint::sqdist` and only must be `Ord`.

- `KdRegion`: Represents the bounds of the entire tree or some subtree.  Types implementing this trait only
  need to have four methods.  They should be able to represent a single point, but it is not necessary to be
  able to represent an empty region (it's wrapped in `Option`) and it's ok if the region overestimates its size
  as long as it does not underestimate its (eg it can return a lower number than the truth for `min_sqdist` but not
  a higher number)
  - `KdRegion::split`: Given a region, a point in the region, and a layer in the tree, split the region into
    the two subregions defined by that point in that layer.  For example, if the KD tree is 2D Cartesian,
	even layers might split it horizontally and odd layers vertically.  In general terms, this might involve
	chopping the region with a line/plane/hyperplane, or just splitting a convex set into two convex parts.
	The outer bounds on the tree are stored explicitly, but the bounds of every subtree are calculated laxly
	during traversal (ie they will generally be overestimates).
  - `KdRegion::single_point`: Create a region from a single point
  - `KdRegion::extend`: Extend a region so that in includes an additional point.  Frequently, regions will be
    AABBs so this is insanely simple to implement, but it could be much more complicated in general.
  - `KdRegion::min_sqdist`: Return a number <= the minimum distance between a given point and this region.
    In particular, points inside the region, such as those added by `single_point` and `extend`, MUST return `0` (or
	something <= every other return value).  Generally, regions are convex and so linear combinations of interior points
	if applicable are in the region.  Points not in the region should ideally return > 0 so that the search can be
	maximally pruned, but not > their actual distance or of course the search could be incorrect.

There are also default implementations of these provided for Cartesian KD trees in arbitrary dimension with
(almost) any coordinate type (must be `Ord + NumRef + Clone`): `CuPoint` and `CuRegion`.

## External Crates:
  `num-traits`: This crate is certainly no Numerical Prelude (Haskall), but it's an aboslute godsend for making
  mathematically generic traits without having to put 200+ type bounds relating to overloading numeric operators
  for different permutations of reference types.  Basically impossible to overload these opertors without it.
  `rand`: It's extremely weird that C++ has a builtin rand library but Rust doesn't. Rust typically makes
  much better decisions with what should and shouldn't enter the language.  That said, this random number
  crate is INSANELY powerful and good, even moreso than Quickcheck (Haskall).  Overloading the `Uniform`
  random distribution to generate `CuPoint<T, N>` generically took like 30 seconds.


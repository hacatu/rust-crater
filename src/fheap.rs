//! A fibonacci heap is a type of data structure optimized for use as a priority queue.
//! It is commonly useful in pathfinding algorithms like Dijkstra's and A*.
//! 
//! Below is a very in depth comparison and analysis of various types of heaps,
//! skip to [`FibHeap`] if you only want to know how to use this library.
//! Reading the tests is also a great way to find examples.  Building code as tests will
//! help debug fibheap code in particular because additional checks are performed under `cfg(test)`
//! that will `abort` if the heap breaks (ie because a duplicate node was added, a node was removed
//! that wasn't in the heap, or there is a bug in the library).
//! 
//! Depending on your needs, [`crate::mmheap::MmHeap`] may be more useful to you, or
//! Rust's builtin [`std::collections::BinaryHeap`].  Both of these are binary heaps,
//! but the former offers fast access to both the minimum and the maximum.
//! 
//! # Comparison of Heaps
//! ## Asymptotic Time
//!|            | Binary | Pairing | Fibonacci | Bucket |
//!|------------|--------|---------|-----------|--------|
//!|find-min    | O(1)   | O(1)    | O(1)      | O(1)   |
//!|extract-min | O(logn)| O(logn) | O(logn)   | O(1)   |
//!|insert      | O(logn)| O(1)    | O(1)      | O(1)   |
//!|decrease-key| O(logn)| O(logn) | O(1)      | O(1)   |
//!|increase-key| O(logn)| O(logn) | O(logn)   | O(1)   |
//!|meld        | O(n)   | O(1)    | O(1)      | O(n)   |
//! Not all of these operations are directly supported.  In Fibonacci heaps, `increase-key`
//! must be implemented by removing and re-inserting an element.  In binary heaps,
//! decrease/increase key can be difficult to implement because the elements exist in the heap's
//! internal buffer and there isn't a direct way to get a reference/index to them.
//! Also in binary heaps, meld is not directly supported and must be implemented by extending
//! the internal buffer of one heap by that of another and re-heapifying.
//! Bucket queues are extremely good when relevant, but they only support a fixed number of
//! priorities, literally having a bucket for each priority, so they are not always suitable.
//! 
//! ## Linked vs Unlinked Data Structures
//! Asymptotic time complexity is not the whole story.  Fibonacci heaps and pairing heaps are linked data
//! structures, so links to adjacent nodes are stored as pointers, whereas binary heaps
//! have such a simple, rigid structure that they can be stored directly in an array
//! (eg the children of a node at index i will be at indices 2i + 1 and 2i + 2).
//! Linked data structures can still be stored in an array, but the offsets between nodes
//! are not consistent and so generally pointers are used, which can introduce overhead in several
//! ways.  Linked data structures often underperform relative to "rigid" data structures
//! even though their asymptotic complexity is better.  And even among linked data structures,
//! Fibonacci heaps are considered complex and may perform worse than pairing heaps.
//! 
//! ## Amortized Running Time
//! The runtimes of Fibonacci heaps are amortized using potential, which means if the
//! number of other calls is much much greater than the number of extract-min calls,
//! the extract-min calls may take much longer because they are picking up more work.
//! This doesn't happen in most use cases since generally a large part of the heap will be
//! drained in algorithms like heapsort, A*, etc.  When it's not the case that a large part
//! of the heap will be drained, a minmax heap can often be a good choice since it allows
//! dropping entries that will never be used efficiently.  In particular, if the number of
//! entries that will be extracted from the heap is constant and small, a minmax heap will
//! probably be best.  If the number of entries that will be extracted is proportional to
//! the number inserted and/or large, a fibonacci heap may be the best.
//! 
//! ## Explanation of Algorithms
//! There are two operations (extract-min and decrease-key) that are central to Fibonacci heaps,
//! and all other operations are trivial.
//! A Fibonacci heap consists of a collection of subheaps, each of which is a tree where
//! nodes can have any number of children.  This is implemented by giving each node 4 pointers:
//! its parent, its first child, its next sibling, and its previous sibling.  For the subheap roots,
//! their parent is null and their next/previous siblings are instead other roots.
//! Each node also stores its degree (number of chilren) and whether or not it has had a child removed.
//! 
//! The heap itself stores a pointer to the minimal root, and the number of elements in the heap.
//! 
//! This structure is restricted by a few invariants.  First, each node is less than or equal to
//! all of its children according to the heap comparison function.  Second, the total size of a
//! subheap whose root has k children is at least F_(k+2), where F_k is the kth Fibonacci number,
//! with F_1 = F_2 = 1, F_3 = 2, F_4 = 3, etc.
//! 
//! When extract-min is called, first we remove the minimal root.
//! Then we iterate over the roots of all other subheaps, plus the children of the removed root,
//! placing them into an array with O(logn) elements.
//! Each such node goes into the index in the array corresponding to its degree.  If there is already a subheap there,
//! we instead remove that subheap, merge it with the current one, and keep going.  When we merge the subheaps,
//! the one with the larger root is added as a child of the one with the smaller root, so the degree of the latter goes
//! up and then we try to put it in the next corresponding index in the array, repeating until we reach a degree that
//! isn't in the array yet.
//! 
//! At the end, this gives us an array of subheaps with distinct degrees.
//! We re-link these into a double circularly linked list, and update the minimal root pointer to point to the minimal one.
//! 
//! If all subheaps obey the `size >= F_(degree + 2)` invariant before `extract-min`, they will also obey it after.
//! 
//! If there are two or fewer nodes in the heap before we extract-min, we can optimize by not merging subheaps and just
//! setting the new minimal root to either the one remaining node or null.
//! 
//! When decrease-key is called, if the key is still greater than or equal to the parent's key, or the node is
//! a root node and the key is still greater than or equal to the minimal root, or the root node is the minimal root,
//! there is nothing to be done.  If the node is a root node but not the minimal root and the key becomes
//! less than the minimal heap, update the minimal heap.
//! 
//! Finally, if the key is now less than the parent's key, remove the node from its sibling list and as a child of its parent.
//! Add it as a new subeap.  If the parent was not marked as already having a child removed, mark it as such
//! and then remove it as well, repeating for any remaining parents.  However, root nodes need not be marked as such.
//! 
//! This marking is needed to ensure that the `size >= F_(degree + 2)` invariant does not break
//! 
//! Insert and meld just require adding a new subheap with one element, and stitching together the subheap root linked lists,
//! respectively.
//! 
//! Find-min just requires inspecting the min_root pointer.
//! 
//! ## Further Reading
//! [https://en.wikipedia.org/wiki/Fibonacci_heap]
//! [https://youtu.be/6JxvKfSV9Ns]
//! 
//! A very popular priority queue crate.  This crate implements priority queues using a hash table to allow key-indexed node
//! access, and builds the priority queue on top of its internal hash table.  It has a good api and is a good choice most of
//! the time.  Crater has some advantages over this crate: our fib heap allows building priority queues with any backing
//! store, eg KD trees, vectors, just putting each node in `Box`, etc; and references to nodes have generous lifetimes,
//! so storing references instead of keys is possible and makes lookup as fast as possible.  However, unless you need these
//! features, this crate is a better choice than Crater.
//! [https://crates.io/crates/priority-queue]

use std::{cell::UnsafeCell, cmp::Ordering, marker::PhantomData, ptr};

/// The intrusive struct itself that should be embedded in any types that
/// implement [`Node`].  See [`FibHeap`] for more information.
#[derive(Debug)]
pub struct RawNode<'a, T: Node<'a> + ?Sized> {
	prev: UnsafeCell<Option<&'a T>>,
	next: UnsafeCell<Option<&'a T>>,
	first_child: UnsafeCell<Option<&'a T>>,
	parent: UnsafeCell<Option<&'a T>>,
	has_split: UnsafeCell<bool>,
	degree: UnsafeCell<u8>
}

/// Any struct can be used as the fib heap element simply by embedding a [`RawNode`]
/// in it (or wrapping it in a struct containing a raw node) and implementing this trait.
pub trait Node<'a> {
	/// Comparison function for nodes, can just wrap Ord impl if present
	fn cmp(&'a self, other: &'a Self) -> Ordering;
	/// Get a reference to the embedded raw node.  This is used internally to traverse and bookkeep the heap.
	/// Accessing this is not thread safe as-is.  Implementors can place the raw node under a lock,
	/// but it's better to lock the entire [`FibHeap`] to avoid race conditions.
	fn get_raw(&'a self) -> &RawNode<'a, Self>;
}

struct SiblingIter<'a, T: Node<'a> + ?Sized> {
	start: Option<&'a T>,
	iter: &'a T
}

impl<'a, T: Node<'a> + ?Sized> Iterator for SiblingIter<'a, T> {
	type Item = &'a T;
	fn next(&mut self) -> Option<Self::Item> {
		self.start?;
		let res = self.iter;
		self.iter = (*unsafe { res.get_raw().next.get().as_ref() }.unwrap()).unwrap();
		if ptr::eq(self.iter, self.start.unwrap()) { self.start = None }
		Some(res)
	}
}

struct UnlinkingSiblingIter<'a, T: Node<'a> + ?Sized> {
	start: Option<&'a T>,
	iter: &'a T
}

impl<'a, T: Node<'a> + ?Sized> Iterator for UnlinkingSiblingIter<'a, T> {
	type Item = &'a T;
	fn next(&mut self) -> Option<Self::Item> {
		self.start?;
		let res = self.iter;
		let links = res.get_raw();
		self.iter = (*unsafe { links.next.get().as_ref() }.unwrap()).unwrap();
		if ptr::eq(self.iter, self.start.unwrap()) { self.start = None }
		unsafe {
			*links.next.get() = None;
			*links.prev.get() = None;
			*links.parent.get() = None;
		}
		Some(res)
	}
}

#[cfg(test)]
enum FibHeapError<'a, T: Node<'a> + ?Sized> {
	BrokenPrevLink(&'a T),
	LessThanParent(&'a T),
	BrokenParentLink(&'a T),
	WrongDegree(&'a T),
	TooSmall(&'a T),
	WrongCount
}

#[cfg(test)]
impl<'a, T: Node<'a> + ?Sized> PartialEq for FibHeapError<'a, T> {
	fn eq(&self, other: &Self) -> bool {
		use FibHeapError::*;
		match (self, other) {
			(BrokenPrevLink(a), BrokenPrevLink(b))
			| (LessThanParent(a), LessThanParent(b))
			| (BrokenParentLink(a), BrokenParentLink(b))
			| (WrongDegree(a), WrongDegree(b))
			| (TooSmall(a), TooSmall(b)) => ptr::eq(a, b),
			(WrongCount, WrongCount) => true,
			_ => false
		}
	}
}

#[cfg(test)]
impl<'a, T: Node<'a> + ?Sized> std::fmt::Debug for FibHeapError<'a, T> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		use FibHeapError::*;
		match self {
			BrokenPrevLink(a) => write!(f, "BrokenPrevLink({})", a as *const _ as usize),
			LessThanParent(a) => write!(f, "LessThanParent({})", a as *const _ as usize),
			BrokenParentLink(a) => write!(f, "BrokenParentLink({})", a as *const _ as usize),
			WrongDegree(a) => write!(f, "WrongDegree({})", a as *const _ as usize),
			TooSmall(a) => write!(f, "TooSmall({})", a as *const _ as usize),
			WrongCount => write!(f, "WrongCount")
		}
	}
}

impl<'a, T: Node<'a> + ?Sized> Default for RawNode<'a, T> {
	fn default() -> Self {
		Self{prev: None.into(), next: None.into(), first_child: None.into(), parent: None.into(), has_split: false.into(), degree: 0.into()}
	}
}

#[cfg(test)]
fn iter_siblings<'a, T>(node: &'a T) -> SiblingIter<'a, T>
where T: Node<'a> + ?Sized {
	SiblingIter{start: Some(node), iter: node}
}

fn unlinking_siblings<'a, T>(node: &'a T) -> UnlinkingSiblingIter<'a, T>
where T: Node<'a> + ?Sized {
	UnlinkingSiblingIter{start: Some(node), iter: node}
}

fn unlinking_children<'a, T>(node: &'a T) -> UnlinkingSiblingIter<'a, T>
where T: Node<'a> + ?Sized {
	let links = node.get_raw();
	(*unsafe { links.first_child.get().as_ref() }.unwrap()).inspect(|_| unsafe {
		*links.first_child.get() = None;
		*links.degree.get() = 0
	}).map_or(UnlinkingSiblingIter{start: None, iter: node}, unlinking_siblings)
}

unsafe fn remove_root<'a, T>(node: &'a T) -> Option<&'a T>
where T: Node<'a> + ?Sized {
	let links = node.get_raw();
	let next = (*unsafe { links.next.get().as_ref() }.unwrap()).unwrap();
	let prev = (*unsafe { links.prev.get().as_ref() }.unwrap()).unwrap();
	if ptr::eq(next, prev) {
		if ptr::eq(next, node) {
			None
		} else {
			let next_links = next.get_raw();
			unsafe {
				*next_links.next.get() = Some(next);
				*next_links.prev.get() = Some(next);
				*links.next.get() = Some(node);
				*links.prev.get() = Some(node);
			}
			Some(next)
		}
	} else {
		let next_links = next.get_raw();
		let prev_links = prev.get_raw();
		unsafe {
			*next_links.prev.get() = Some(prev);
			*prev_links.next.get() = Some(next);
			*links.next.get() = Some(node);
			*links.prev.get() = Some(node);
		}
		Some(next)
	}
}

unsafe fn merge_same_deg<'a, T>(node: &'a T, other: &'a T) -> &'a T
where T: Node<'a> + ?Sized {
	// first, find which of node / other has the minimal key, breaking ties with node
	let (res_ptr, other_ptr) = if node.cmp(other).is_le() {
		(node, other)
	} else { (other, node) };
	let res_links = res_ptr.get_raw();
	let other_links = other_ptr.get_raw();
	match *res_links.degree.get() {
		0 => unsafe { // the roots both have degree zero, so there are no children to fix up
			*other_links.next.get() = Some(other_ptr);
			*other_links.prev.get() = Some(other_ptr);
			*other_links.parent.get() = Some(res_ptr);
			*res_links.first_child.get() = Some(other_ptr);
		},
		1 => unsafe { // the roots both have degree one, so res has one existing child
			let first_child = (*res_links.first_child.get()).unwrap();
			*other_links.next.get() = Some(first_child);
			*other_links.prev.get() = Some(first_child);
			let first_child_links = first_child.get_raw();
			*first_child_links.next.get() = Some(other_ptr);
			*first_child_links.prev.get() = Some(other_ptr);
			*other_links.parent.get() = Some(res_ptr);
		},
		_ => unsafe { // the roots both have degree greater than one, so res has a distinct existing first and last child
			let first_child = (*res_links.first_child.get()).unwrap();
			*other_links.next.get() = Some(first_child);
			let first_child_links = first_child.get_raw();
			let last_child = (*first_child_links.prev.get()).unwrap();
			*other_links.prev.get() = Some(last_child);
			*last_child.get_raw().next.get() = Some(other_ptr);
			*first_child_links.prev.get() = Some(other_ptr);
			*other_links.parent.get() = Some(res_ptr);
		}
	}
	*res_links.degree.get() += 1;
	res_ptr
}
#[cfg(test)]
fn check<'a, T>(node: &'a T) -> Result<usize, FibHeapError<'a, T>>
where T: Node<'a> + ?Sized {
	use FibHeapError::*;
	let mut degree = 0;
	let mut first = None;
	let mut prev = None;
	let mut count = 1;
	let (mut fib_d1, mut fib_d2) = (1, 1);
	for child in (*unsafe { node.get_raw().first_child.get().as_ref().unwrap() }).into_iter().flat_map(iter_siblings) {
		degree += 1;
		(fib_d1, fib_d2) = (fib_d2, fib_d1 + fib_d2);
		let child_links = child.get_raw();
		if child.cmp(node) == Ordering::Less {
			return Err(LessThanParent(child))
		}
		if prev.is_none() {
			first = Some(child);
		} else if !unsafe { *child_links.prev.get() }.is_some_and(|p|ptr::eq(p, prev.unwrap())) {
			Err(BrokenPrevLink(child))?
		} else if !unsafe { *child_links.parent.get() }.is_some_and(|p|ptr::eq(p, node)) {
			return Err(BrokenParentLink(child))
		}
		prev = Some(child);
		count += check(child)?;
	}
	if let Some(first_node) = first {
		let child_links = first_node.get_raw();
		if !unsafe { *child_links.prev.get() }.is_some_and(|p|ptr::eq(p, prev.unwrap())) {
			return Err(BrokenPrevLink(first_node))
		}
		// we don't need to check "prev".next because that's the termination condition of `iter_siblings`
	}
	if degree != unsafe { *node.get_raw().degree.get() } {
		Err(WrongDegree(node))
	} else if count < fib_d2 {
		Err(TooSmall(node))
	} else { Ok(count) }
}

/// A fibonacci heap is a type of data structure optimized for use as a priority queue.
/// It is commonly useful in pathfinding algorithms like Dijkstra's and A*.
/// Fib heaps have the following amortized time complexities:
/// - find-min: O(1)
/// - extract-min: O(logn)
/// - insert: O(1), compare to O(logn) in a binary heap
/// - decrease-key: O(1), compare to O(logn) in a binary heap / pairing heap
/// - meld: O(1), compare to O(n) in a binary heap
/// Extracting (or deleting) an arbitrary element is also O(logn).
/// Increase-key is not directly supported, but can be done by removing and reinserting.
/// Fib heaps will generally be a better choice than other heaps when:
/// - The number of elements extracted from the heap is proportional to the number inserted and/or large, not fixed and small
/// - The keys/priorities of elements need to be decreased frequently
/// - The keys/priorities of elements don't need to be increased frequently (also the comparison function doesn't need to be changed)
pub struct FibHeap<'a, T: Node<'a> + ?Sized, U> {
	min_root: Option<&'a T>,
	count: usize,
	container: PhantomData<&'a mut U>
}

impl<'a, T: Node<'a> + ?Sized + 'a, U> FibHeap<'a, T, U> {
	/// Create a new fibonacci heap whose backing storage will live at least as long as `_container`.
	/// Typically, if the nodes are stored in a `Vec` or a [`crate::kdtree::KdTree`] or so on,
	/// this backing container will have to remain effectively pinned for as long as the fibonacci
	/// heap exists, so that references to nodes remain valid.
	/// 
	/// Conceptually, only the lifespan of `_container` matters: by passing it in here and holding a `PhantomData`
	/// reference with the same lifespan, we force the shared reference to the container to outlive the
	/// fibonacci heap and prevent any mutable references to it from existing for that time.
	/// 
	/// Internally, [`RawNode`] uses [`UnsafeCell`] so that nodes can be mutated through shared reference.
	/// Otherwise, correctly mutating nodes is excessively complicated (requires using raw pointers throughout,
	/// and requires the backing storage to have some way to get a mutable subreference that doesn't overlap with
	/// any node so we can assume it is effectively pinned, most of the time this would be a mutable reference to
	/// an empty slice).
	pub fn new(_container: &'a U) -> Self {
		Self{min_root: None, count: 0, container: PhantomData}
	}
	/// Get the minimal element if it exists, returning a reference to it without removing it.
	/// It is safe to decrease the key of the result, but not to increase it
	/// if doing so could cause it to no longer be minimal (the key probably lives behind an [`UnsafeCell`])
	pub fn peek_min(&self) -> Option<&'a T> {
		self.min_root
	}
	/// Get and remove the minimal element if it exists, returning a reference to it.
	/// The key of the result may be freely modified
	pub fn pop_min(&mut self) -> Option<&'a T> {
		if self.count <= 1 {
			let res = self.min_root?;
			self.count = 0;
			self.min_root = None;
			#[cfg(test)]{
				assert!(self.check().is_ok())
			}
			return Some(res)
		}
		// Calculate the ceiling of the base 2 log of self.count, then multiply by the reciprocal of the base 2 log of the golden ratio
		let max_degree = ((((self.count - 1).ilog2() + 1) as f64)*1.4404200904125567).ceil() as usize;
		let mut roots = vec![None; max_degree + 1];
		let min_root = self.min_root.unwrap();
		let other_roots = unsafe { remove_root(min_root) };
		// iterate over all roots besides min_root (other_roots.flat_map(RawNode::unlinking_siblings)),
		// followed by all children of min_root.
		// unlinking_siblings/children are "destructive" and will automatically remove all sibling links.
		let iter = other_roots.into_iter().flat_map(unlinking_siblings)
			.chain(unlinking_children(min_root));
		for mut root_ptr in iter {
			loop { // repeatedly try to insert the root into the array of roots, merging it with the root with the same degree until it has unique degree
				let degree = unsafe { *root_ptr.get_raw().degree.get() } as usize;
				match roots[degree].take() {
					None => {
						roots[degree] = Some(root_ptr);
						break
					},
					Some(other_ptr) => root_ptr = unsafe { merge_same_deg(root_ptr, other_ptr) }
				}
			}
		}
		let mut iter = roots.into_iter().flatten();
		let [first_ptr, mut min_ptr, mut last_ptr] = [iter.next().unwrap(); 3];
		for root_ptr in iter {
			let last_links = last_ptr.get_raw();
			let curr_links = root_ptr.get_raw();
			unsafe {
				*last_links.next.get() = Some(root_ptr);
				*curr_links.prev.get() = Some(last_ptr);
			}
			if root_ptr.cmp(min_ptr).is_le() {
				min_ptr = root_ptr;
			}
			last_ptr = root_ptr;
		}
		if ptr::eq(first_ptr, last_ptr) {
			let first_links = first_ptr.get_raw();
			unsafe {
				*first_links.prev.get() = Some(first_ptr);
				*first_links.next.get() = Some(first_ptr);
			}
		} else { unsafe {
			*first_ptr.get_raw().prev.get() = Some(last_ptr);
			*last_ptr.get_raw().next.get() = Some(first_ptr);
		}}
		self.count -= 1;
		self.min_root = Some(min_ptr);
		#[cfg(test)]{
			assert_eq!(self.check(), Ok(()));
		}
		Some(min_root)
	}
	/// Add a node to the heap.  This is unsafe because the node must not already be in
	/// the heap, nor in a different heap.  It is not strictly required for all nodes
	/// to have the same backing store as long as the borrow checker confirms they outlive
	/// the heap, but most of the time all references will be to elements of the backing store.
	pub unsafe fn push(&mut self, ent: &'a T) {
		unsafe { self.reattach(ent) }
		self.count += 1;
		#[cfg(test)]{
			assert_eq!(self.check(), Ok(()))
		}
	}
	/// Called immediately AFTER a node's key is decreased, to ensure that the heap invariants
	/// are maintained.  This is unsafe because the node must be an element of the heap.
	pub unsafe fn decrease_key(&mut self, ent: &'a T) {
		if unsafe { *ent.get_raw().parent.get()}.is_some_and(|p|ent.cmp(p) == Ordering::Less) {
			self.separate_node(ent)
		}
	}
	/// Remove a node from the heap by reference.  Since the heap is intrusive and does not own its
	/// nodes, nothing is returned because the caller already has a reference to the removed node.
	/// This is unsafe because the node must be an element of the heap.
	pub unsafe fn remove(&mut self, node: &'a T) {
		self.separate_node(node);
		self.min_root = Some(node);
		self.pop_min();
	}
	unsafe fn reattach(&mut self, node: &'a T) {
		match self.count {
			0 => {
				self.min_root = Some(node);
				let links = node.get_raw();
				*links.next.get() = Some(node);
				*links.prev.get() = Some(node);
			},
			1 => {
				let root_links = self.min_root.unwrap().get_raw();
				*root_links.next.get() = Some(node);
				*root_links.prev.get() = Some(node);
				let node_links = node.get_raw();
				*node_links.next.get() = self.min_root;
				*node_links.prev.get() = self.min_root;
				if node.cmp(self.min_root.unwrap()).is_le() {
					self.min_root = Some(node)
				}
			},
			_ => {
				let root_links = self.min_root.unwrap().get_raw();
				let next_root = (*root_links.next.get()).unwrap();
				let next_links = next_root.get_raw();
				let node_links = node.get_raw();
				*node_links.prev.get() = self.min_root;
				*next_links.prev.get() = Some(node);
				*node_links.next.get() = Some(next_root);
				*root_links.next.get() = Some(node);
				if node.cmp(self.min_root.unwrap()).is_le() {
					self.min_root = Some(node)
				}
			}
		}
	}
	unsafe fn separate_node(&mut self, mut node: &'a T) {
		loop {
			let Some(parent) = *node.get_raw().parent.get() else { return };
			let next = remove_root(node);
			let parent_links = parent.get_raw();
			if (*parent_links.first_child.get()).is_some_and(|f|ptr::eq(node, f)) {
				*parent_links.first_child.get() = next;
			}
			*node.get_raw().parent.get() = None;
			self.reattach(node);
			*parent_links.degree.get() -= 1;
			if !*parent_links.has_split.get() {
				if (*parent_links.parent.get()).is_none() {
					*parent_links.has_split.get() = true;
				}
				break
			}
			*parent_links.has_split.get() = false;
			node = parent;
		}
		#[cfg(test)]{
			assert!(self.check().is_ok())
		}
	}
	#[cfg(test)]
	fn check(&self) -> Result<(), FibHeapError<'a, T>> {
		use FibHeapError::*;
		if (self.count == 0) != self.min_root.is_none() {
			return Err(WrongCount)
		} if self.count == 0 {
			return Ok(())
		}
		#[cfg(feature = "stress_tests")]{
			return Ok(())
		}
		let root = self.min_root.unwrap();
		let mut first = None;
		let mut prev: Option<&T> = None;
		let mut count = 0;
		for child in iter_siblings(root) {
			if child.cmp(root) == Ordering::Less {
				return Err(LessThanParent(child))
			}
			let child_links = child.get_raw();
			if prev.is_none() {
				first = Some(child);
			} else if !unsafe { *child_links.prev.get() }.is_some_and(|p|ptr::eq(p, prev.unwrap())) {
				Err(BrokenPrevLink(child))?
			} else if unsafe { *child_links.parent.get() }.is_some() {
				return Err(FibHeapError::BrokenParentLink(child))
			}
			prev = Some(child);
			count += check(child)?;
		}
		if let Some(child) = first {
			let child_links = child.get_raw();
			if !unsafe { *child_links.prev.get() }.is_some_and(|p|ptr::eq(p, prev.unwrap())) {
				return Err(FibHeapError::BrokenPrevLink(child))
			}
			// we don't need to check "prev".next because that's the termination condition of `iter_siblings`
		}
		if count != self.count {
			Err(FibHeapError::WrongCount)
		} else { Ok(()) }
	} 
}

#[cfg(test)]
mod tests {
    use std::{cell::UnsafeCell, cmp::Ordering, fs, ops::Deref};

    use num_traits::Euclid;

    use super::{FibHeap, Node, RawNode};

	struct GenNode<'a> {
		multiple: UnsafeCell<u64>,
		prime: UnsafeCell<u64>,
		_node: RawNode<'a, Self>
	}

	impl<'a> Node<'a> for GenNode<'a> {
		fn cmp(&'a self, other: &'a Self) -> Ordering {
			unsafe { (*self.multiple.get()).cmp(other.multiple.get().as_ref().unwrap()) }
		}
		fn get_raw(&'a self) -> &RawNode<Self> {
			&self._node
		}
	}

	#[test]
	fn prime_fheap() {
		let mut prime_sum = 0;
		let scapegoat = Box::new(());
		let mut ref_holder = Vec::new();
		let mut my_heap = FibHeap::new(&scapegoat);
		let ub = 100;
		for n in 2..ub {
			loop {
				if !my_heap.peek_min().is_some_and(|g: &GenNode|unsafe { *g.multiple.get() } < n ) {
					break
				}
				let node = my_heap.pop_min().unwrap();
				unsafe {
					*node.multiple.get() += *node.prime.get();
					if *node.multiple.get() < ub {
						my_heap.push(node)
					}
				}
			}
			if !my_heap.peek_min().is_some_and(|g|unsafe { *g.multiple.get() } == n) {
				prime_sum += n;
				if n*n < ub {
					let node = Box::new(GenNode{multiple: (n*n).into(), prime: n.into(), _node: Default::default()});
					ref_holder.push(node);
					unsafe {
						let last_ref = (ref_holder.last().unwrap().deref() as *const GenNode).as_ref().unwrap();
						my_heap.push(last_ref)
					}
				}
			}
		}
		eprintln!("Sum of primes < {} = {}", ub, prime_sum);
		assert_eq!(prime_sum, 1060);
	}

	struct GridNode<'a> {
		cost: u64,
		h: u64,
		min_dist: UnsafeCell<u64>,
		visited: UnsafeCell<bool>,
		_node: RawNode<'a, Self>
	}

	impl<'a> Node<'a> for GridNode<'a> {
		fn cmp(&'a self, other: &'a Self) -> Ordering {
			unsafe { *self.min_dist.get() + self.h }.cmp(&unsafe { *other.min_dist.get() + other.h })
			//unsafe { *self.min_dist.get() }.cmp(&unsafe { *other.min_dist.get() })
		}
		fn get_raw(&'a self) -> &RawNode<'a, Self> {
			&&self._node
		}
	}

	#[test]
	fn shortest_path() {
		let f = fs::read_to_string("resources/0083_matrix.txt").unwrap();
		let mut cols = None;
		let mut grid = Vec::new();
		for l in f.lines() {
			let a = grid.len();
			grid.extend(l.split(',').map(|t|GridNode{cost: t.parse().unwrap(), h: 0, min_dist: 0.into(), visited: false.into(), _node: Default::default()}));
			let c = grid.len() - a;
			cols.get_or_insert(c);
			assert_eq!(cols.unwrap(), c, "All rows must be the same length");
		}
		let cols = cols.unwrap_or_default();
		let rows = grid.len().checked_div(cols).unwrap_or_default();
		let min_cost = grid.iter().map(|n|n.cost).min();
		for (i, n) in grid.iter_mut().enumerate() {
			let (row, col) = i.div_rem_euclid(&cols);
			n.h = min_cost.unwrap()*(rows - row + cols - col - 1) as u64;
		}
		let mut frontier = FibHeap::new(&grid);
		let node = grid.first().unwrap();
		unsafe {
			*node.min_dist.get() = node.cost;
			frontier.push(node);
		}
		loop {
			let node = frontier.pop_min().unwrap();
			let i = unsafe { (node as *const GridNode<'_>).offset_from(grid.as_ptr()) } as usize;
			if i == grid.len() - 1 {
				break
			}
			let (row, col) = i.div_rem_euclid(&cols);
			unsafe { *node.visited.get() = true };
			for (dr, dc) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
				let r1 = (row as isize + dr).clamp(0, rows as isize - 1) as _;
				let c1 = (col as isize + dc).clamp(0, cols as isize - 1) as _;
				if (r1, c1) == (row, col) {
					continue
				}
				let n1 = &grid[r1*cols + c1];
				unsafe {
					if *n1.visited.get() {
						continue
					}
					let curr_dist = *n1.min_dist.get();
					let new_dist = *node.min_dist.get() + n1.cost;
					if curr_dist == 0 {
						*n1.min_dist.get() = new_dist;
						frontier.push(n1);
					} else if curr_dist > new_dist {
						*n1.min_dist.get() = new_dist;
						frontier.decrease_key(n1);
					}
				}
			}
		}
		let res = unsafe { *grid.last().unwrap().min_dist.get() };
		eprintln!("Min path sum: {res}");
		assert_eq!(res, 425185);
	}
}
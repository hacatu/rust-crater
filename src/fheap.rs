use std::{cell::UnsafeCell, cmp::Ordering, marker::PhantomData, ptr};

#[derive(Debug)]
pub struct RawNode<'a, T: Node<'a> + ?Sized> {
	prev: UnsafeCell<Option<&'a T>>,
	next: UnsafeCell<Option<&'a T>>,
	first_child: UnsafeCell<Option<&'a T>>,
	parent: UnsafeCell<Option<&'a T>>,
	has_split: UnsafeCell<bool>,
	degree: UnsafeCell<u8>
}

pub trait Node<'a> {
	fn cmp(&'a self, other: &'a Self) -> Ordering;
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

pub struct FibHeap<'a, T: Node<'a> + ?Sized, U> {
	min_root: Option<&'a T>,
	count: usize,
	container: PhantomData<&'a mut U>
}

impl<'a, T: Node<'a> + ?Sized + 'a, U> FibHeap<'a, T, U> {
	pub fn new(_container: &'a U) -> Self {
		Self{min_root: None, count: 0, container: PhantomData}
	}
	pub fn peek_min(&self) -> Option<&'a T> {
		self.min_root
	}
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
	pub fn push(&mut self, ent: &'a T) {
		unsafe { self.reattach(ent) }
		self.count += 1;
		#[cfg(test)]{
			assert_eq!(self.check(), Ok(()))
		}
	}
	pub unsafe fn decrease_key(&mut self, ent: &'a T) {
		if unsafe { *ent.get_raw().parent.get()}.is_some_and(|p|ent.cmp(p) == Ordering::Less) {
			self.separate_node(ent)
		}
	}
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
    use std::{cell::UnsafeCell, cmp::Ordering, ops::Deref};

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
				unsafe { *node.multiple.get() += *node.prime.get() };
				if unsafe { *node.multiple.get() } < ub {
					my_heap.push(node);
				};
			}
			if !my_heap.peek_min().is_some_and(|g|unsafe { *g.multiple.get() } == n) {
				prime_sum += n;
				if n*n < ub {
					let node = Box::new(GenNode{multiple: (n*n).into(), prime: n.into(), _node: Default::default()});
					ref_holder.push(node);
					let last_ref = unsafe { (ref_holder.last().unwrap().deref() as *const GenNode).as_ref() }.unwrap();
					my_heap.push(last_ref);
				}
			}
		}
		eprintln!("Sum of primes < {} = {}", ub, prime_sum);
		assert_eq!(prime_sum, 1060);
	}
}
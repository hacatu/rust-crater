use std::cmp::{max, Ordering};

/// An implicit binary heap that can efficiently find the min or max of its contained elements.
/// - Find min / find max: O(1)
/// - Pop min / pop max: O(log(n))
/// - Push: O(log(n))
/// - Heapify: O(n)
/// The comparison function can be changed after creation by just calling MmHeap::ify_by again.
pub struct MmHeap<T> {
    buf: Vec<T>
}



impl<T> MmHeap<T> {
	/// Create an empty MmHeap
    pub fn new() -> Self {
        Self{buf: Vec::new()}
    }

	/// Create an MmHeap out of a vector and immediately heapify it according to cmp
	/// Note that cmp takes references, which means the common case of storing references
	/// in the heap can require wrapping the comparison function since it will have to take
	/// references to references
    pub fn make(buf: Vec<T>, cmp: &impl Fn(&T, &T) -> Ordering) -> Self {
        let mut res = Self{buf};
        res.ify_by(cmp);
        res
    }

	/// Reorder the heap according to a new comparison function.
	/// It's impossible to add to the heap's private buffer directly, so this is only necessary
	/// when changing the comparison function.  When calling MmHeap::push etc with the same comparison
	/// function, the heap will maintain its invariant
    pub fn ify_by(&mut self, cmp: &impl Fn(&T, &T) -> Ordering) {
        let nonleaf_idx_upper_bound = (usize::MAX >> 1) >> self.buf.len().leading_zeros();
        for i in (0..nonleaf_idx_upper_bound).rev()
            { self.sift_down_by(i, cmp) }
    }

	/// Get the number of elements in the heap
	pub fn len(&self) -> usize {
		self.buf.len()
	}

	/// Get the minimum element without removing it
    pub fn peek_min(&self) -> Option<&T> {
        self.buf.first()
    }

	/// Get the maximum element without removing it
    pub fn peek_max_by(&self, cmp: &impl Fn(&T, &T) -> Ordering) -> Option<&T> {
        match self.buf.get(1..3) {
            Some(slice_ref) => slice_ref.into_iter().max_by(|a, b|cmp(a, b)),
            None => self.buf.get(max(self.buf.len(), 1) - 1)
        }
    }

    fn sift_up_by(&mut self, mut i: usize, cmp: &impl Fn(&T, &T) -> Ordering) {
        if i == 0 || i >= self.buf.len()
            { return }
        // nodes with index i will be in layer n where 2^n is the maximal power of 2 <= i + 1,
        // so we can check if n is odd/even by checking if the number of leading zeros in (i + 1)
        // is odd/even.  In odd layers, nodes should be >= their descendents, and in even layers, <=.
        let mut ord = match (i + 1).leading_zeros()&1
            { 1 => Ordering::Less, _ => Ordering::Greater };
        let mut i1 = (i - 1) >> 1;
        if cmp(&self.buf[i1], &self.buf[i]) == ord {
            self.buf.swap(i, i1);
            i = i1;
            ord = ord.reverse()
        }
        while i > 2 {
            i1 = (i - 3) >> 2;
            if cmp(&self.buf[i], &self.buf[i1]) == ord {
                self.buf.swap(i, i1);
                i = i1
            } else { break }
        }
    }

    fn sift_down_by(&mut self, mut i: usize, cmp: &impl Fn(&T, &T) -> Ordering) {
        let ord = match (i + 1).leading_zeros()&1
            { 1 => Ordering::Less, _ => Ordering::Greater };
        while 2*i + 1 < self.buf.len() {
            // Find m, the index of the extremal element among the children and grandchildren
            // of the element at index i. For min layers, extremal means
            // minimal, and for max layers it means maximal
            let mut m = 2*i + 1;
            for ii in [2*i + 2, 4*i + 3, 4*i + 4, 4*i + 5, 4*i + 6].into_iter().take_while(|&j|j<self.buf.len()) {
                if cmp(&self.buf[ii], &self.buf[m]) == ord
                    { m = ii }
            }
            // If m is a grandchild of i (as should be the case most of the time)
            // we may have to sift down farther after fixing up here
            if m > 2*i + 2 {
                if cmp(&self.buf[m], &self.buf[i]) == ord {
                    self.buf.swap(m, i);
                    let p = (m - 1) >> 1;
                    if cmp(&self.buf[p], &self.buf[m]) == ord
                        { self.buf.swap(m, p) }
                    i = m;
                } else { break }
            } else {// otherwise em is a direct child so it must be a leaf or its invariant would be wrong
                if cmp(&self.buf[m], &self.buf[i]) == ord
                    { self.buf.swap(m, i) }
                break
            }
        }
    }

	/// Insert an element into the heap
	/// Elements that compare equal are fine, but their order will be unspecified
    pub fn push_by(&mut self, e: T, cmp: &impl Fn(&T, &T) -> Ordering) {
        self.buf.push(e);
        self.sift_up_by(self.buf.len() - 1, cmp)
    }

	/// Get the minimal element and remove it
    pub fn pop_min_by(&mut self, cmp: &impl Fn(&T, &T) -> Ordering) -> Option<T> {
        self.pop_idx_by(0, cmp)
    }

	/// Get the maximal element and remove it
    pub fn pop_max_by(&mut self, cmp: &impl Fn(&T, &T) -> Ordering) -> Option<T> {
        match self.buf.get(1..3) {
            Some(slice_ref) => self.pop_idx_by(1 + slice_ref.into_iter().enumerate().max_by(|(_i,a),(_j,b)|cmp(a,b)).unwrap().0, cmp),
            None => self.buf.pop()
        }
    }

    fn pop_idx_by(&mut self, i: usize, cmp: &impl Fn(&T, &T) -> Ordering) -> Option<T> {
        let l = self.buf.len();
        if i + 1 >= l
            { return self.buf.pop() }
        self.buf.swap(i, l - 1);
        let res = self.buf.pop();
        self.sift_down_by(i, cmp);
        res
    }

	/// Insert a new element and remove the min element in the resulting heap in a single operation
	/// Could be faster than MmHeap::push_by + MmHeap::pop_min_by separately
	/// (the current implementation is only more efficient if the heap is empty, but it will always
	/// be at least as good)
    pub fn pushpop_min_by(&mut self, e: T, cmp: &impl Fn(&T, &T) -> Ordering) -> T {
        if self.buf.is_empty() { e } else {
            self.push_by(e, cmp);
            self.pop_min_by(cmp).unwrap()
        }
    }

	/// Insert a new element and remove the max element in the resulting heap in a single operation
    pub fn pushpop_max_by(&mut self, e: T, cmp: &impl Fn(&T, &T) -> Ordering) -> T {
        if self.buf.is_empty() { e } else {
            self.push_by(e, cmp);
            self.pop_max_by(cmp).unwrap()
        }
    }

    pub fn extend_by<U: IntoIterator<Item=T>>(&mut self, iter: U, cmp: &impl Fn(&T, &T) -> Ordering) {
        for x in iter {
            self.push_by(x, cmp)
        }
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


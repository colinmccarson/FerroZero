use std::mem::MaybeUninit;
use std::ops::Index;
use std::ops::IndexMut;
// TODO want to use this for all MoveLists
pub struct Array<T, const N: usize> {  // TODO use maybe uninit things
    arr: [MaybeUninit<T>; N],
    len: usize,
}

impl<T, const N: usize> Array<T, N> {
    pub fn new() -> Self {
        let arr: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };
        Self { arr, len: 0 }
    }

    pub fn push(&mut self, item: T) {
        self.arr[self.len].write(item);
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_raw_ref(&self) -> &[T; N] {
        assert_eq!(self.len, N, "Array is not fully initialized!");
        unsafe { &*(&self.arr as *const [MaybeUninit<T>; N] as *const [T; N]) }
    }

    pub fn as_raw(&self) -> [T; N] {
        assert_eq!(self.len, N, "Array is not fully initialized!");
        unsafe { std::mem::transmute::<[MaybeUninit<T>; N], [T; N]>(self.arr) }
    }

    pub fn iter(&self) -> ArrayIter<'_, T> {
        let slice: &[T] = unsafe {
            std::slice::from_raw_parts(self.arr.as_ptr() as *const T, self.len)
        };
        ArrayIter { slice }
    }
}


impl<T, const N: usize> Index<usize> for Array<T, N> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len, "index out of bounds");
        unsafe { self.arr[index].assume_init_ref() }
    }
}

impl<T, const N: usize> IndexMut<usize> for Array<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < self.len, "index out of bounds");
        unsafe { self.arr[index].assume_init_mut() }
    }
}

pub struct ArrayIter<'a, T> {
    slice: &'a [T],
}

impl <'a, T> Iterator for ArrayIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            None
        } else {
            let (first, rest) = self.slice.split_first().unwrap();
            self.slice = rest;
            Some(first)
        }
    }
}

impl<'a, T, const N: usize> IntoIterator for Array<T, N> {
    type Item = T;
    type IntoIter = ArrayIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
       self.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a Array<T, N> {
    type Item = &'a T;
    type IntoIter = ArrayIter<'a, T>;
    fn into_iter(&self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, const N: usize> Drop for Array<T, N> {
    fn drop(&mut self) {
        for elem in &mut self.arr[0..self.len] {
            unsafe { elem.assume_init_drop() };
        }
    }
}

pub struct RingBuffer<T, const N: usize> {
    arr: Array<T, N>,
    head: usize,
    tail: usize,
    size: usize
}

impl<T, const N: usize> RingBuffer<T, N> {
    pub fn new() -> Self {
        Self { arr: Array::new(), head: 0usize, tail: 0, size: 0 }
    }

    pub fn push(&mut self, item: T) -> Result<(), ()> {
        if self.size == N {
            return Err(());
        }
        self.arr.push(item);
        self.size += 1;
        self.tail = (self.tail + 1 ) % N;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }
        let result = unsafe { self.arr[self.head].assume_init() };
        self.head += (self.head + 1) % N;
        self.size -= 1;
        Some(result)
    }

    pub fn len(&self) -> usize {
        self.size
    }
}


impl<T, const N: usize> Index<usize> for RingBuffer<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.size, "index out of bounds");
        self.arr[(self.head + index) % N]
    }
}

impl<T, const N: usize> IndexMut<usize> for RingBuffer<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.size, "index out of bounds");
        self.arr[(self.head + index) % N]
    }
}


pub struct Arena<T> {
    arena: Vec<Option<T>>,
    free: Vec<usize>,
    size: usize,
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Self { arena: Vec::new(), free: Vec::new(), size: 0 }
    }

    pub fn push(&mut self, item: T) -> usize {
        if self.free.len() > 0 {
            let ind = self.free.pop().unwrap();
            self.arena[ind] = Some(item);
            self.size += 1;
            ind
        }
        else {
            let ind = self.arena.len();
            self.arena.push(Some(item));
            self.size += 1;
            ind
        }
    }

    pub fn pop(&mut self, index: usize) -> T {
        let ret = self.arena[index].take().unwrap();
        self.free.push(index);
        self.size -= 1;
        ret
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.arena[index].as_ref()
    }

    pub fn get_mut(&self, index: usize) -> Option<&mut T> {
        self.arena[index].as_mut()
    }

    pub fn len(&self) -> usize {
        self.size
    }

}

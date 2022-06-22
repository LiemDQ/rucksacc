use std::iter::{FusedIterator, ExactSizeIterator};


/// Represents a position in a text file.
/// Useful for error reporting during parsing, as the errors will occur in a specific position.
#[derive(Debug, Clone)]
pub struct TextPosition {
    pub line: usize,
    pub col: usize,
    pub filename: Option<String>,
}

impl TextPosition {
    pub fn new(line: usize, col: usize, filename: Option<&str>) -> Self {
        Self {
            line: line,
            col: col,
            filename: filename.and_then(|s| Some(s.to_string())),
        }
    }

    pub fn new_file_only(filename: &str) -> Self {
        Self { line: 0, col: 0, filename: Some(filename.to_string()) }
    }

    pub fn new_without_file(line: usize, col: usize) -> Self {
        Self { line: line, col: col, filename: None }

    }

    pub fn add_offset(&self, offset: usize) -> Self {
        Self { line: self.line, col: self.col + offset, filename: self.filename.clone() }
    }
}

/// Trait for creating iterators that can peek ahead multiple entries.
pub trait PeekN: Iterator + Sized {
    fn peekable_n(self) -> PeekNIterator<Self>;
}

impl<I: Iterator> PeekN for I {
    fn peekable_n(self) -> PeekNIterator<I> {
        PeekNIterator { 
            iterator: self, 
            queue: Vec::new(), 
            cursor: 0usize 
        }
    }
}

/// Iterator that allows for peeking multiple values ahead without consuming a value.
/// This is needed for parsing certain symbols, such as `>>=`, which consume 3 characters and thus cannot
/// be parsed with a single `char` lookahead.
/// 
/// The implementation is taken from [`peekmore`]: https://github.com/foresterre/peekmore
/// 
/// Some implementations have been changed slightly, and a few additional methods have been added. It was originally
/// intended to use a custom Error type which is why this implementation was copied over rather than simply importing
/// the original [`peekmore`] crate.
/// 
/// Note that this iterator must allocate memory in order to suppose peeking ahead by multiple values. 
/// If performance and memory contraints are important and there is no need to peek ahead by more than 1 value,
/// use the standard [`core::iter::Peekable`] iterator instead.
pub struct PeekNIterator<I: Iterator> {
    iterator: I,
    ///Contains the items of the iterator that have not been consumed
    /// Once an element is consumed, it will be dequeued.
    queue: Vec<Option<I::Item>>,

    /// Points to the element currently being looked at. Peeking at the 0th element is equivalent
    /// to peeking with [`core::iter::Peekable::peek`].
    cursor: usize,
}

impl<I: Iterator> PeekNIterator<I> {
    /// Get a reference to the element where the cursor currently points to. If no such element exists, 
    /// return `None` will be returned. 
    #[inline]
    pub fn peek(&mut self) -> Option<&I::Item> {
        self.fill_queue(self.cursor);
        self.queue.get(self.cursor).and_then(|v| v.as_ref())
    }

    /// Advance the cursor to the next element and return a reference to that value.    
    #[inline]
    pub fn peek_next(&mut self) -> Option<&I::Item> {
        let this = self.advance_cursor();
        this.peek()
    }
    
    /// Move the cursor back to the previous element and return a reference to that value.
    /// The return value will be wrapped in an `Ok` result.
    /// 
    /// If no such element exists, an `Err` result containing [`PeekError::ElementHasBeenConsumed`] will
    /// be thrown instead. 
    #[inline]
    pub fn peek_previous(&mut self) -> Result<Option<&I::Item>, PeekError> {
        if self.cursor >= 1 {
            self.move_cursor_back().map(|iter| iter.peek())
        } else {
            Err(PeekError::ElementHasBeenConsumed)
        }
    }
    
    /// Move the cursor `n` steps forward and peek at the element the cursor points to.
    #[inline]
    pub fn peek_forward(&mut self, n: usize) -> Option<&I::Item> {
        let this = self.advance_cursor_by(n);
        this.peek()
    }

    /// Move the cursor `n` steps backward and peek at the element the cursor then points to.
    /// 
    /// If there aren't `n` elements prior to the element the cursor currently points at, a
    /// [`PeekError::ElementHasBeenConsumed`] is returned instead. 
    /// 
    /// If you prefer to peek at the first unconsumed element rather than throwing an error, use
    /// the [`peek_backward_or_first`] method instead.
    #[inline]
    pub fn peek_backward(&mut self, n: usize) -> Result<Option<&I::Item>, PeekError> {
        let _ = self.move_cursor_back_by(n)?;
        Ok(self.peek())
    }

    /// Peek at nth element without moving the cursor.
    #[inline]
    pub fn peek_nth(&mut self, n: usize) -> Option<&I::Item> {
        self.fill_queue(n);
        self.queue.get(n).and_then(|v| v.as_ref())
    }

    /// Returns a view into the elements between `start..end`, without consuming them or advancing the iterator. 
    /// 
    /// If `end` goes past the end of the underlying container, `None` is returned at those
    /// indices.
    pub fn peek_range(&mut self, start: usize, end: usize) -> &[Option<I::Item>] {
        if start >= end {
            return &[];
        }

        if end > self.queue.len() {
            self.fill_queue(end);
        }

        &self.queue.as_slice()[start..end]
    }
    
    /// Returns a view into the next `n` unconsumed elements of the iterator.
    #[inline]
    pub fn peek_next_n(&mut self, n: usize) -> &[Option<I::Item>] {
        self.peek_range(0, n)
    }

    /// Returns a view of all subsequent items that satisfy the predicate `P`.
    pub fn peek_while<P: Fn(Option<&I::Item>) -> bool> (&mut self, predicate: P) -> &[Option<I::Item>] {
        let mut end = 0;
        let mut item = self.peek();
        while predicate(item) {
            end += 1;
            item = self.peek_nth(end);
        }

        &self.queue.as_slice()[0..end]
    }

    /// Check if the underlying iterable starts with the items in `collection`. 
    /// 
    /// This function does not move the cursor or consume the underlying elements. 
    /// 
    /// ```rust
    /// use trucc::utils::PeekN;
    /// 
    /// let string = "Hello world!";
    /// let mut iter = string.chars().peekable_n();
    /// 
    /// assert!(iter.starts_with("Hel".chars()));
    /// assert!(!iter.starts_with("lo".chars()));
    /// ```
    pub fn starts_with<T>(&mut self, collection: T) -> bool 
        where T: IntoIterator<Item = <PeekNIterator<I> as Iterator>::Item>,
              T::Item: PartialEq
    {
        let iter = collection.into_iter();
        for (n,v) in iter.enumerate() {
            let x = self.peek_nth(n);
            if x.is_none() {
                return false;
            }

            if *x.unwrap() != v {
                return false;
            }
        }
        true
    }

    /// Fills the queue up to the cursor, inclusively.
    #[inline]
    fn fill_queue(&mut self, req_elements: usize) {
        let stored_elements = self.queue.len();
        if stored_elements <= req_elements {
            for _ in stored_elements..=req_elements {
                self.push_next_to_queue();
            }
        }
    }
    
    ///Consume the underlying iterator element and push an element into the queue. 
    #[inline]
    fn push_next_to_queue(&mut self) {
        let item = self.iterator.next();
        self.queue.push(item);
    }

    /// Advance cursor to the next peekable element, checking for overflow.
    #[inline]
    pub fn advance_cursor(&mut self) -> &mut Self {
        self.increment_cursor();
        self
    }
    
    /// Advance cursor by `n`, checking for overflow. 
    /// Sets the cursor to the maximum value of `usize` otherwise.
    #[inline]
    pub fn advance_cursor_by(&mut self, n: usize) -> &mut Self {
        if self.cursor < core::usize::MAX - n {
            self.cursor += n;
        } else {
            self.cursor = core::usize::MAX;
        }
        self
    }

    /// Advance cursor by n, without checking for overflow.
    #[inline]
    pub fn advance_cursor_by_unchecked(&mut self, n: usize) -> &mut Self {
        self.cursor += n;
        self
    }
    
    /// Move the cursor forward until the predicate `P` is no longer true.
    /// Afterwards, the cursor will point to the first instance where `P` is false or `P` is `None`.
    #[inline]
    pub fn advance_cursor_while<P: Fn(&I::Item) -> bool>(&mut self, predicate: P)
    -> &mut Self {
        if let Some(view) = self.peek() {
            if predicate(view) {
                self.increment_cursor();
                self.advance_cursor_while(predicate);
            }
        } 
        self
    }
    
    /// Move the cursor to the `n`-th element of the queue. If `n` exceeds the length of the queue, 
    /// the cursor is moved to the length of the queue instead. 
    #[inline]
    pub fn move_nth(&mut self, mut n: usize) -> &mut Self {
        n = n.clamp(0usize, self.queue.len()-1);
        self.cursor = n;
        self
    }

    /// Move the cursor to the previous peekable element, If such an element doesn't exist, a `PeekError` will be 
    /// thrown. Otherwise, a mutable reference to the iterator wrapped in an `Ok` will be returned. 
    #[inline]
    pub fn move_cursor_back(&mut self) -> Result<&mut Self, PeekError> {
        if self.cursor >= 1 {
            self.decrement_cursor();
            Ok(self)
        } else {
            Err(PeekError::ElementHasBeenConsumed)
        }
    }

    /// Moves cursor back by `n` elements. If there are insufficient elements, an error will be returned instead. 
    /// In the event of an error, the cursor will remain in its previous position.
    #[inline]
    pub fn move_cursor_back_by(&mut self, n: usize,) -> Result<&mut Self, PeekError> {
        if self.cursor < n {
            Err(PeekError::ElementHasBeenConsumed)
        } else {
            self.cursor -= n;
            Ok(self)
        }
    }

    /// Moves cursor back by `n` elements, or until its position is reset to the first non-consumed element.
    #[inline]
    pub fn move_cursor_back_or_reset(&mut self, n: usize) -> &mut Self {
        if self.cursor < n {
            self.reset_cursor();
        } else {
            self.cursor -= n;
        }
        self
    }

    /// Reset the position of the cursor.
    /// 
    /// If [`peek`] is called just after a reset, it will return a reference to the first element.
    #[inline]
    pub fn reset_cursor(&mut self){
        self.cursor = 0;
    }

    /// Advance the iterator to the cursor, discarding all elements in between 
    /// the iterator position and the cursor position.
    /// 
    /// After calling this method, `iter.peek() == iter.next().as_ref()`.
    /// 
    /// Note that for instances where the cursor is not pointing to the end of the queue, i.e. the cursor has been advanced and moved back,
    /// the removal will require shifting the elements in the queue once. If this proves to be problematic
    /// from a performance standpoint, `VecDeque` can be substituted in instead, for provides efficient removal from the front and back.
    /// 
    ///```rust
    /// use trucc::utils::PeekN;
    ///
    /// let iterable = [1, 2, 3, 4];
    /// let mut iter = iterable.iter().peekable_n();
    ///
    /// iter.advance_cursor_by(2);
    /// assert_eq!(iter.peek(), Some(&&3));
    /// assert_eq!(iter.next(), Some(&1));
    /// iter.skip_to_cursor();
    /// assert_eq!(iter.peek(), Some(&&3));
    /// assert_eq!(iter.next(), Some(&3));
    ///```
    pub fn skip_to_cursor(&mut self) {
        // if the cursor is greater than the queue length,
        // remove the overflow from the iterator.
        // note that the underlying iterator is "ahead" of the queue, which is why this approach works.
        for _ in 0..self.cursor.saturating_sub(self.queue.len()){
            let _ = self.iterator.next(); 
        }

        // if the cursor is equal to the queue length, just clear the queue for better performance.
        if self.cursor == self.queue.len() {
            self.queue.clear();
        } else {            
            self.queue.drain(0..self.cursor.clamp(0, self.queue.len()));
        }
        self.cursor = 0;
    }

    /// Increments the cursor, checking for overflow.
    #[inline]
    fn increment_cursor(&mut self) {
        if self.cursor < core::usize::MAX {
            self.cursor += 1;
        }
    }
    
    /// Decrements the cursor, checking for underflow.
    #[inline]
    fn decrement_cursor(&mut self){
        if self.cursor > core::usize::MIN {
            self.cursor -= 1;
        }
    }

    // #[doc(hidden)]
    // #[cfg(test)]
    #[inline]
    pub fn cursor(&self) -> usize {
        self.cursor
    }
}

impl <'a, I: Iterator> Iterator for PeekNIterator<I>{
    type Item = I::Item;
    /// Advances the iterator and returns the next value.
    /// 
    /// Returns `None` when the iteration is finished.
    fn next(&mut self) -> Option<Self::Item> {
        let res = if self.queue.is_empty() {
            self.iterator.next()
        } else {
            self.queue.remove(0)
        };
        self.decrement_cursor();
        res
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for PeekNIterator<I> {}

impl<I: FusedIterator> FusedIterator for PeekNIterator<I> {}

#[derive(Debug, Eq, PartialEq)]
pub enum PeekError {
    ElementHasBeenConsumed,
}
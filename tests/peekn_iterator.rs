use rucksacc::utils::{PeekN, PeekNIterator};

#[test]
fn peek_forward_with_reassignment() {
    let iterable = [1,2,3,4];
    let mut peek = iterable.iter().peekable_n();

    assert_eq!(peek.peek(), Some(&&1));

    let peek = peek.advance_cursor();
    assert_eq!(peek.peek(), Some(&&2));

    let peek = peek.advance_cursor();
    assert_eq!(peek.peek(), Some(&&3));

    let peek = peek.advance_cursor();
    assert_eq!(peek.peek(), Some(&&4));

    let peek = peek.advance_cursor();
    assert_eq!(peek.peek(), None);
}

#[test]
fn peek_forward_and_reset_view() {
    let iterable = [1,2,3,4];
    let mut iter = iterable.iter().peekable_n();
    let v1 = iter.peek();
    assert_eq!(v1, Some(&&1));

    let v2 = iter.peek_next();
    assert_eq!(v2, Some(&&2));

    iter.reset_cursor();
    let v1again = iter.peek();
    assert_eq!(v1again, Some(&&1));

    let v2again = iter.peek_next();
    assert_eq!(v2again, Some(&&2));
}

#[test]
fn peek_and_advance() {
    let iterable = [1,2,3,4];
    let mut peek = iterable.iter().peekable_n();

    let v1 = peek.peek();
    assert_eq!(v1, Some(&&1));

    let v2 = peek.peek_next();
    assert_eq!(v2, Some(&&2));

    let v3 = peek.peek_next();
    assert_eq!(v3, Some(&&3));

    let v4 = peek.peek_next();
    assert_eq!(v4, Some(&&4));

    let v5 = peek.peek_next();
    assert_eq!(v5, None);
}

#[test]
fn peek_advance_and_move_backwards_or_reset() {
    let iterable = "abcdefg";
    let mut iter = iterable.chars().peekable_n();

    let v1 = iter.peek();
    assert_eq!(v1, Some(&'a'));

    iter.advance_cursor_by(3);
    let v2 = iter.peek();
    assert_eq!(v2, Some(&'d'));

    let iter = iter.move_cursor_back().unwrap();
    let v3 = iter.peek();
    assert_eq!(v3, Some(&'c'));

    let iter = iter.move_cursor_back_or_reset(5);
    let v1again = iter.peek();
    assert_eq!(v1again, Some(&'a'));
}

#[test]
fn move_backward_or_reset_empty() {
    let iterable = "";
    let mut iter = iterable.chars().peekable_n();

    assert!(iter.peek().is_none());
    assert_eq!(iter.cursor(), 0);

    let _ = iter.move_cursor_back_or_reset(5);

    assert!(iter.peek().is_none());
    assert_eq!(iter.cursor(), 0);
}

#[test]
fn move_forward_while_empty() {
    let iterable: [i32; 0] = [];
    let mut iter = iterable.iter().peekable_n();

    let _ = iter.advance_cursor_while(|i| **i != 3);
    let peek = iter.peek();

    assert_eq!(peek, None);
    assert_eq!(iter.cursor(), 0);
}

#[test]
fn move_forward_while_is_some() {
    let iterable = [1,2,3,4];
    let mut iter = iterable.iter().peekable_n();

    let _ = iter.advance_cursor_while(|_| ());

    let peek = iter.peek();
    assert_eq!(peek, None);
    assert_eq!(iter.cursor(), 4);

    let result = iter.move_cursor_back();
    assert!(result.is_ok());
    assert_eq!(result.unwrap().peek(), Some(&&4));
}

#[test]
fn peek_nth_with_valid_amount() {
    let iterable = [1,2,3,4];
    let mut iter = iterable.iter().peekable_n();

    assert_eq!(iter.peek_nth(0), Some(&&1));
    assert_eq!(iter.cursor(), 0);
    assert_eq!(iter.peek_nth(1), Some(&&2));
    assert_eq!(iter.cursor(), 0);
    assert_eq!(iter.peek_nth(2), Some(&&3));
    assert_eq!(iter.cursor(), 0);
    assert_eq!(iter.peek_nth(3), Some(&&4));
    assert_eq!(iter.cursor(), 0);
    assert_eq!(iter.peek_nth(4), None);
    assert_eq!(iter.cursor(), 0);
}

#[test]
fn peek_backward_to_beginning() {
    let iterable = [1,2,3,4];
    let mut iter = iterable.iter().peekable_n();

    let _ = iter.advance_cursor_by(3);
    assert_eq!(iter.peek(), Some(&&4));
    
    let result = iter.peek_backward(2);
    assert_eq!(result.unwrap(), Some(&&2));

    let result = iter.peek_backward(1);
    assert_eq!(result.unwrap(), Some(&&1));

    let result = iter.peek_backward(1);
    assert!(result.is_err());

    assert_eq!(iter.peek(), Some(&&1));
}

#[test]
fn peek_nth_empty() {
    let iterable: [i32; 0] = [];

    let mut iter = iterable.iter().peekable_n();

    assert_eq!(iter.peek_nth(0), None);
    assert_eq!(iter.cursor(), 0);
    assert_eq!(iter.peek_nth(1), None);
    assert_eq!(iter.cursor(), 0);
}

#[test]
fn moving_cursor_backwards_past_beginning_returns_none() {
    let iterable = "abcd";
    let mut iter = iterable.chars().peekable_n();

    let v1 = iter.peek();
    assert_eq!(v1, Some(&'a'));

    let iter = iter.advance_cursor_by(2);
    let v2 = iter.peek();
    assert_eq!(v2, Some(&'c'));

    let result = iter.move_cursor_back_by(5);
    assert!(result.is_err());
}

#[test]
fn check_advance_while() {
    let iterable = [1,2,3,4,5,6,7];
    let mut iter = iterable.iter().peekable_n();

    let v1 = iter.peek();
    assert_eq!(v1, Some(&&1));

    let iter = iter.advance_cursor_while(|x| x < 5);
    let v5 = iter.peek();
    assert_eq!(v5, Some(&&5));
}

#[test]
fn check_empty() {
    let iterable: [i32; 0] = [];
    let mut iter = iterable.iter().peekable_n();
    
    assert_eq!(iter.peek(), None);

    let v1 = iter.peek_next();
    assert_eq!(v1, None);

    let iter = iter.advance_cursor();
    assert_eq!(iter.peek(), None);
    assert_eq!(iter.peek_next(), None);
}

#[test]
fn check_skip_to_cursor() {
    let iterable = [1,2,3,4,5];
    let mut iter = iterable.iter().peekable_n();
    
    iter.advance_cursor_by(3);
    assert_eq!(iter.peek(), Some(&&4));
    
    iter.skip_to_cursor();
    assert_eq!(iter.peek(), Some(&&4));

    assert_eq!(iter.next(), Some(&4));
}

#[test]
fn skip_to_cursor_with_backwards_movement() {
    let iterable = [1,2,3,4,5,6];
    let mut iter = iterable.iter().peekable_n();

    iter.advance_cursor_by(3);
    assert_eq!(iter.peek(), Some(&&4));

    let iter = iter.move_cursor_back_by(2).unwrap();
    assert_eq!(iter.peek(), Some(&&2));

    iter.skip_to_cursor();
    assert_eq!(iter.peek(), Some(&&2));
    assert_eq!(iter.next(), Some(&2));
}

#[test]
fn skip_to_cursor_equal_length_queue() {
    let iterable = [1,2,3,4];
    let mut iter = iterable.iter().peekable_n();
    iter.peek();
    iter.advance_cursor();
    iter.skip_to_cursor();

    assert_eq!(iter.next(), Some(&2));
    assert_eq!(iter.next(), Some(&3));
    assert_eq!(iter.next(), Some(&4));
    assert_eq!(iter.next(), None);
}

#[test]
fn skip_to_cursor_on_empty_collection() {
    let mut iter = core::iter::empty::<i32>().peekable_n();
    iter.advance_cursor();
    assert_eq!(iter.cursor(), 1);

    iter.skip_to_cursor();
    assert_eq!(iter.cursor(), 0);

    assert!(iter.peek().is_none());
}

#[test]
fn skip_to_cursor_is_noop_when_queue_is_empty_from_no_peeking() {
    let iterable = [1,2,3,4];
    let mut iter = iterable.iter().peekable_n();
    iter.skip_to_cursor();

    assert_eq!(iter.peek(), Some(&&1));
}

#[test]
fn skip_to_cursor_is_noop_when_queue_is_empty_from_iteration() {
    let iterable = [1,2,3,4];
    let mut iter = iterable.iter().peekable_n();
    
    iter.peek_forward(2);
    iter.next();
    iter.next();
    iter.next();

    iter.skip_to_cursor();
    assert_eq!(iter.peek(), Some(&&4));
}

#[test]
fn peek_range_from_start_smaller_than_underlying_container() {
    let iterable = [0,1,2,3];
    let mut iter = iterable.iter().peekable_n();
    let view = iter.peek_range(0, 2);
    
    assert_eq!(view[0], Some(&0));
    assert_eq!(view[1], Some(&1));
    assert_eq!(view.len(), 2);
}

#[test]
fn peek_range_invalid_indices_returns_empty_slice() {
    let iterable = [0,1,2,3];
    let mut iter = iterable.iter().peekable_n();
    let view = iter.peek_range(3, 0);

    assert!(view.is_empty());
    assert_eq!(view.len(), 0);
}

#[test]
fn peek_range_larger_than_underlying_container() {
    let iterable = [0,1,2,3];
    let mut iter = iterable.iter().peekable_n();
    let view = iter.peek_range(0, 6);
    
    assert_eq!(view[3], Some(&3));
    assert!(view[4].is_none());
    assert!(view[5].is_none());
}

#[test]
fn peek_range_from_middle() {
    let iterable = [0,1,2,3];
    let mut iter = iterable.iter().peekable_n();
    let view = iter.peek_range(2, 5);
    
    assert_eq!(view[0], Some(&2));
    assert!(view[2].is_none());
}

#[test]
fn peek_next_n_with_valid_amount() {
    let iterable = [0,1,2,3];
    let mut iter = iterable.iter().peekable_n();
    let n = 3;
    let view = iter.peek_next_n(n);
    
    assert_eq!(view.len(), n);
    assert_eq!(view[0], Some(&0));
    assert_eq!(view[2], Some(&2));
}

#[test]
fn starts_with_works_with_strings(){
    let iterable = "Hello world!".to_string();
    let mut iter = iterable.chars().peekable_n();
    assert!(iter.starts_with("Hello".chars()));

    assert!(!iter.starts_with(" world".chars()));
}

#[test]
fn starts_with_works_with_slices(){
    let iterable = [1,2,3,4];
    let mut iter = iterable.iter().peekable_n();

    assert!(iter.starts_with(&[1,2,3]));
    assert!(!iter.starts_with(&[3]));
}

#[test]
fn check_peek_while() {
    let iterable = [2,4,6,8,10,7,12];
    let mut iter = iterable.iter().peekable_n();
    let slice = iter.peek_while(|x| if let Some(&&v) = x{v % 2 == 0} else {false});
    
    assert!(slice.starts_with(&[Some(&2)]));
    assert_eq!(slice.len(), 5);
    assert!(slice.ends_with(&[Some(&10)]));
}
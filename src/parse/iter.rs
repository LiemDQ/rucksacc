
///Utility functions for working with an iterator of Tokens. 
use crate::utils::{TextPosition,  PeekNIterator};
use crate::err::{ParseErrMsg, ParseRes};
use crate::lex::{Token, TokenKind};
use super::parse_err::*;



/// Ensures the next token in the iterator is of kind `tk`. If it is, skip to the next value.
/// If the token is of the wrong kind, throws an `Err` containing information about the expected 
/// kind.
/// 
/// This method is useful for parsing symbols that **must** appear in a particular branch. If the token 
/// needs to be returned, use `consume_if_equal` instead.
#[must_use]
pub fn ensure_and_consume<I: Iterator<Item = Token>>(iter: &mut PeekNIterator<I>,  tk: TokenKind) -> ParseRes<()> {
    //this should never be EOF
    let token = iter.peek().unwrap();
    if token.token_type != tk {
        return Err(gen_parser_err(ParseErrMsg::ExpectedSymbol(tk), token));
    } 
    let _ = iter.next();
    Ok(())
}

/// Determine if the next token in the iterator has TokenKind `kind`. This does not
/// consume the iterator element. This is useful for checking for optional symbols during parsing.
/// If consuming the element is desired, use [`consume_if_equal`] instead.
pub fn check_if_equal(iter: &mut impl Iterator<Item = Token>, kind: TokenKind) -> bool {
    let mut iter = iter.peekable();
    iter.peek()
        .and_then(|tok| Some(tok.token_type == kind))
        .unwrap_or(false)

}

/// Consumes the current iterator element if its kind is equal to `kind`. Returns `true` if the 
/// element is consumed, `false` otherwise. Unlike [`ensure_and_consume`], this does not throw an error
/// if the token is not equal. 
/// 
/// This method is useful for branches in the syntax where the presence of one symbol leads
/// to a different branch to parse. 
pub fn consume_if_equal(iter: &mut impl Iterator<Item = Token>, kind: TokenKind) -> bool {
    let mut iter = iter.peekable();
    if let Some(tok) = iter.peek(){
        if tok.token_type == kind {
            true
        } else {
            false
        }
    } else {
        false
    }
}

/// Extract the [`TextPosition`] from the next token in the iterator, without consuming it.
/// If there is no next token, returns an EOF error. 
pub fn extract_position(iter: &mut impl Iterator<Item = Token>) -> ParseRes<TextPosition> {
    let mut iter = iter.peekable();
    let token = match iter.peek() {
        Some(token) => token,
        None => return gen_eof_error(),
    };

    Ok(token.pos.clone())
}

/// Checks if the next item in the iterator is `None`, returning an `Err` if this is the case.
/// Returns the token otherwise.
#[inline]
pub fn consume_token_or_eof(iter: &mut impl Iterator<Item = Token>) -> ParseRes<Token> {
    match iter.next() {
        Some(token) => Ok(token),
        None => gen_eof_error(),
    }
}

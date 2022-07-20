use crate::err::{ParseErrMsg, ParseRes, ParseErr};
use crate::lex::{Token, TokenKind};
use crate::utils::TextPosition;

#[inline]
pub fn gen_parser_err(msg: ParseErrMsg, token: &Token) -> ParseErr {
    ParseErr::new(token.pos.clone(), token.pos.clone(), msg)
}

#[inline]
pub fn gen_eof_error<T>() -> ParseRes<T> {
    Err(gen_parser_err(ParseErrMsg::EOF,&Token::eof_token(None)))
}

#[inline]
pub fn gen_expected_error<T>(tk: &Token, expected: TokenKind) -> ParseRes<T> {
    Err(gen_parser_err(ParseErrMsg::ExpectedSymbol(expected), tk))
}

#[inline]
pub fn gen_internal_error<T>(tk: &Token, line: u32) -> ParseRes<T> {
    Err(gen_parser_err(ParseErrMsg::InternalError(line), tk))
}

#[inline]
pub fn gen_internal_error_pos<T>(pos: &TextPosition, line: u32) -> ParseRes<T> {
    Err(gen_parser_err_pos(ParseErrMsg::InternalError(line), pos))
}

#[inline]
pub fn gen_expected_one_of_error<T>(tk: &Token, expected: Vec<TokenKind>) -> ParseRes<T> {
    Err(gen_parser_err(ParseErrMsg::ExpectedOneOfSymbols(expected), tk))
}

#[inline]
pub fn gen_parser_err_pos(msg: ParseErrMsg, pos: &TextPosition) -> ParseErr {
    ParseErr::new(pos.clone(), pos.clone(), msg)
}

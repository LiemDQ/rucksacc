use crate::utils::{TextPosition, PeekNIterator};
use crate::err::{ParseErrMsg, ParseRes};
use std::collections::HashSet;
use std::fmt::Display;

use super::token_err::*;

#[derive(PartialEq, Clone, Debug)]
pub enum Keyword {
    Any,
    Typedef,
    Extern,
    Static,
    Auto,
    Restrict,
    Register,
    Bool,
    Const,
    Complex,
    ConstExpr,
    Generic,
    Volatile,
    Void,
    Signed,
    Unsigned,
    Char,
    Int,
    Short,
    Long,
    Float,
    Double,
    Struct,
    Enum,
    Union,
    Inline,
    ThreadLocal,
    StaticAssert,
    AlignOf,
    AlignAs,
    Atomic,
    Asm,
    Typeof,
    Attribute,
    Thread,
    If,
    Else,
    For,
    Do,
    While,
    Switch,
    Case,
    Default,
    Sizeof,
    Goto,
    Break,
    Continue,
    Return,
    NoReturn,
    INVALID,
}

impl Display for Keyword {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Keyword::Auto => "auto",
            Keyword::Break => "break",
            Keyword::Case => "case",
            Keyword::Const => "const",
            Keyword::Continue => "continue",
            Keyword::Default => "default",
            Keyword::Do => "do",
            Keyword::Double => "double",
            Keyword::Else => "else",
            Keyword::Enum => "enum",
            Keyword::Extern => "extern",
            Keyword::Float => "float",
            Keyword::For => "for",
            Keyword::Goto => "goto",
            Keyword::If => "if",
            Keyword::Inline => "inline",
            Keyword::Int => "int",
            Keyword::Long => "long",
            Keyword::Register => "register",
            Keyword::Restrict => "restrict",
            Keyword::Return => "return",
            Keyword::Short => "short",
            Keyword::Signed => "signed",
            Keyword::Sizeof => "sizeof",
            Keyword::Static => "static",
            Keyword::Struct => "struct",
            Keyword::Typedef => "typedef",
            Keyword::Union => "union",
            Keyword::Unsigned => "unsigned",
            Keyword::Void => "void",
            Keyword::While => "while",
            Keyword::AlignAs => "_Alignas",
            Keyword::AlignOf => "_Alignof",
            Keyword::Atomic => "_Atomic",
            Keyword::Bool => "_Bool",
            Keyword::Complex => "_Complex",
            Keyword::Generic => "_Generic",
            Keyword::NoReturn => "_Noreturn",
            Keyword::StaticAssert => "_Static_assert",
            Keyword::ThreadLocal => "_Thread_local",
            _ => panic!("Invalid keyword"),
        };
        write!(f, "{}", s)
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Punct {
    Any,
    Backslash,
    OpenParen,
    CloseParen,
    OpenBrace,
    CloseBrace,
    OpenBoxBracket,
    CloseBoxBracket,
    Comma,
    Semicolon,
    Colon,
    PoundPound, //## but i have no idea what this actually does
    Point,
    Arrow, // -> 
    Inc, // ++
    Dec, // --
    Add(PMKind), // +
    Sub(PMKind), // -
    Asterisk(AstKind), //asterisk has multiple possible meanings: pointer type, deference operator, multiplication
    Div,
    Mod,
    Not,
    BitwiseNot,
    Ampersand(AmpKind),
    Shl,
    Shr,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    Xor,
    Or,
    LAnd,
    LOr,
    Question,
    Assign,
    AssignAdd,
    AssignSub,
    AssignMul,
    AssignDiv,
    AssignMod,
    AssignShl,
    AssignShr,
    AssignAnd,
    AssignXor,
    AssignOr,
    Hash,
    Vararg,
}

impl Display for Punct {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Punct::OpenParen =>{"("},
            Punct::CloseParen =>{")"},
            Punct::OpenBrace =>{"{"},
            Punct::CloseBrace =>{"}"},
            Punct::Backslash => {"\\"},
            Punct::OpenBoxBracket => {"["},
            Punct::CloseBoxBracket =>{"]"},
            Punct::Comma =>{","},
            Punct::Semicolon =>{";"},
            Punct::Colon =>{":"},
            Punct::PoundPound =>{"##"}, //## but i have no idea what this actually does
            Punct::Point =>{"."},
            Punct::Arrow =>{"->"}, // -> 
            Punct::Inc =>{"++"}, // ++
            Punct::Dec =>{"--"}, // --
            Punct::Add(_) =>{"+"}, // +
            Punct::Sub(_) =>{"-"}, // -
            Punct::Asterisk(_) =>{"*"}, //asterisk has multiple possible meanings: pointer type, deference operator, multiplication
            Punct::Div =>{"/"},
            Punct::Mod =>{"%"},
            Punct::Not =>{"!"},
            Punct::BitwiseNot =>{"~"},
            Punct::Ampersand(_) =>{"&"},
            Punct::Shl =>{"<<"},
            Punct::Shr =>{">>"},
            Punct::Lt =>{"<"},
            Punct::Le =>{"<="},
            Punct::Gt =>{">"},
            Punct::Ge =>{">="},
            Punct::Eq =>{"=="},
            Punct::Ne =>{"!="},
            Punct::Xor =>{"^"},
            Punct::Or =>{"|"},
            Punct::LAnd =>{"&&"},
            Punct::LOr =>{"||"},
            Punct::Question =>{"?"},
            Punct::Assign =>{"="},
            Punct::AssignAdd =>{"+="},
            Punct::AssignSub =>{"-="},
            Punct::AssignMul =>{"*="},
            Punct::AssignDiv =>{"/="},
            Punct::AssignMod =>{"%="},
            Punct::AssignShl =>{"<<="},
            Punct::AssignShr =>{">>="},
            Punct::AssignAnd =>{"&="},
            Punct::AssignXor =>{"^="},
            Punct::AssignOr =>{"|="},
            Punct::Hash =>{"#"},
            Punct::Vararg =>{"..."},
            _ => panic!("Invalid punctuation"),
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AstKind {
    Undet, //undetermined
    Deref, //deference
    Ptr, //pointer
    Mult, //multiplication
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmpKind {
    Undet, //undetermined
    And, //bitwise and
    Addr, //address
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PMKind {
    Undet, //undetermined
    Unary,
    Binary,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Bits {
    Bits8,
    Bits16,
    Bits32,
    Bits64,
}

/// TODO: consider splitting out the punctuation and keywords into the actual enum as
/// it is leading to unnecessary layering right now
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Ident(String), // identifiers
    Punct(Punct), // punctuators
    Keyword(Keyword), // keywords
    Str(String), //string literals
    Char(char), //char literals
    Int(i64, Bits), //numeric literals
    Float(f64), //float literals
    PPNum, //preprocessor numbers
    EOF, //end of file
}

impl Display for TokenKind {
    /// TokenKinds are converted into their original C code representations. 
    /// This means that converting C code into a token stream, and then printing it
    /// via `fmt` should result in the same C code back, except with all whitespace removed
    /// (since C ignores whitespace).
    /// 
    /// This is particularly useful for writing unit tests.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            TokenKind::Ident(id) => id.to_string(),
            TokenKind::Punct(pct) => format!("{}", pct),
            TokenKind::Keyword(kw) => format!("{}", kw),
            TokenKind::Str(st) => {st.clone()}, //need to add escaped quotes before and after
            TokenKind::Char(c) => {c.to_string()},
            TokenKind::Int(val, _) => {val.to_string()},
            TokenKind::Float(val) => {val.to_string()},
            TokenKind::PPNum => {todo!("Preprocessor numbers")},
            TokenKind::EOF => {String::new()},

        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub token_type : TokenKind, //token kind
    pub hideset : HashSet<String>, // for macro expansion
    pub pos: TextPosition, 
    pub has_space: bool, //true if the token follows a whitespace character
    pub at_bol: bool, //is the token at the beginning of a line?
}

impl Token {
    pub fn new(ttype: TokenKind, line: usize, col: usize, filename: Option<&str>) -> Self {
        Self {
            token_type: ttype,
            at_bol: false,
            has_space: false,
            hideset: HashSet::new(),
            pos: TextPosition::new(line, col, filename),
        }
    }

    pub fn add_hideset(&mut self, s: String){
        self.hideset.insert(s);
    }

    /// generates an token to indicate end of file. 
    pub fn eof_token(filename: Option<&str>) -> Self {
        Self { 
            token_type: TokenKind::EOF, 
            hideset: HashSet::new(), 
            pos: TextPosition { line: 0, col: 0, filename: filename.and_then(|s| Some(s.to_string())) }, //TODO: add option to add filename
            has_space: false, 
            at_bol: false }
    }

    pub fn is_in_hideset(&self) -> bool {
        self.hideset.contains(self.get_name())
    }

    pub fn get_name(&self) -> &str {
        match &self.token_type {
            TokenKind::Ident(i) => i,
            _ => &""
        }
    }
}

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
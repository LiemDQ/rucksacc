use crate::utils::TextPosition;
use std::collections::HashSet;

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

#[derive(PartialEq, Debug, Clone)]
pub enum Punct {
    Any,
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


/// TODO: consider splitting out the punctuation and keywords into the actual enum as
/// it is leading to unnecessary layering right now
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Ident(String), // identifiers
    Punct(Punct), // punctuators
    Keyword(Keyword), // keywords
    Str(String), //string literals
    Char(char), //char literals
    Int(i64), //numeric literals
    Float(f64), //float literals
    PPNum, //preprocessor numbers
    EOF, //end of file
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
    pub fn new(ttype: TokenKind, has_space: bool, line: usize, col: usize, filename: Option<&str>) -> Self {
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
}
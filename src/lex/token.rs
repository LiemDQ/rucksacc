use crate::utils::TextPosition;
use std::collections::HashSet;
use std::fmt::Display;

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
            TokenKind::Int(val) => {val.to_string()},
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
    pub fn new(ttype: TokenKind, has_space: bool, line: usize, col: usize, filename: Option<&str>) -> Self {
        Self {
            token_type: ttype,
            at_bol: false,
            has_space: has_space,
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
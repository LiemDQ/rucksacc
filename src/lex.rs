use std::collections::{HashSet};
use std::error::Error;
use std::fs;
use std::fs::{File};
use std::str::Chars;
use std::cell::Cell;

use crate::utils::{TextPosition, PeekNIterator, PeekN};
use crate::err::{ParseRes, ParseErr, ParseErrMsg};

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
    pub fn new(ttype: TokenKind, has_space: bool, line: usize, col: usize, filename: &str) -> Self {
        Self {
            token_type: ttype,
            at_bol: false,
            has_space: false,
            hideset: HashSet::new(),
            pos: TextPosition::new(line, col, Some(filename)),
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

/// Replaces \r or \r\n with \n.
fn canonicalize_newline(contents : &mut str) -> String {
    let mut iter = contents.chars().peekable_n();
    let mut processed  = String::new();

    while let Some(c) = iter.next() {
        if c == '\r' && iter.peek().unwrap_or(&'0') == &'\n' {
            processed.push('\n');
            iter.nth(0);
        } else if c == '\r' {
            processed.push('\n');
        } else {
            processed.push(c);
        }
    }
    processed
}

/// Removes backslashes followed by a newline
fn remove_backslash_newline(contents : &mut str) {
    let mut iter = contents.chars().peekable();
    let mut processed = String::new();
    let mut n = 0;

    while let Some(c) = iter.next() {
        if c == '\\' && iter.peek().unwrap_or(&'0') == &'\n' {
            todo!()
        }
    }
}

/// Contains the state of the lexer. 
/// TODO: one possible alternative implementation is to split the varying state (such as the current line and current column 
/// into a separate `LexerState` struct have have the lexer merely contain immutable things like the filename and body. This could 
/// facilitate reasoning about the code and ease borrow checker restrictions.
struct Lexer<'a> {
    curr_line: usize,
    curr_col: usize,
    prev_line_col: usize,
    body: &'a str
}

/// Contains information about the lexing state, such as the active source file and preprocessor information.
struct LexerState<'a> {
    lexer: Lexer<'a>,
    input_files: Vec<SourceFile>,
    cond_stack : Vec<bool>,
    active: usize,
}

/// Contains information about the source files being lexed. 
struct SourceFile {
    filename: String,
    file: File,
    body: String,
    tokens: Vec<Token>,
}

impl SourceFile {
    pub fn new(filename: &str) -> Self {
        Self {
            filename: filename.to_string(),
            file: File::open(filename).unwrap(),
            body: fs::read_to_string(filename).unwrap(),
            tokens: Vec::new(),
        }
    }
}

impl<'a> LexerState<'a> {
    pub fn new() -> Self {
        Self { lexer: Lexer::new(&""), input_files: Vec::new(), cond_stack: Vec::new(), active: 0 }
    }

    pub fn add_file(&mut self, filename: &str) {
        self.input_files.push(SourceFile::new(filename));
    }
}

impl<'a> Lexer<'a>{

    pub fn new(body: &'a str) -> Self {
        Lexer { 
            curr_line: 0, 
            curr_col: 0, 
            prev_line_col: 0,
            body: body,
        }
    }

    
    pub fn tokenize_file(&mut self, filepath: &str) -> Result<Vec<Token>, Box<dyn Error>> {
        let contents = fs::read_to_string(filepath)?;
    
        self.tokenize(contents)
    }

    fn is_punctuator(&self, c: char, n: usize) -> bool {
        if c.is_ascii_punctuation() {
            return true;
        }
        let iter = self.body.chars().skip(n);
        false
    }

    fn get_symbol(&self, n: usize, has_space: bool) -> Punct {
        let mut iter = self.body.chars().skip(n).peekable();
        let view = &self.body[n..];
        let c = iter.next().unwrap();
        match c {
            '[' => Punct::OpenBoxBracket,
            ']' => Punct::CloseBoxBracket,
            '{' => Punct::OpenBrace,
            '}' => Punct::CloseBrace,
            '(' => Punct::OpenParen,
            ')' => Punct::CloseParen,
            _ => Punct::Add(PMKind::Undet),
        }
    }

    fn read_identifier_or_keyword(&mut self, iter: &mut impl Iterator<Item = char>) -> ParseRes<TokenKind> {
        let mut identifier = String::new();
        
        for c in self.advance(iter) {
            match c {
                'a'..='z' | 'A'..='Z' | '_' | '0'..='9' => identifier.push(c),
                _ => break,
            };
        }

        if let keyword = self.get_keyword_from_string(identifier.as_str())? {
            Ok(TokenKind::Keyword(keyword))
        } else {
            Ok(TokenKind::Ident(identifier))
        }
    }

    /// Advance an iterator, while also incrementing the active position in the lexer. This should be called 
    /// in lieu of calling `iter.next()` to ensure that the data remains consistent.
    #[inline]
    fn advance(&mut self, iter: &mut impl Iterator<Item = char>) -> Option<char> {
        match iter.next() {
            Some(c) if c == '\n' => {
                self.prev_line_col += self.curr_col;
                self.curr_col = 0;
                self.curr_line += 1;
                Some(c)
            },
            Some(c) => {
                self.curr_col += 1;
                Some(c)
            }
            None => None //end of file
        }
    }

    /// Skip ahead by `n` values, discarding the elements while updating the internal position. This is useful in situations
    /// where a parsing function has successfully parsed a multi-char token, and now needs to update the iterator before terminating.
    #[inline]
    fn skip_n(&mut self, iter: &mut impl Iterator<Item = char>, n: usize) {
        for _ in 0..n {
            let _ = self.advance(iter);
        }
    }

    fn read_numeric_literal(&mut self, iter: &mut impl Iterator<Item = char>) -> ParseRes<TokenKind> {
        let mut num = 0;

        while let Some(c) = self.advance(iter){
            match c {
                '0'..='9' => num = num*10+c.to_digit(10).unwrap() as i64,
                '.' => todo!(), //TODO: handle float values
                _ => break,
            }
        }
        Ok(TokenKind::Int(num))
    }

    /// See section 6.4.1 for a list of all keywords in the C language. 
    fn get_keyword_from_string(&self, identifier: &str) -> ParseRes<Keyword> {
        let kw = match identifier {
            "auto" => Keyword::Auto,
            "break" => Keyword::Break,
            "case" => Keyword::Case,
            "const" => Keyword::Const,
            "continue" => Keyword::Continue,
            "default" => Keyword::Default,
            "do" => Keyword::Do,
            "double" => Keyword::Double,
            "else" => Keyword::Else,
            "enum" => Keyword::Enum,
            "extern" => Keyword::Extern,
            "float" => Keyword::Float,
            "for" => Keyword::For,
            "goto" => Keyword::Goto,
            "if" => Keyword::If,
            "inline" => Keyword::Inline,
            "int" => Keyword::Int,
            "long" => Keyword::Long,
            "register" => Keyword::Register,
            "restrict" => Keyword::Restrict,
            "return" => Keyword::Return,
            "short" => Keyword::Short,
            "signed" => Keyword::Signed,
            "sizeof" => Keyword::Sizeof,
            "static" => Keyword::Static,
            "struct" => Keyword::Struct,
            "typedef" => Keyword::Typedef,
            "union" => Keyword::Union,
            "unsigned" => Keyword::Unsigned,
            "void" => Keyword::Void,
            "while" => Keyword::While,
            "_Alignas" => Keyword::AlignAs,
            "_Alignof" => Keyword::AlignOf,
            "_Atomic" => Keyword::Atomic,
            "_Bool" => Keyword::Bool,
            "_Complex" => Keyword::Complex, //not supported
            "_Generic" => Keyword::Generic,
            "_Noreturn" => Keyword::NoReturn,
            "_Static_assert" => Keyword::StaticAssert,
            "_Thread_local" => Keyword::ThreadLocal,
            _ => Keyword::INVALID,
        };
        
        if kw != Keyword::INVALID {
            Ok(kw)
        } else {
            Err(self.gen_parse_err(ParseErrMsg::Something))
        }
    }

    fn gen_parse_err(&self, msg: ParseErrMsg) -> ParseErr {
        let offset = 1; //placeholder for now
        ParseErr::new(
            TextPosition { line: self.curr_line, col: self.curr_col, filename: Some(self.filename.to_string()) }, 
            TextPosition {line: self.curr_line, col: self.curr_col + offset, filename: Some(self.filename.to_string())}, 
            msg,
        )
    }

    fn read_string_literal(&mut self, iter: &mut impl Iterator<Item = char>) -> ParseRes<TokenKind> {
        let mut literal = String::new();
        let mut iter = iter.peekable_n();
        while let Some(c) = self.advance(&mut iter) {
            match c {
                '"' => break,
                '\\' => literal.push(Lexer::read_escaped_char(&mut iter).unwrap()),
                _ => literal.push(c),
            }
        }

        Ok(TokenKind::Str(literal))
    }

    fn read_char_literal(&self, iter: &mut impl Iterator<Item = char>) -> ParseRes<TokenKind> {
        let mut iter = iter.peekable_n();
        let c = iter.peek();
        let &c = if c.is_some() {
            c.unwrap()
        } else {
            return Err(self.gen_parse_err(ParseErrMsg::Something));
        };
        let c = {
            if c == '\\' {
                Lexer::read_escaped_char(&mut iter).unwrap()
            } else {
                c
            }
        };
        if iter.peek_next().unwrap() != &'\'' {
            self.gen_parse_err(ParseErrMsg::Something);
        }

        Ok(TokenKind::Char(c))

    }

    /// TODO: it's not clear what escaped characters the chars() iterator recognizes, and which ones it doesn't. 
    /// Need to investigate this to determine which characters need to be explicitly handled, and which ones can
    /// fall through to the usual functions.
    /// It may be worth revisiting whether chars() is the correct abstraction and whether
    /// we should use a raw bytestream instead, and parse all of the escape characters manually. 
    fn read_escaped_char(iter: &mut impl Iterator<Item = char>) -> Option<char> {
        let mut iter = iter.peekable_n();
        let &c = if iter.peek().is_some() {
            iter.peek().unwrap()
        } else {
            return None;
        };

        
        let ret = match c {
            '\'' | '"' | '?' | '\\' => c,
            'a' => '\x07',
            'b' => '\x08',
            'f' => '\x0c',
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            'v' => '\x0b',
            'x' => {todo!()}, //handle hex numbers. Hex numbers must have exactly two digits after the escape character.
            '0'..='7' => {
                todo!("handle octal numbers") //handle octal numbers
            }
            _ => c,

        };

        Some(ret)
    }

    fn read_symbol(&mut self, iter: &mut impl Iterator<Item = char>) -> ParseRes<TokenKind> {
        let mut iter = iter.peekable_n();
        let symbol = match self.advance(&mut iter).unwrap_or(' ') {
            '[' => Punct::OpenBoxBracket,
            ']' => Punct::CloseBoxBracket,
            '{' => Punct::OpenBrace,
            '}' => Punct::CloseBrace,
            '(' => Punct::OpenParen,
            ')' => Punct::CloseParen,
            ';' => Punct::Semicolon,
            '?' => Punct::Question,
            ':' => Punct::Colon,
            '~' => Punct::BitwiseNot,
            ',' => Punct::Comma,
            '!' => {
                match iter.peek().unwrap_or(&' ') {
                    '=' => Punct::Ne,
                    _ => Punct::Not,
                }
            },
            '&' => {
                match iter.peek().unwrap_or(&' ') {
                    '&' => Punct::LAnd,
                    '=' => Punct::AssignAnd,
                    _   => Punct::Ampersand(AmpKind::Undet)
                }
            },
            '=' => {
                match iter.peek().unwrap_or(&' '){
                    '=' => Punct::Eq,
                    _   => Punct::Assign,
                }
            },
            '*' => {
                match iter.peek().unwrap_or(&' '){
                    '=' => Punct::AssignMul,
                    //asterisks can be ambiguous in terms of what they mean. For example, `T** b;` can be either
                    //`(T)*(*b);` which multiplies `T` by the deference of `b`, or `(T**) b;` which declares a pointer-to-pointer `b` with underlying
                    // base type `T`. 
                    _   => Punct::Asterisk(AstKind::Undet),
                }
            },
            '+' => {
                match iter.peek().unwrap_or(&' '){
                    //NOTE: the ordering of this match statement resolves an ambiguity in the C grammar in a specific way. 
                    //`a+++b` can be parsed as either `(a++)+b` or `a+(++b)`, which would result in a different evaluation. With this
                    //implementation, we opt for the former over the latter. 
                    //TODO: something like `a+++++b` can only be parsed as `(a++) + (++b)` which currently will fail to be parsed 
                    //as it will be tokenized as Inc Inc Add which will fail at the parsing step.
                    //this may not be a huge issue as both clang and gcc are incapable of parsing this anyway.
                    '+' => Punct::Inc, 
                    '=' => Punct::AssignAdd,
                    _   => Punct::Add(PMKind::Undet),
                }
            },
            '-' => {
                match iter.peek().unwrap_or(&' '){
                    '-' => Punct::Dec,
                    '=' => Punct::AssignSub,
                    '>' => Punct::Arrow,
                    _   => Punct::Sub(PMKind::Undet)
                }
            },
            '%' => {
                match iter.peek().unwrap_or(&' '){
                    '=' => Punct::AssignMod,
                    _   => Punct::Mod,
                }
            },
            '^' => {
                match iter.peek().unwrap_or(&' ') {
                    '=' => Punct::AssignXor,
                    _ => Punct::Xor,
                }
            },
            '>' => {
                match iter.peek().unwrap_or(&' '){
                    '>' => match iter.peek_nth(1).unwrap_or(&' '){
                        '=' => Punct::AssignShr,
                        _ => Punct::Shr,
                    }
                    '=' => Punct::Ge,
                    _ => Punct::Gt,
                }
            },
            '<' => {
                match iter.peek().unwrap_or(&' '){
                    '<' => match iter.peek_nth(1).unwrap_or(&' '){
                        '=' => Punct::AssignShl,
                        _ => Punct::Shl,
                    }
                    '=' => Punct::Le,
                    _ => Punct::Lt,
                }
            },
            '|' => {
                match iter.peek().unwrap_or(&' '){
                    '|' => Punct::LOr,
                    '=' => Punct::AssignOr,
                    _ => Punct::Or
                }
            },
            '/' => {
                match iter.peek().unwrap_or(&' '){
                    '=' => Punct::AssignDiv,
                    _ => Punct::Div,
                }
            },
            '.' => {
                match iter.peek().unwrap_or(&' '){
                    '0'..='9' => return Err(self.gen_parse_err(ParseErrMsg::Something)), //is a float, this should never happen
                    '.' => if iter.peek_nth(1).unwrap_or(&' ') == &'.' {
                            Punct::Vararg
                        } else {
                            return Err(self.gen_parse_err(ParseErrMsg::Something))
                        }
                    _ => Punct::Point,
                }
            },
            '#' => {
                match iter.peek().unwrap_or(&' '){
                    '#' => Punct::PoundPound,
                    _ => Punct::Hash,
                }
            }
            _ => return Err(self.gen_parse_err(ParseErrMsg::Something)),
        };

        Ok(TokenKind::Punct(symbol))
    }

    /// Process a C source file into a vector of tokens. 
    pub fn tokenize(&mut self, body: String) -> Result<Vec<Token>, Box<dyn Error>> {
        
        
        let mut iter = body.chars().peekable_n();
        let mut has_space = false;
        let mut tokens = Vec::<Token>::new();
        while let Some(&c) = iter.peek() {
            
            //skip comments
            if iter.starts_with("//".chars()) {
                let mut end = c;
                while end != '\n' {
                    match self.advance(&mut iter) {
                        Some(e) => end = e,
                        None => break,
                    } //consume until end of line
                }
                continue;
            }
            if iter.starts_with("/*".chars()) {
                while !iter.starts_with("*/".chars()){
                    iter.next();
                }
                continue;
            }

            //eat whitespace
            if c.is_whitespace() {
                has_space = true;
                self.advance(&mut iter);
                continue;
            }

            let tok = match c {
                'a'..='z' | 'A'..='Z' | '_' => self.read_identifier_or_keyword(&mut iter), //read identifier
                '0'..='9' => self.read_numeric_literal(&mut iter), //number literal
                '\"' => self.read_string_literal(&mut iter), //string literal
                '\'' => self.read_char_literal(&mut iter), //character literal
                _ => self.read_symbol(&mut iter),
            }?;

            tokens.push(
                Token::new(tok, has_space, self.curr_line, self.curr_col, &self.filename)
            );
            has_space = false;
        }
    
        Ok(tokens)
    }
}


#[test]
fn read_identifier_drops_non_alphanumeric() {
    let line = "my_variable2 = 3;";
    let mut iter = line.chars();
    let token = Lexer::read_identifier_or_keyword(&mut iter).unwrap();
    
    assert_eq!(token, TokenKind::Ident("my_variable2".to_string()));
}

#[test]
fn read_symbol_outputs_correctly() {
    let s1 = ">>= 697";
    let mut iter = s1.chars();
    assert_eq!(Lexer::read_symbol(&mut iter).unwrap(),TokenKind::Punct(Punct::AssignShr));
}
mod preprocess; 

pub mod token;
pub mod lexer;
pub mod token_err;
pub mod iter;

pub use token::{Token, TokenKind, Keyword, Punct, AstKind, PMKind, AmpKind, Bits};
use crate::err::{ParseErr, ParseErrMsg, ParseRes, IOErr};
use crate::utils::{PeekNIterator, PeekN, TextPosition};
use std::fs::File;
use std::{fs};
use std::error::Error;


/// Contains the state of the lexer. 
/// TODO: one possible alternative implementation is to split the varying state (such as the current line and current column 
/// into a separate `LexerState` struct have have the lexer merely contain immutable things like the filename and body. This could 
/// facilitate reasoning about the code and ease borrow checker restrictions.
struct LexerState {
    curr_line: usize,
    curr_col: usize,
    prev_line_col: usize,
    source_file_index: usize,
}

/// Contains all lexing data, such as the active source file and preprocessor information.
struct Lexer {
    state: LexerState,
    input_files: Vec<SourceFile>,
    cond_stack : Vec<bool>,
}

impl LexerState {

    pub fn new(index: usize) -> Self {
        LexerState { 
            curr_line: 0, 
            curr_col: 0, 
            prev_line_col: 0,
            source_file_index: index,
        }
    }
    

    pub fn set_source_file(&mut self, index: usize) {
        self.curr_line = 0;
        self.curr_col = 0;
        self.prev_line_col = 0;
        self.source_file_index = index; 
    }
}

/// Contains information about the source files. 
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

impl Lexer {
    pub fn new() -> Self {
        Self { state: LexerState::new(0), input_files: Vec::new(), cond_stack: Vec::new() }
    }

    pub fn add_file(&mut self, filename: &str) {
        self.input_files.push(SourceFile::new(filename));
    }

    fn get_current_filename(&self) -> Option<&str> {
        self.input_files.get(self.state.source_file_index).map(|file| file.filename.as_str())
    }

    /// Advance an iterator, while also incrementing the active position in the lexer. This should be called 
    /// in lieu of calling `iter.next()` to ensure that the data remains consistent.
    #[inline]
    fn advance(&mut self, iter: &mut impl Iterator<Item = char>) -> Option<char> {
        match iter.next() {
            Some(c) if c == '\n' => {
                self.state.prev_line_col += self.state.curr_col;
                self.state.curr_col = 0;
                self.state.curr_line += 1;
                Some(c)
            },
            Some(c) => {
                self.state.curr_col += 1;
                Some(c)
            }
            None => None //end of file
        }
    }


    pub fn tokenize_inputs(&mut self) -> Result<Vec<Vec<Token>>, Box<dyn Error>> {
        if self.input_files.is_empty() {
            return Err(Box::new(IOErr::NoFilesSpecified));
        }

        let mut tokens = Vec::new();
        
        let mut i = self.input_files.len() - 1; 
        while let Some(file) = self.input_files.pop() {
            self.state.set_source_file(i);
            tokens.push(self.tokenize(&file.body)?);
            i -= 1;
        }

        Ok(tokens)
    }

    fn read_identifier_or_keyword<I: Iterator<Item = char>>(&mut self, iter: &mut PeekNIterator<I>) -> ParseRes<TokenKind> {
        let mut identifier = String::new();
        
        while let Some(&c) = iter.peek() {
            match c {
                'a'..='z' | 'A'..='Z' | '_' | '0'..='9' =>{
                    identifier.push(c);
                    self.advance(iter);
                }
                _ => break,
            };
        }

        if let Ok(keyword) = self.get_keyword_from_string(identifier.as_str()) {
            Ok(TokenKind::Keyword(keyword))
        } else {
            Ok(TokenKind::Ident(identifier))
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

    fn read_numeric_literal<I: Iterator<Item = char>>(&mut self, iter: &mut PeekNIterator<I>) -> ParseRes<TokenKind> {
        let mut num = 0;

        while let Some(c) = iter.peek(){
            match c {
                '0'..='9' => {
                    num = num*10+c.to_digit(10).unwrap() as i64;
                    self.advance(iter);
                },
                '.' => todo!(), //TODO: handle float values
                _ => break,
            }
        }
        Ok(TokenKind::Int(num, Bits::Bits64))
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
            Err(self.gen_parse_err(ParseErrMsg::UnrecognizedKeyword))
        }
    }

    fn gen_parse_err(&self, msg: ParseErrMsg) -> ParseErr {
        let offset = 1; //placeholder for now
        ParseErr::new(
            TextPosition { line: self.state.curr_line, col: self.state.curr_col, filename: self.get_current_filename().map(|s| s.to_string()) }, 
            TextPosition {line: self.state.curr_line, col: self.state.curr_col + offset, filename: self.get_current_filename().map(|s| s.to_string())}, 
            msg,
        )
    }

    fn read_string_literal<I: Iterator<Item = char>>(&mut self, iter: &mut PeekNIterator<I>) -> ParseRes<TokenKind> {
        let mut literal = String::new();
        while let Some(c) = self.advance(iter) {
            match c {
                '"' => break,
                '\\' => literal.push(Lexer::read_escaped_char(iter).unwrap()),
                _ => literal.push(c),
            }
        }

        Ok(TokenKind::Str(literal))
    }

    fn read_char_literal<I: Iterator<Item = char>>(&self, iter: &mut PeekNIterator<I>) -> ParseRes<TokenKind> {
        let c = iter.peek();
        let &c = if c.is_some() {
            c.unwrap()
        } else {
            return Err(self.gen_parse_err(ParseErrMsg::EOF));
        };
        let c = {
            if c == '\\' {
                Lexer::read_escaped_char(iter).unwrap()
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
    fn read_escaped_char<I: Iterator<Item = char>>(iter: &mut PeekNIterator<I>) -> Option<char> {
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

    fn read_symbol<I: Iterator<Item = char>>(&mut self, iter: &mut PeekNIterator<I>) -> ParseRes<TokenKind> {
        let symbol = match self.advance(iter).unwrap_or(' ') {
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
            '\\'=> Punct::Backslash,
            '!' => {
                match iter.peek().unwrap_or(&' ') {
                    '=' => {self.advance(iter); Punct::Ne},
                    _ => Punct::Not,
                }
            },
            '&' => {
                match iter.peek().unwrap_or(&' ') {
                    '&' => {self.advance(iter); Punct::LAnd},
                    '=' => {self.advance(iter); Punct::AssignAnd},
                    _   => Punct::Ampersand(AmpKind::Undet)
                }
            },
            '=' => {
                match iter.peek().unwrap_or(&' '){
                    '=' => {self.advance(iter); Punct::Eq},
                    _   => Punct::Assign,
                }
            },
            '*' => {
                match iter.peek().unwrap_or(&' '){
                    '=' => {self.advance(iter); Punct::AssignMul},
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
                    '+' => {self.advance(iter); Punct::Inc}, 
                    '=' => {self.advance(iter); Punct::AssignAdd},
                    _  => Punct::Add(PMKind::Undet),
                }
            },
            '-' => {
                match iter.peek().unwrap_or(&' '){
                    '-' => {self.advance(iter); Punct::Dec},
                    '=' => {self.advance(iter); Punct::AssignSub},
                    '>' => {self.advance(iter); Punct::Arrow},
                    _   => Punct::Sub(PMKind::Undet)
                }
            },
            '%' => {
                match iter.peek().unwrap_or(&' '){
                    '=' => {self.advance(iter); Punct::AssignMod},
                    _   => Punct::Mod,
                }
            },
            '^' => {
                match iter.peek().unwrap_or(&' ') {
                    '=' => {self.advance(iter); Punct::AssignXor},
                    _ => Punct::Xor,
                }
            },
            '>' => {
                match iter.peek().unwrap_or(&' '){
                    '>' => {
                        self.advance(iter);
                        match iter.peek().unwrap_or(&' '){
                            '=' => {self.advance(iter); Punct::AssignShr},
                            _ => Punct::Shr,
                        }
                    },
                    '=' => {self.advance(iter); Punct::Ge},
                    _ => Punct::Gt,
                }
            },
            '<' => {
                match iter.peek().unwrap_or(&' '){
                    '<' => match iter.peek().unwrap_or(&' '){
                        '=' => {self.advance(iter); Punct::AssignShl},
                        _ => Punct::Shl,
                    }
                    '=' => {self.advance(iter); Punct::Le},
                    _ => Punct::Lt,
                }
            },
            '|' => {
                match iter.peek().unwrap_or(&' '){
                    '|' => {self.advance(iter); Punct::LOr},
                    '=' => {self.advance(iter); Punct::AssignOr},
                    _ => Punct::Or
                }
            },
            '/' => {
                match iter.peek().unwrap_or(&' '){
                    '=' => {self.advance(iter); Punct::AssignDiv},
                    _ => Punct::Div,
                }
            },
            '.' => {
                match iter.peek().unwrap_or(&' '){
                    '0'..='9' => return Err(self.gen_parse_err(ParseErrMsg::InternalError(line!()))), //is a float, this should never happen
                    '.' => if iter.peek_nth(1).unwrap_or(&' ') == &'.' {
                            Punct::Vararg
                        } else {
                            return Err(self.gen_parse_err(ParseErrMsg::InvalidSymbol))
                        }
                    _ => Punct::Point,
                }
            },
            '#' => {
                match iter.peek().unwrap_or(&' '){
                    '#' => {self.advance(iter); Punct::PoundPound},
                    _ => Punct::Hash,
                }
            }
            _ => return Err(self.gen_parse_err(ParseErrMsg::InvalidSymbol)),
        };

        Ok(TokenKind::Punct(symbol))
    }

    /// Process a C source file into a vector of tokens. 
    pub fn tokenize(&mut self, body: &str) -> Result<Vec<Token>, Box<dyn Error>> {
        
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
                Token::new(tok, self.state.curr_line, self.state.curr_col, self.get_current_filename())
            );
            has_space = false;
        }
    
        Ok(tokens)
    }
}

#[cfg(test)]
mod test {
    use crate::lex::token::Bits;

    use super::*;
    use std::fs;

    fn tokenstream_to_C_code(tokens: Vec<Token>) -> String {
        tokens.into_iter().map(|token| format!("{}", token.token_type)).collect::<String>()
    }

    #[test]
    fn tokenize_assignment() {
        let line = "my_variable2 = 3;";

        let tokens = Lexer::new().tokenize(line).unwrap();
        assert_eq!(tokens.first().unwrap().token_type, TokenKind::Ident("my_variable2".to_string()));
        // assert_eq!(tokens.len(),4);
        assert_eq!(tokens.into_iter().map(|tk| tk.token_type).collect::<Vec<TokenKind>>(), 
            vec![TokenKind::Ident("my_variable2".to_string()), 
                TokenKind::Punct(Punct::Assign), 
                TokenKind::Int(3, Bits::Bits32), 
                TokenKind::Punct(Punct::Semicolon)]);
    }

    #[test]
    fn read_symbol_outputs_correctly() {
        let s1 = ">>= 697";
        let mut lex = Lexer::new();
        let mut iter = s1.chars().peekable_n();
        assert_eq!(lex.read_symbol(&mut iter).unwrap(),TokenKind::Punct(Punct::AssignShr));
    }

    #[test]
    fn tokenize_function() {
        let funcdef = "int func(){\n\
            int x = 0;\n\
            for(int i = 0; i < 4; i++){\n\
                x++;\n\
            }\n\
            return x;\n\
        }";
        let mut lex = Lexer::new();
        let tokens = lex.tokenize(funcdef).unwrap();
        assert_eq!(tokens.into_iter().map(|tk| tk.token_type).collect::<Vec<TokenKind>>(),
            vec![
                //int func(){
                TokenKind::Keyword(Keyword::Int),
                TokenKind::Ident("func".to_string()),
                TokenKind::Punct(Punct::OpenParen),
                TokenKind::Punct(Punct::CloseParen),
                TokenKind::Punct(Punct::OpenBrace),
                //int x = 0;
                TokenKind::Keyword(Keyword::Int),
                TokenKind::Ident("x".to_string()),
                TokenKind::Punct(Punct::Assign),
                TokenKind::Int(0, Bits::Bits32),
                TokenKind::Punct(Punct::Semicolon),
                //for(int i = 0; i < 4; i++)
                TokenKind::Keyword(Keyword::For),
                TokenKind::Punct(Punct::OpenParen),
                TokenKind::Keyword(Keyword::Int),
                TokenKind::Ident("i".to_string()),
                TokenKind::Punct(Punct::Assign),
                TokenKind::Int(0, Bits::Bits32),
                TokenKind::Punct(Punct::Semicolon),
                TokenKind::Ident("i".to_string()),
                TokenKind::Punct(Punct::Lt),
                TokenKind::Int(4, Bits::Bits32),
                TokenKind::Punct(Punct::Semicolon),
                TokenKind::Ident("i".to_string()),
                TokenKind::Punct(Punct::Inc),
                TokenKind::Punct(Punct::CloseParen),
                //{ x++; }
                TokenKind::Punct(Punct::OpenBrace),
                TokenKind::Ident("x".to_string()),
                TokenKind::Punct(Punct::Inc),
                TokenKind::Punct(Punct::Semicolon),
                TokenKind::Punct(Punct::CloseBrace),
                //return x;
                TokenKind::Keyword(Keyword::Return),
                TokenKind::Ident("x".to_string()),
                TokenKind::Punct(Punct::Semicolon),
                //}
                TokenKind::Punct(Punct::CloseBrace)
            ]
        );
    }
    
    #[test]
    fn compare_tokenization_roundtrip(){
        let code = "void main(int argc, char** argv){\n\
            int x = 0;\n\
            int y = 1;\n\
            x >>= y & 2 | 3;\n\
            int arr[] = {1,2,3,4,5};\n\
            int z = arr[1];\n\
            return 0;\n\
        }";
        let mut lex = Lexer::new();
        let tokens = lex.tokenize(code).unwrap();
        let code_processed = code.to_string().split_ascii_whitespace().collect::<String>();
        assert_eq!(code_processed, tokenstream_to_C_code(tokens));
    }

}

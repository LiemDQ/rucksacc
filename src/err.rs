use ariadne::{Report, ReportKind, Label, Source};
use crate::utils::TextPosition;
use crate::lex::TokenKind;
use std::fmt;

/// Generic error type used throughout the compiler. 
/// The `TextPositions` indicate where they occur in the text.
#[derive(Debug, Clone)]
pub struct ParseErr {
    position : TextPosition,
    position_end : TextPosition,
    message: ParseErrMsg,    
}

/// Errors for the lexer and parser. 
/// TODO: set up useful error messages. 
#[derive(Debug, PartialEq, Clone)]
pub enum ParseErrMsg {
    Something,
    ExpectedSymbol(TokenKind),
    ExpectedOneOfSymbols(Vec<TokenKind>),
    EOF,
    InvalidFloatOperation,
    NotCompileTimeConstant,
    NotConstExpr,
    InvalidSizeof,
    InvalidExpressionToken,
    InternalError(u32), //line number
    UnknownIdentifier,
    InvalidFunctionCall,
    TooManyArguments,
    InvalidStructReference,
    UnknownTypename,
    TypeError(String),
}

impl ParseErrMsg {
    pub fn print_message(&self) -> &str {
        match self {
            ParseErrMsg::Something => "Something happened",
            ParseErrMsg::ExpectedSymbol(tk) => &format!("Expected {:?}", tk),
            ParseErrMsg::ExpectedOneOfSymbols(v) => &format!("Expected one of: {:?}", v),
            ParseErrMsg::InvalidFloatOperation => "Invalid floating-point operation",
            ParseErrMsg::InvalidSizeof => "Invalid object for sizeof call",
            ParseErrMsg::NotCompileTimeConstant => "Not a compile time constant",
            ParseErrMsg::NotConstExpr => "Not a valid constant expression",
            ParseErrMsg::InternalError(line) => &format!("Internal compiler error at line {}", line),
            ParseErrMsg::EOF => "End of file reached",
            _ => "Unimplemented error",
        }
    }
}

pub type ParseRes<T> = Result<T, ParseErr>;

impl ParseErr {
    pub fn new(pos: TextPosition, pos_end: TextPosition, msg : ParseErrMsg) -> Self {
        ParseErr { position: pos, position_end: pos_end, message: msg, }
    }

    pub fn report_single(&self) {
        if let Some(filename) = self.position.filename {
            Report::build(ReportKind::Error, &filename, self.position.col)
                .with_message(format!("{:?}", self.message))
                .with_label(Label::new((&filename, self.position.col..self.position_end.col)))
                .finish()
                .eprint((&filename, Source::from(std::fs::read_to_string(&filename).unwrap())))
                .unwrap();
        } else {
            println!("Error: no filename found.");
        }
    }
}

impl std::error::Error for ParseErr {}

impl fmt::Display for ParseErr {
    fn fmt(&self, f:  &mut fmt::Formatter) -> fmt::Result {
        match self.position.filename {
            Some(s) => write!(f, "Error in {} at {:?}, message: \n{:?}", s, self.position, self.message),
            None => write!(f, "Error at {:?}, message: \n{:?}", self.position, self.message),
        }
    }
}


use super::{token::{Token, TokenKind, ensure_and_consume}, Bits, Punct, token_err::{gen_eof_error, gen_expected_error, gen_expected_one_of_error, gen_internal_error, gen_parser_err, gen_compiler_error}};
use super::iter::*;

use crate::{utils::{TextPosition, PeekN, PeekNIterator}, err::ParseErr, lex::Lexer, parse::parser::Parser};
use crate::err::{ParseRes, ParseErrMsg};
use std::{collections::{HashMap, HashSet}};
use std::error::Error;
use ::lazy_static::{self, lazy_static};

lazy_static!{
    static ref linux_include_paths: Vec<&'static str> = vec![
        "./include/",
        "/include/",
        "/usr/include/",
        "/usr/include/linux/",
        "/usr/include/x86_64-linux-gnu/",
        "./include/", 
        "./", //using #include "xxxx" simply means iterating through this list in reverse
    ];
}

pub enum Directive {
    Include,
    Define,
    Undef,
    If,
    Ifdef,
    Ifndef,
    Else,
    Endif,
    Error,
    Pragma,
}

#[derive(Debug, Clone)]
pub enum MacroType {
    ObjLike(Vec<Token>), //body
    FuncLike(Vec<String>, Vec<Token>), //params, body
}

#[derive(Debug, Clone)]
pub struct Macro {
    name: String,
    mtype: MacroType,
    varargs_name: Option<String>,
}

impl Macro {
    pub fn new(name: String, mtype: MacroType) -> Self {
        Self {
            name,
            mtype,
            varargs_name: None,
        }
    }
}

pub enum PreprocessToken {
    HeaderName(String),
    Identifier(String),
    Number(i64), //TODO determine appropriate type for this
    CharConst(char), //character constant
    StrLiteral(String), //string literal
    Punct, //punctuator
}

/// if/elses can be nested, so they are represented with a stack.
struct ConditionalIncl {
    included: bool,
    body: Vec<Token>,
    in_context: ConditionalContext,
}

enum ConditionalContext {
    Then,
    Elif,
    Else,
}

struct Preprocessor {
    macro_map: HashMap<String, Macro>,
    cond_stack: Vec<ConditionalIncl>,
}


impl Preprocessor {
    pub fn new() -> Self {
        Self { macro_map: HashMap::default(), cond_stack: Vec::new() }
    }

    pub fn preprocess(&mut self, body: Vec<Token>) -> ParseRes<Vec<Token>> {
        let mut result = Vec::new();
        let mut has_expanded = false;
        loop {
            let mut iter = body.into_iter();
            while let Some(token) = iter.next()  {
                match token.token_type {
                    //#
                    TokenKind::Punct(Punct::Hash) => {
                        if let Some(t) = iter.next() {
                            match t.token_type {
                                TokenKind::Ident(ref directive) => {
                                    self.parse_directive(&mut iter, directive);
                                },
                                _ => gen_expected_error(&t, TokenKind::Ident("".to_string()))?,
                            };
                        } else {
                            gen_eof_error()?;
                        }
                    },
                    //##
                    //TODO
                    TokenKind::Punct(Punct::PoundPound) => {
                        
                    }
                    TokenKind::Ident(_) => {
                        expand(&self.macro_map, token);
                        has_expanded = true;
                    },
                    _ => {},
                }
            }
            //preprocessor will stop once there are no macros left to expand. 
            if !has_expanded {
                break;
            } else {
                has_expanded = false;
            }
        }
        Ok(result)
    }

    fn parse_directive(&mut self, iter: &mut impl Iterator<Item = Token>, directive: &String) -> ParseRes<()> {
        match directive.as_str() {
            "include" => {
                let (local_first, filename) = Preprocessor::parse_include_filename(iter)?;
                let filepath = Preprocessor::get_filepath(&filename, local_first);

            },
            "define" => {
                let name_token = consume_token_or_eof(iter)?;
                match name_token.token_type {
                    TokenKind::Ident(ref name) =>{
                        self.add_macro(name.to_string(), body);
                    },
                    _ => return gen_expected_error(&name_token, TokenKind::Ident("()".to_string())),
                }
            },
            "if" => {
                let ctx = ConditionalContext::Then;
                //special case: if the following sequence is `defined()`
                let cond = self.eval_const_expr(iter)?;
                
                if cond != 0 {
                    //how should we handle this? we can achieve this through recursive function calls
                    //or we can handle the loop internally.
                    //the latter case should be simpler to implement
                    
                } else {
                    
                }
            },
            "ifdef" => {
                let ctx = ConditionalContext::Then;
                let cond = consume_token_or_eof(iter)?;
                //check if macro has been defined
                if self.macro_map.contains_key(cond.get_name()) {

                }
            },
            "ifndef" => {
                let ctx = ConditionalContext::Then;
                let cond = consume_token_or_eof(iter)?;
                //check if macro has been defined
                if !self.macro_map.contains_key(cond.get_name()) {

                }
            },
            "elif" => {
                let ctx = ConditionalContext::Elif;
                let cond = self.eval_const_expr(iter);
            },
            "else" => {
                let ctx = ConditionalContext::Else;
                
            },
            "endif" => {
                let cond = self.cond_stack.pop();
            },
            "pragma" => {
                let token = consume_token_or_eof(iter)?;
                match token.token_type {
                    TokenKind::Ident(ref ident) => {
                        match ident.as_str() {
                            "once" => {},
                            "pack" => {
                                return gen_compiler_error("pragma pack is currently unsupported", &token.pos);
                            },
                            _ => {
                                return gen_compiler_error("unsupported pragma keyword", &token.pos);
                            },
                        }
                    },
                    _ => return gen_expected_error(&token, TokenKind::Ident("()".to_string())),
                }

            },
            "error" => {
                
                let error_tokens = Vec::new();
                while let Some(token) = iter.next() {
                    if token.at_bol {
                        break;
                    }
                    error_tokens.push(token);
                }
                let err_msg = error_tokens
                    .iter()
                    .map(|tok| format!("{}", tok.token_type))
                    .collect::<String>();
                
                return Err(gen_parser_err(ParseErrMsg::CompilerError(err_msg), &error_tokens[0]));
            },
            "line" => {
                let line_no = consume_token_or_eof(iter)?;
                match line_no.token_type {
                    TokenKind::Int(val,_) => {
                        //all subsequent token text positions must be adjusted.

                    },
                    _ => return gen_expected_error(&line_no, TokenKind::Int(0, Bits::Bits32)),
                }                
            },
            "undef" => {
                let undef = consume_token_or_eof(iter)?;
                self.macro_map.remove(undef.try_get_name()?);
            },
        }
        Ok(())
    }

    fn cond_skip(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<Vec<Token>> {
        let tokens = Vec::new();
        //ensure that nested conditionals are preserved.
        let depth = 0;
        while let Some(token) = iter.next() {
            //look for #elif, #else, #endif
            if token.token_type == TokenKind::Punct(Punct::Hash) {
                tokens.push(value)
                let directive = consume_token_or_eof(iter)?;
                match directive.token_type {
                    TokenKind::Ident(ident) => { match ident.as_str() {
                        "else"| "elif" | "endif" => {},
                        _ => { tokens.push(directive)},
                    }},
                    _ => {tokens.push(directive)},
                }

            }
        }
    }

    fn add_macro(&mut self, name: String, body: &[Token]) -> ParseRes<()> {
        if body.is_empty() {
            //redefinition is just a warning in clang/gcc, not an error. 
            let mtype = MacroType::ObjLike(Vec::new());
            self.macro_map.insert(name, Macro::new(name, mtype));
            Ok(())
        } else {
            if body[0].token_type == TokenKind::Punct(Punct::OpenParen) {
                //func-like macro
                let (begin, args) = Preprocessor::parse_macro_args(body)?;
                let end = Preprocessor::find_macro_end(&body[begin..]);
                let body = body[begin..end]
                    .iter()
                    .cloned()
                    .filter(|token| token.token_type != TokenKind::Punct(Punct::Backslash))
                    .collect();
                let mtype = MacroType::FuncLike(args, body);
                self.macro_map.insert(name, Macro::new(name, mtype));

                Ok(())
            } else {
                //obj-like macro
                let end = Preprocessor::find_macro_end(body);
                //remove backslashes as they are no longer needed
                let body: Vec<Token> = body[0..end]
                    .iter()
                    .cloned()
                    .filter(|token| token.token_type != TokenKind::Punct(Punct::Backslash))
                    .collect();

                let mtype = MacroType::ObjLike(body);
                self.macro_map.insert(name, Macro::new(name, mtype));
                Ok(())                
            }
        }
    }

    fn add_include_file(filepath: &str) -> ParseRes<Vec<Token>> {
        let mut lexer = Lexer::new();
        //TODO: perhaps come up with a better file handling path than this.
        //in this specific case, this read should never fail as we check for the existence
        //of the filepath beforehand.
        let body = std::fs::read_to_string(filepath).expect("Invalid include path specified");
        
        
        let tokens = lexer.tokenize(&body)?;
        let preprocessor = Preprocessor::new();
        //Convert the token stream to an expanded token stream. 
        //This will happen recursively until there is nothing to expand or include.
        preprocessor
            .preprocess(tokens)
            
    }
    
    ///Returns the filepath to include based on the include path and the provided filename.
    ///If no valid filepath could be found, returns None.
    fn get_filepath(filename: &str, local_first: bool) -> Option<String> {
        use std::path::Path;
        if local_first {
            linux_include_paths
            .iter()
            .find(|path| {
                let full_filepath = format!("{}{}", path, filename);
                Path::new(full_filepath.as_str()).exists()
            })
            .and_then(|path| Some(format!("{}{}", path, filename)) )
        } else {
            linux_include_paths
            .iter()
            .rev()
            .find(|path| {
                let full_filepath = format!("{}{}", path, filename);
                Path::new(full_filepath.as_str()).exists()
            })
            .and_then(|path| Some(format!("{}{}", path, filename)) )
        }
    }

    fn parse_include_filename(iter: &mut impl Iterator<Item = Token>) -> ParseRes<(bool, String)> {
        if let Some(ref tok) = iter.next() {
            match tok.token_type {
                TokenKind::Str(filename) => Ok((true, filename)),
                //filename denoted with <__filename__>
                TokenKind::Punct(Punct::Lt) => {
                    let mut filename = String::new();
                    let mut closed = false;
                    while let Some(ref filename_tok) = iter.next() {
                        //parse tokens until the '>' character is reached
                        if filename_tok.token_type != TokenKind::Punct(Punct::Gt) && !filename_tok.at_bol {
                            filename.push_str(&format!("{}", filename_tok.token_type));
                        } else if filename_tok.at_bol {
                            //if there is a newline without completing the filename, throw an error.
                            return gen_expected_error(filename_tok, TokenKind::Punct(Punct::Gt));
                        } else {
                            break;
                        }
                    }
                    Ok((false, filename))
                },
                _ => gen_expected_one_of_error(tok, vec![TokenKind::Punct(Punct::Lt), TokenKind::Str("".to_string())]),
            }
        } else {
            gen_eof_error()
        }
    }
    
    /// For a function-like macro, get a list of macro params, as a vec of strings,
    /// and also indicate in index in the input slice where the body begins.
    fn parse_macro_args(body: &[Token]) -> ParseRes<(usize, Vec<String>)> {
        let args = Vec::new();
        let idx = 0;
        let mut iter = body.iter();
        
        while let Some(token) = iter.next() {
            match &token.token_type {
                TokenKind::Ident(arg) => {
                    args.push(arg.clone());
                    
                    //parameters must be comma separated
                    //TODO: I don't think these parameters can be expanded, but it would be good to check.
                    if let Some(separator) = iter.next() {
                        if separator.token_type != TokenKind::Punct(Punct::Comma) {
                            gen_expected_error(separator, TokenKind::Punct(Punct::Comma))?;
                        }
                    }
                },
                _ => gen_expected_error(token, TokenKind::Ident("()".to_string()))?,
            }
            idx += 1;
        }

        if let Some(token) = &body.get(idx+1) {
            if token.token_type == TokenKind::Punct(Punct::CloseParen) {
                Ok((idx+2, args))
            } else {
                gen_expected_error(token, TokenKind::Punct(Punct::CloseParen))
            }
        } else {
            gen_eof_error()
        }
    }

    /// Find the end of a macro definition, as denoted by a newline. 
    /// Skips newlines preceded by a backslace '\'.
    fn find_macro_end(body: &[Token]) -> usize {
        let mut end_idx = 0;
        let mut backslash_present = false;
        for token in body.iter() {
            end_idx += 1;
            if token.at_bol && !backslash_present {
                break;
            } else if token.at_bol {
                backslash_present = false;
            }

            if token.token_type == TokenKind::Punct(Punct::Backslash) {
                backslash_present = true;
            }
        }
        end_idx
    }
    
    /// Evaluate expressions used for determining macro conditionals.
    fn eval_const_expr(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<i64> {
        let expr_tokens = self.read_constexpr(iter)?;
        let mut parser = Parser::new();
        let node = parser.parse_as_const_expr(expr_tokens)?;

        node.eval_int()
    }

    /// Read constant expression used for determining macro conditionals. 
    fn read_constexpr(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<Vec<Token>> {
        let mut tokens = Vec::new();
        loop {
            let token = consume_token_or_eof(iter)?;
            if token.at_bol {
                break;
            }

            let token = self.expand(&token).unwrap();
            match token.token_type {
                TokenKind::Ident(ident) => {
                    if ident == "defined" {
                        tokens.push(self.check_if_defined(iter)?);
                    } else {
                        //variables can't be used in macro conditionals
                        //so any identifiers are converted to i0 values.
                        let token = Token::new(
                            TokenKind::Int(0, Bits::Bits32),
                            token.pos.line,
                            token.pos.col,
                            token.pos.filename.map(|s| &*s),
                        );
                        tokens.push(token);
                    }
                },
                _ => tokens.push(token),
            }
        }
        Ok(tokens)
    }
    
    /// Check if a certain identifier has been defined in the macro system.
    fn check_if_defined(&self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<Token> {
        let open_paren = consume_token_or_eof(iter)?;
        if open_paren.token_type == TokenKind::Punct(Punct::OpenParen) {
            let name_token = consume_token_or_eof(iter)?;
            let name = name_token.try_get_name()?;
            let result = if self.macro_map.contains_key(name) {
                Token::new(
                        TokenKind::Int(1, Bits::Bits32), 
                        name_token.pos.line, 
                        name_token.pos.col, 
                        name_token.pos.filename.map(|s| &*s))
            } else {
                Token::new(
                    TokenKind::Int(0, Bits::Bits32), 
                    name_token.pos.line, 
                    name_token.pos.col, 
                    name_token.pos.filename.map(|s| &*s))
            };
            let close_paren = consume_token_or_eof(iter)?;
            if close_paren.token_type == TokenKind::Punct(Punct::CloseParen) {
                Ok(result)
            } else {
                gen_expected_error(&close_paren, TokenKind::Punct(Punct::CloseParen))
            }
        } else {
            gen_expected_error(&open_paren, TokenKind::Punct(Punct::OpenParen))
        }
    }
    
    fn expand(&self, tok: &Token) -> Option<Token> {
        
        let name = tok.get_name();
        match name {
            "__LINE__" => {
                return Some(Token::new(
                    TokenKind::Int(tok.pos.line as i64, Bits::Bits32),
                    tok.pos.line,
                    tok.pos.col,
                    tok.pos.filename.map(|x| &*x)
                ));
            },
            "__FILE__" => {
                return Some(Token::new(
                    TokenKind::Str(tok.pos.filename.unwrap_or("".to_string())),
                    tok.pos.line,
                    tok.pos.col,
                    tok.pos.filename.map(|x| &*x)
                ));
            }
            _ => {},
        }
    
        if tok.is_in_hideset() || !self.macro_map.contains_key(tok.get_name()) {
            Some(tok.clone())
        } else {
            //unwrapping should be safe as we have already checked for the presence of 
            //the key
            match self.macro_map.get(name).unwrap().mtype {
                MacroType::FuncLike(ref names, ref body) => {
                    self.expand(substitute(input, hideset, output))
                }
                MacroType::ObjLike(ref body) => {
                    expand_obj_macro(token, name, macro_body)
                }
            }
        }
        
    
    }

    fn expand_obj_macro(token: Token, name: String, macro_body: &Vec<Token>) {
       let body = macro_body.iter()
        .map(|tok| {
            let mut t = tok.clone();
            t.add_hideset(name);
            t.pos = token.pos.clone();
            t
        })
        .collect();
    }
}


fn merge_hideset(hideset: &HashSet<String>, token: &Token) -> Token {
    
    let mut result = token.clone();
    result.hideset = token.hideset
        .union(hideset)
        .map(|s| s.clone())
        .collect();
    result
}

/// Substitute macro args, handle stringize and paste

// fn substitute(input: &[Token], fp, ap, hideset: &HashSet<String>, output: &[Token]) {
//     if input.is_empty() {
//         merge_hideset(hideset, &output[0]);
//     } 

// }

/// Convert a stream of tokens into a concatenated string.
fn stringize(tp: &TextPosition, ts: &[Token]) -> Token {
    let string = ts
        .iter()
        .map(|token| {
            format!(
                "{}{}",
                if token.has_space {" "} else {""},
                token.token_type
            )
        })
        .fold("".to_string(), |a, s| a + s.as_str())
        .trim_start() //remove leading spaces
        .to_string();
    
    Token::new(TokenKind::Str(string), tp.line, tp.col, tp.filename.map(|x| &*x))
}

///Given a list of comma-separated tokens corresponding to a list of args, select the ith argument
///from the list, as a token sequence.
fn select(idx: usize, token_seq: &[Token]) -> &[Token] {
    let mut count = 0;
    let mut begin = 0;
    let mut end = 0;
    for (i,token) in token_seq.iter().enumerate() {
        if count < idx {
            match token.token_type {
                TokenKind::Punct(punct) => match punct {
                    Punct::Comma => { 
                        count += 1;
                        if count == idx {
                            begin = i;
                        }
                    },
                    _ => {}
                },
                _ => {},
            }
        } else {
            
            match token.token_type {
                TokenKind::Punct(punct) => match punct {
                    Punct::Comma => {end = i; break;},
                    _ => {}
                },
                _ => {},
            }
        }
    }
    &token_seq[begin..end]
}

// fn ts(token: &String) -> PreprocessToken {

// }

// /// formal parameters
// fn fp(token: &String) -> String {

// }
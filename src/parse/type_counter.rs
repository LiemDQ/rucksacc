
use super::parse_err::gen_parser_err;
use crate::err::{ParseErrMsg, ParseRes};
use crate::lex::{Token};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TypeCounter {
    pub VOID: u8,
    pub BOOL: u8,
    pub CHAR: u8,
    pub SHORT: u8,
    pub INT: u8,
    pub LONG: u8,
    pub FLOAT: u8,
    pub DOUBLE: u8,
    pub OTHER: u8,
    pub SIGNED: u8,
    pub UNSIGNED: u8,
}

impl TypeCounter {
    pub fn check_consistency(&self, token: &Token) -> ParseRes<()> {
        
        if self.VOID > 1 {
            return Err(gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.BOOL > 1 {
            return Err(gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.CHAR > 1 {
            return Err(gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.SHORT > 1 {
            return Err(gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.INT > 1 {
            return Err(gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.LONG > 2 {
            //exceptionally, `long long` is acceptable
            return Err(gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.FLOAT > 1 {
            return Err(gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.DOUBLE > 1 {
            return Err(gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.SIGNED > 1 {
            return Err(gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.UNSIGNED > 1 {
            return Err(gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.SIGNED > 1 && self.UNSIGNED > 1 {
            //signed and unsigned are mutually exclusive
            return Err(gen_parser_err(ParseErrMsg::Something, token));
        }

        Ok(())
    }
}

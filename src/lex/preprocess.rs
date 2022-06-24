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

pub enum PreprocessToken {
    HeaderName(String),
    Identifier(String),
    Number(i64), //TODO determine appropriate type for this
    CharConst(char), //character constant
    StrLiteral(String), //string literal
    Punct, //punctuator
}
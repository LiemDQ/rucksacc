use rand::distributions::{Alphanumeric, DistString};

use crate::lex::{Token, TokenKind, Punct, Keyword, AstKind, AmpKind, PMKind};
use crate::err::{ParseErr, ParseRes, ParseErrMsg};
use crate::ast::*;
use crate::types::{StorageClass, TypeInfo, Qualifiers, QualifiedTypeInfo, TypeKind, RecordMember};
use crate::utils::{PeekN, TextPosition};



/// Contains the state of the parser. 
pub struct Parser<'a> {
    syms: Symbols<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeclaratorTypes {
    Function(Vec<Declarator>), 
    Object,
    Array,
    Undetermined,
}

/// A declarator is an intermediate struct for constructing certain AST nodes. 
#[derive(Debug, Clone, PartialEq)]
pub struct Declarator {
    pub decl_type: DeclaratorTypes,
    pub name: String,
    pub qty: QualifiedTypeInfo,
}


#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct TypeCounter {
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
            return Err(Parser::gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.BOOL > 1 {
            return Err(Parser::gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.CHAR > 1 {
            return Err(Parser::gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.SHORT > 1 {
            return Err(Parser::gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.INT > 1 {
            return Err(Parser::gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.LONG > 2 {
            //exceptionally, `long long` is acceptable
            return Err(Parser::gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.FLOAT > 1 {
            return Err(Parser::gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.DOUBLE > 1 {
            return Err(Parser::gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.SIGNED > 1 {
            return Err(Parser::gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.UNSIGNED > 1 {
            return Err(Parser::gen_parser_err(ParseErrMsg::Something, token));
        }
        if self.SIGNED > 1 && self.UNSIGNED > 1 {
            //signed and unsigned are mutually exclusive
            return Err(Parser::gen_parser_err(ParseErrMsg::Something, token));
        }

        Ok(())
    }
}

struct Symbols<'a> {
    constants: SymbolTable<'a>,
    globals: SymbolTable<'a>,
    typedefs: SymbolTable<'a>,
    identifiers: SymbolTable<'a>,
    labels: SymbolTable<'a>,
    externals: SymbolTable<'a>,
}

impl<'a> Symbols<'a> {
    pub fn new() -> Self {
        
        Self { 
            constants: SymbolTable::new(), 
            globals: SymbolTable::new(), 
            typedefs: SymbolTable::new(), 
            identifiers: SymbolTable::new(), 
            labels: SymbolTable::new(), 
            externals: SymbolTable::new(), 
        }
    }
}

/// Determine whether an AST node is a constant expression or not. 
/// 
fn is_const_expr(node: &ASTNode) -> ParseRes<bool> {
    Ok(match node.kind {
        ASTKind::Int(_, _) | ASTKind::Float(_, _) => true,
        ASTKind::BinaryOp(binop) => is_const_expr(&*binop.rhs)? && is_const_expr(&*binop.lhs)?,
        ASTKind::UnaryOp(unop) => is_const_expr(&*unop.id)?,
        ASTKind::TertiaryOp(tertop) => {
            if is_const_expr(&*tertop.lhs)? { //conditional 
                if tertop.lhs.eval_int()? != 0 {
                    return is_const_expr(&*tertop.mid);
                } else {
                    return is_const_expr(&*tertop.rhs);
                }
            } 
            false
        }
        ASTKind::Cast(node, _) => {
            is_const_expr(&* node)?
        }
        //TODO: handle comma nodes 
        _ => false
    })
}

/// C only has two postfix operators: the increment `++` and decrement `--` operators. 
/// However, there are also the member access 
fn postfix_binding_power(tok: &TokenKind) -> Option<(u8, ())> {
    let pow = match tok {
        TokenKind::Punct(pn) => match pn {
            Punct::Inc | Punct::Dec => (4, ()),
            Punct::OpenParen | Punct::OpenBoxBracket => (2,()),
            _ => return None,
        },
        _ => return None,
    };
    Some(pow)
}

fn prefix_binding_power(tok: &TokenKind) -> Option<((), u8)> {
    let pow = match tok {
        TokenKind::Punct(pn) => match pn {
            Punct::Add(_) |
            Punct::Sub(_) => ((),1),
            Punct::Inc | Punct::Dec => ((),1),
            _ => return None,
        }
        _ => return None,
    };

    Some(pow)

    //todo!("Implement prefix binding powers");
}

/// Returns lhs and rhs binding priority of the token, for binary operands.
/// This is needed for the Pratt parser. Lower values indicate higher priority.
/// 
/// TODO: determine desired order for right-to-left associativity.
fn infix_binding_power(tok: &TokenKind) -> Option<(u8, u8)> {
    let res = match tok {
        TokenKind::Punct(pn) => match pn {
            Punct::Arrow | Punct::Point => (1,2), // ->  or .
            Punct::Add(PMKind::Binary) | Punct::Sub(PMKind::Binary) => (8,9),
            Punct::Div | Punct::Asterisk(AstKind::Mult) => (6,7),
            Punct::Shr | Punct::Shl => (10,11),
            Punct::Le | Punct::Lt | 
            Punct::Gt | Punct::Ge => (12,13),
            Punct::Eq | Punct::Ne => (14,15),
            Punct::Ampersand(AmpKind::And) => (16,17),
            Punct::Xor => (18,19),
            Punct::Or => (20,21),
            Punct::LAnd => (22,23),
            Punct::LOr => (24,25),
            Punct::Question => (26,27),
            Punct::Assign | 
            Punct::AssignAdd | Punct::AssignSub | 
            Punct::AssignDiv | Punct::AssignMul | 
            Punct::AssignAnd | Punct::AssignXor |
            Punct::AssignShl | Punct::AssignShl |
            Punct::AssignOr  | Punct::AssignMod => (29,28),
            Punct::Comma => (30, 31),
            _ => return None,
        },
        TokenKind::Keyword(kw) => match kw {
            Keyword::Sizeof => (4,3),
            Keyword::AlignOf => (4,3),
            _ => return None,
        },
        _ => return None,
    };
    Some(res)
}

/// Ensures the next token in the iterator is of kind `tk`. If it is, skip to the next value.
/// If the token is of the wrong kind, throws an `Err` containing information about the expected 
/// kind.
/// 
/// This method is useful for parsing symbols that **must** appear in a particular branch.
#[must_use]
fn ensure_and_consume(iter: &mut impl Iterator<Item = Token>,  tk: TokenKind) -> ParseRes<()> {
    let mut iter = iter.peekable();
    //this should never be EOF
    let token = iter.peek().unwrap();
    if token.token_type != tk {
        return Err(Parser::gen_parser_err(ParseErrMsg::ExpectedSymbol(tk), token));
    } 
    let _ = iter.next();
    Ok(())
}

/// Determine if the next token in the iterator has TokenKind `kind`. This does not
/// consume the iterator element. This is useful for checking for optional symbols during parsing.
/// If consuming the element is desired, use [`consume_if_equal`] instead.
fn check_if_equal(iter: &impl Iterator<Item = Token>, kind: TokenKind) -> bool {
    let iter = iter.peekable();
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
fn consume_if_equal(iter: &mut impl Iterator<Item = Token>, kind: TokenKind) -> bool {
    let iter = iter.peekable();
    iter.peek()
        .and_then(|tok| Some(if tok.token_type == kind { let _ = iter.next(); true} else {false}))
        .unwrap_or(false)
}

/// Extract the [`TextPosition`] from the next token in the iterator, without consuming it.
/// If there is no next token, returns an EOF error. 
fn extract_position(iter: &mut impl Iterator<Item = Token>) -> ParseRes<TextPosition> {
    let iter = iter.peekable();
    let token = match iter.peek() {
        Some(token) => token,
        None => return Parser::gen_eof_error(),
    };

    Ok(token.pos)
}

/// Checks if the next item in the iterator is `None`, returning an `Err` if this is the case.
/// Returns the token otherwise.
#[inline]
fn consume_token_or_eof(iter: &mut impl Iterator<Item = Token>) -> ParseRes<Token> {
    match iter.next() {
        Some(token) => Ok(token),
        None => Parser::gen_eof_error(),
    }
}


impl<'a> Parser<'a> {
    /// This is the starting point for parsing a C program.
    /// 
    /// 6.9
    /// 
    /// `translation-unit ::= external-declaration | translation-unit external-declaration`
    /// 
    /// `external-declaration ::= function-definition | declaration`
    /// 
    /// Function definitions come with additional restrictions: they cannot be `extern`. 
    fn parse_translation_unit(&'a mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<Vec<ASTNode>> {
        
        let nodes = Vec::new();
        let mut iter = &mut iter.peekable();
        while let Some(tok) = iter.peek() {
            let qty = self.parse_declaration_specifier(iter)?; //TODO: determine return type of this function call.

            if self.is_func_def(iter)? {
                nodes.push(self.parse_function_definition(iter, qty)?);
                continue;
            }

            self.parse_declaration(iter);        
        }

        Ok(nodes)
    }

    /// Not an official symbol, but separated out for convenience. 
    fn parse_global_variable(&mut self, iter: &mut impl Iterator<Item = Token>) {
        let mut is_first = true;
        for token in iter.next() {
            if token.token_type == TokenKind::Punct(Punct::Semicolon) {
                //end of statement.
                break;
            }
            if !is_first {
                todo!("Ensure parsing fails unless the next token is a comma");
            }
        }

    }

    /// `type-name ::= specifier-qualifier+ abstract-declarator?`
    fn parse_type_name(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<Declarator> {
        let qty = self.parse_declaration_specifier(iter)?;
        
        self.parse_abstract_declarator(iter, qty) //TODO: this is an optional value. Determine how to remove the need for this, or justify why it should be present.
    }

    /// `abstract-declarator ::= pointer direct-abstract-declarator | direct-abstract-declarator
    fn parse_abstract_declarator(&mut self, iter: &mut impl Iterator<Item = Token>, qty: QualifiedTypeInfo) -> ParseRes<Declarator> {
        let ptr = self.parse_pointer(iter, qty)?;
        if consume_if_equal(iter, TokenKind::Punct(Punct::OpenParen)) {
            let decl = self.parse_abstract_declarator(iter, qty)?;
            ensure_and_consume(iter, TokenKind::Punct(Punct::CloseParen))?;
            Ok(self.parse_declarator_suffix(iter, decl)?)

        } else {
            //TODO: it's not clear what the declarator type should be.
            //Object seems to be a safe bet for now.
            Ok(Declarator{
                decl_type: DeclaratorTypes::Object, 
                name: "".to_string(),
                qty: qty,
            })
        }
        
    }

    /// 6.5.1
    /// 
    /// `primary-expression ::= identifier | constant | string-literal | '(' expression ')' | generic-selection`
    /// 
    /// If primary-expression is an identifier, it is either an lvalue (if it has been declared as an object) or
    /// a function designator (if it has been declared as a function).
    /// 
    /// 
    fn parse_primary_expression(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<ASTNode> {
        let token = consume_token_or_eof(iter)?;

        let node = match token.token_type {
            TokenKind::Ident(s) => {
                //TODO: possibly different behavior if it is an lvalue vs function designator?
                if let Some(symbol) = self.syms.identifiers.get_symbol(&s) {
                    symbol.node
                } else {
                    return Err(Parser::gen_parser_err(ParseErrMsg::UnknownIdentifier, &token));
                }
            },
            TokenKind::Int(v) => ASTNode::new(ASTKind::Int(v, 8), token.pos), //TODO: how to determine proper size?
            TokenKind::Float(v) => ASTNode::new(ASTKind::Float(v, 8), token.pos),
            TokenKind::Str(s) => ASTNode::new(ASTKind::String(s), token.pos),
            TokenKind::Char(c) => ASTNode::new(ASTKind::Char(c as i16), token.pos),
            TokenKind::Punct(Punct::OpenParen) => self.parse_expression(iter)?,
            _ => return Parser::gen_internal_error(&token, line!()),
        };

        Ok(node)
    }

    /// 6.5.2
    /// 
    /// `postfix-expression ::= '(' type-name ')' '{' initializer-list '}' 
    ///                         | primary-expression  
    fn parse_postfix_expression(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<ASTNode> {
        let mut iter = iter.peekable();
        
        if consume_if_equal(&mut iter, TokenKind::Punct(Punct::OpenParen)) {
            let name = self.parse_type_name(&mut iter)?;
            //parse type-name
            ensure_and_consume(&mut iter, TokenKind::Punct(Punct::CloseParen))?;
            ensure_and_consume(&mut iter, TokenKind::Punct(Punct::OpenBrace))?;
            //TODO: this needs to be an initializer-list, not just an initializer!
            let node = self.parse_initializer_list(&mut iter, &mut name.qty.ty)?;
            ensure_and_consume(&mut iter, TokenKind::Punct(Punct::OpenBrace))?;
            Ok(node)
        } else {
            self.parse_primary_expression(&mut iter)
        }
    }

    /// 6.7.9.
    /// 
    /// `initializer-list ::=  (designation? initializer ',')* designation? initializer`
    /// 
    /// `designation = designator* '='`
    /// 
    /// `designator = '['constant-expression']' | '.' identifier`
    /// 
    /// If the object has static or thread storage duration (i.e. most things)
    /// then the expressions in the initializer shall be constant expressions or string literals. 
    fn parse_initializer_list(&mut self, iter: &mut impl Iterator<Item = Token>, ty: &mut TypeInfo) -> ParseRes<ASTNode> {
        // no need to check for braces
        let token = consume_token_or_eof(iter)?;

        if let TokenKind::Str(s) = token.token_type {
            return self.parse_string_initializer(iter, &mut ty, s);
        }
        match ty.kind {
            TypeKind::Array(_, _) => self.parse_array_initializer(iter, &mut ty),
            TypeKind::Struct(_, _) | TypeKind::Union(_, _, _) => self.parse_struct_initializer(iter, &ty),
            _ => self.parse_assignment_expression(iter)
        }
    }

    
    /// `initializer ::= assignment-expression | '{' initializer-list ','? '}'`
    /// 
    /// Initializers consist of either assignment-expressions or nestings of initializers. 
    /// Nested initializers are for array or struct types. 
    fn parse_initializer(&mut self, iter: &mut impl Iterator<Item = Token>, ty: &mut TypeInfo) -> ParseRes<ASTNode> {
        if match ty.kind {
            TypeKind::Array(_, _) | TypeKind::Struct(_, _) | TypeKind::Union(_, _, _) => true,
            _ => false,
            } {
            self.parse_initializer_list(iter, &mut ty)
        }
        else if consume_if_equal(iter, TokenKind::Punct(Punct::OpenBrace)) {
            let pos = extract_position(iter)?;
            let node = self.parse_initializer(iter, ty);
            ensure_and_consume(iter, TokenKind::Punct(Punct::CloseBrace))?;
            node

        } else {
            self.parse_assignment_expression(iter)
        }
    }

    /// String literals in an initializer evaluate to a char array. 
    /// 
    /// E.g. `char my_string[] = "Hello world!";` will initialize a char array of length 12. 
    /// 
    /// This is a special case outlined in 6.7.9.C14-15, and requires a separate initialization method
    /// as there are otherwise no explicit semantics mapping char arrays to a string literal.
    fn parse_string_initializer(&mut self, iter: &mut impl Iterator<Item = Token>, ty: &mut TypeInfo, string: String) -> ParseRes<ASTNode> {
        let pos = extract_position(iter)?;
        let chars = string.chars()
            .map(|c| ASTNode::new(ASTKind::Char(c as i16), pos))
            .collect::<Vec<ASTNode>>();
        
            if let TypeKind::Array(_, ref mut len) = &mut ty.kind {
                *len = chars.len() as i64 + 1 ;
            } else {
                return Err(Parser::gen_parser_err_pos(ParseErrMsg::TypeError("array".to_string()), &pos));
            }

        let kind = ASTKind::InitArray(chars);
        Ok(ASTNode::new(kind, pos))
    }
    
    /// Array initializers 
    fn parse_array_initializer(&mut self, iter: &mut impl Iterator<Item = Token>, ty: &mut TypeInfo) -> ParseRes<ASTNode> {
        let pos = extract_position(iter)?;

        if let TypeKind::Array(ref element_type, ref mut len) = &mut ty.kind {
            let is_flexible = *len < 0;
            let mut elems = Vec::new();
            let element_type = **element_type;
            loop {
                let token = consume_token_or_eof(iter)?;
                if consume_if_equal(iter, TokenKind::Punct(Punct::CloseBrace)) {
                    //TODO: handle case where there is no initial opening brace
                    break;
                }

                let elem = self.parse_initializer(iter, &mut element_type)?;
                elems.push(elem);
                ensure_and_consume(iter, TokenKind::Punct(Punct::Comma))?;
            }
            if is_flexible {
                *len = elems.len() as i64;
            }

            Ok(ASTNode::new(
                ASTKind::InitArray(elems),
                pos,
            ))

        } else {
            return Err(Parser::gen_parser_err_pos(ParseErrMsg::TypeError("array".to_string()), &pos));
        }
    }

    /// Struct initializers are of the form `{member-initializer-1, member-initializer-2, ...}`.
    /// Initializers can be nested, so for example `{{}, {}}` is a valid initializer. This necessitates
    /// a tree structure for representing the initializer. 
    /// 
    /// Type information about the struct members is needed to determine whether the initializer for each member is valid.
    /// Members are initialized in the order in which they are placed in the struct. 
    fn parse_struct_initializer(&mut self, iter: &mut impl Iterator<Item = Token>, ty: &TypeInfo) -> ParseRes<ASTNode> {
        let token = consume_token_or_eof(iter)?;
        let member_types = if let Some(member_types) = ty.get_struct_member_types() { 
            member_types
        } else {
            return Parser::gen_internal_error(&token, line!());
        };

        let mut iter = iter.peekable();
        let mut member_iter = member_types.iter();
        let members = Vec::new();

        while let Some(tok) = iter.peek() {
            if consume_if_equal(&mut iter, TokenKind::Punct(Punct::CloseBrace)){
                break;
            }
            let mem = self.parse_initializer(&mut iter, &mut member_iter.next().unwrap().clone())?;
            members.push(mem);
            ensure_and_consume(&mut iter, TokenKind::Punct(Punct::Comma))?;
        } 
        Ok(ASTNode::new(
            ASTKind::InitStruct(members),
            token.pos,
        ))
    }

    /// Create ASTNodes from a variable declarator. 
    fn make_variable_node_from_declarator(decl: Declarator, tk: &Token) -> ParseRes<ASTNode> {
        if decl.decl_type != DeclaratorTypes::Object {
            return Parser::gen_internal_error(tk, line!());
        }
        //TODO: determine what the offset value should be
        let kind = ASTKind::Variable(Var{name: decl.name, ty: decl.qty.ty, offset: 0});
        Ok(ASTNode { kind: kind, pos: tk.pos})
    }

    
    /// `function-definition ::= declaration-specifier* declarator declaration* compound-statement`
    /// 
    /// Note that we parse the declaration specifier outside of this function. 
    /// A function definition is a scope block (compound statement) with params added to the scope. 
    /// 
    fn parse_function_definition(&'a mut self, iter: &mut impl Iterator<Item = Token>, qty: QualifiedTypeInfo) -> ParseRes<ASTNode> {
        let mut iter = iter.peekable();
        let token = iter.peek().unwrap(); // TODO: error handling for this

        let dec = self.parse_declarator(&mut iter, qty)?; 
        let params = match dec.qty.ty.kind {
            TypeKind::Func(_, params, is_vararg) => params, //TODO: handle varargs
            _ => return Err(Parser::gen_parser_err(ParseErrMsg::Something, iter.peekable().peek().unwrap())),
        };

        //TODO: parse declarations? these are an optional element. 

        let scope = self.syms.identifiers.enter_scope();
        
        //as per the C spec, each function must have a `__func__` symbol that contains the name of the function.
        let func_symbol = ASTNode::new(ASTKind::String(dec.name), TextPosition { line: 0, col: 0, filename: token.pos.filename });
        scope.emplace_symbol(
            "__func__",
            func_symbol,
            dec.qty.storage,
        )?;
        
        let result = match dec.decl_type {
            DeclaratorTypes::Function(funcs) => { 
                //if function has parameters, add them to the scope 
                //before passing it to the block
                for dec in funcs {
                    let node = Parser::make_variable_node_from_declarator(dec, &token)?;
                    scope.emplace_symbol(
                        &dec.name,
                        node,
                        dec.qty.storage,
                    );
                }
                Box::new(self.parse_compound_stmt(&mut iter, Some(scope))?)
            },
            //this should never happen
            _ => return Parser::gen_internal_error(&token, line!()),
        };
        
        let func = Function {name: dec.name, declarator: dec, body: result, stack_size: 0};
        
        let node = ASTNode::new(ASTKind::Func(func),token.pos);
        
        let func = Symbol::new_local(&dec.name, node.clone());
        
        //add function to global symbols
        self.syms.identifiers.push_global_symbol(func)?;
        Ok(node)
    }

    fn token_to_unary_op(kind: &TokenKind, is_prefix: bool) -> Option<UnaryOps> {
        Some(match kind {
            TokenKind::Punct(p) => match p {
                Punct::Inc => if is_prefix {UnaryOps::IncPrefix} else { UnaryOps::IncPostfix},
                Punct::Dec => if is_prefix {UnaryOps::DecPrefix} else {UnaryOps::DecPostfix},
                Punct::Asterisk(_) => UnaryOps::Deref,
                Punct::Ampersand(_) => UnaryOps::Addr,
                Punct::BitwiseNot => UnaryOps::BNot,
                Punct::Not => UnaryOps::LNot,
                Punct::Sub(_) => UnaryOps::Neg,
                _ => return None,
            },
            TokenKind::Keyword(Keyword::AlignOf) => UnaryOps::Alignof,
            TokenKind::Keyword(Keyword::Sizeof) => UnaryOps::Sizeof,
            _ => return None,
        })
    }

    /// Expressions are strings that evaluate to a value. This is in contrast to statements, which do not
    /// evaluate a value and instead produce some side effect. 
    fn parse_expression(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<ASTNode> {
        self.parse_expr_bp(iter, 0)
    }


    
    fn parse_assignment_expression(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<ASTNode> {
       self.parse_expr_bp(iter, 0)
    }

    /// Pratt parser to evaluate expressions with operator precedence. This avoids having to create a complex
    /// nest of mutually recursive expression functions in order to properly capture operator precedence.
    /// The Pratt parser can evaluate operations in the proper order, but individual parsing functions for the operations 
    /// are still needed to create AST nodes. 
    /// 
    /// This is a little trickier because there are multiple entry points into expressions in C. Most expressions
    /// will have the full subset applied, but assignment-expressions can only consider a subset of the symbols seen here.
    /// 
    /// See [Cppreference](https://en.cppreference.com/w/c/language/operator_precedence) for details on operator precedence.
    fn parse_expr_bp(&mut self, iter: &mut impl Iterator<Item = Token>, min_bind_prio: u8) -> ParseRes<ASTNode> {
        //30 is the lowest binding prio possible
        const MIN_BIND_PRIO: u8 = 30;

        let mut iter = &mut iter.peekable();
        
        //Handle atomic tokens. These are individual "units" that need to be transformed to 
        //fundamental AST nodes. 
        let mut lhs = match iter.next() {
            Some(token) => match token.token_type {
                TokenKind::Punct(Punct::OpenParen) => {
                    self.parse_expr_bp(iter, MIN_BIND_PRIO)?
                }
                //literals are tautologically represent themselves
                TokenKind::Int(n) => ASTNode { kind: ASTKind::Int(n, 4), pos: token.pos},
                TokenKind::Float(f) => ASTNode {kind: ASTKind::Float(f, 4), pos: token.pos},
                TokenKind::Char(c) => ASTNode {kind: ASTKind::Char(c as i16), pos: token.pos},
                TokenKind::Str(s) => ASTNode::new(ASTKind::String(s), token.pos),
                //if it is an identifier, retrieve it from the symbol table
                TokenKind::Ident(id) => {
                    if let Some(symbol) = self.syms.identifiers.get_symbol(&id) {
                        symbol.node
                    } else {
                        return Err(Parser::gen_parser_err(ParseErrMsg::UnknownIdentifier, &token));
                    }
                }
                //special handling for Sizeof keyword
                TokenKind::Keyword(Keyword::Sizeof) => {
                    ensure_and_consume(iter, TokenKind::Punct(Punct::OpenParen))?;
                    let declarator = self.parse_type_name(iter)?;
                    ensure_and_consume(iter, TokenKind::Punct(Punct::CloseParen))?;
                    // if declaration specifier is set up correctly, the size should already be contained in the declarator
                    // integer should be of type size_t which is 8 bytes on a 64-bit system?
                    let kind = ASTKind::Int(declarator.qty.ty.size, 8);
                    ASTNode::new(kind, token.pos)
                    
                },
                TokenKind::Keyword(_) | TokenKind::Punct(_) => {

                    let bp = prefix_binding_power(&token.token_type);
                    if let Some(((),r_bp)) = bp {
                        let op = Parser::token_to_unary_op(&token.token_type, true);
    
                        if let Some(op) = op {
                            //stack overflow should never be a practical issue, as expressions rarely go beyond ~30 operations.
                            let expr = Box::new(self.parse_expr_bp(iter, r_bp)?);
                            let unary_expr = UnaryExpr {id: expr, op: op};
                            ASTNode::new(ASTKind::UnaryOp(unary_expr), token.pos)
                        } else {
                            //should never happen
                            return Parser::gen_internal_error(&token, line!());
                        }
                    } else {
                        //if the token has no prefix bounding power, it is not a valid token to prefix an expression
                        return Err(Parser::gen_parser_err(ParseErrMsg::InvalidExpressionToken, &token));
                    }
                }
                _ => return Err(Parser::gen_parser_err(ParseErrMsg::Something, &token)),

            },
            //None should never happen and is indicative of an internal compiler error. 
            None => panic!("Internal compiler error: tokens terminate abruptly."), 
        };

        //Loop, building up the AST until the minimum binding power is found 
        //or end of expression is reached. 
        loop {
            let tok = match iter.peek() {
                Some(t) => t,
                None => return Parser::gen_eof_error(),
            };

            let op = match tok.token_type {
                TokenKind::EOF => break,
                TokenKind::Punct(_) | TokenKind::Keyword(_) => tok.token_type,
                //error: invalid token
                _ => return Parser::gen_expected_one_of_error(&tok, vec![TokenKind::Punct(Punct::Any), TokenKind::Keyword(Keyword::Any), TokenKind::EOF]),
            };
            
            //handle postfix operators. 
            //this also applies to function calls and array accesses. 
            if let Some((l_bp, ())) = postfix_binding_power(&op) {
                if l_bp > min_bind_prio {
                    break;
                }
                iter.next();
                //handle arrays
                //this may need to be a match statement rather than an if/else chain.
                
                //in practice, all the postfix operators are at the highest level of priority, so
                //they can evaluated before anything else. 
                lhs = match op {
                    TokenKind::Punct(Punct::OpenBoxBracket) => {
                        let rhs = self.parse_expr_bp(iter, MIN_BIND_PRIO)?;
                        ensure_and_consume(&mut iter, TokenKind::Punct(Punct::CloseBoxBracket))?;
                        let node = ASTNode {
                            kind: ASTKind::BinaryOp(BinaryExpr{lhs: Box::new(lhs), op: BinaryOps::Add, rhs: Box::new(rhs)}),
                            pos: tok.pos,
                        };
                        ASTNode{kind: ASTKind::Load(Box::new(node)), pos: tok.pos}
                    },
                    TokenKind::Punct(Punct::OpenParen) => {
                        //a postfix parenthesis indicates a function call.
                        //in this case, the most immediate lhs node is a function node.
                        if let ASTKind::Func(f) = lhs.kind {
                            self.parse_func_call(iter, &f)?
                        } else {
                            return Err(Parser::gen_parser_err(ParseErrMsg::InvalidFunctionCall, &tok));
                        }
                    },
                    TokenKind::Punct(Punct::Point) => {
                        if let ASTKind::Variable(var) = lhs.kind {
                            //struct reference
                            if let Some(next) = iter.next() {
                                self.parse_struct_member(iter, &var, &next)?
                            } else {
                                return Parser::gen_eof_error();
                            }
                        } else {
                            return Err(Parser::gen_parser_err(ParseErrMsg::InvalidStructReference, &tok))
                        }
                    },
                    TokenKind::Punct(Punct::Arrow) => {
                        if let ASTKind::Variable(var) = lhs.kind {
                            //arrow operator is deference + struct member access
                            if let Some(next) = iter.next() {
                                todo!("Deference var");
                                self.parse_struct_member(iter, &var, &next)?
                            } else {
                                return Parser::gen_eof_error();
                            }
                        } else {
                            return Err(Parser::gen_parser_err(ParseErrMsg::InvalidStructReference, &tok))
                        }
                    },
                    _ => {
                        let unary_op_opt = Parser::token_to_unary_op(&op, false);
                        if let Some(unary_op) = unary_op_opt {
                            let rhs = Box::new(self.parse_expr_bp(iter, min_bind_prio)?);
                            let unary_expr = UnaryExpr{op: unary_op, id: rhs };
                            ASTNode::new(ASTKind::UnaryOp(unary_expr), tok.pos)
                        } else {
                            //this should never happen if postfix binding power has the correct operators
                            return Parser::gen_internal_error(tok, line!());
                        }
                    }
                };
                continue;
            }

            if let Some((l_bp, r_bp)) = infix_binding_power(&op) {
                if l_bp > min_bind_prio {
                    break;
                }

                iter.next();

                //handle special case of conditional expressions, which are a tertiary expression.
                let lhs = if op == TokenKind::Punct(Punct::Question) {
                    let mhs = self.parse_expr_bp(&mut iter, MIN_BIND_PRIO)?;

                    ensure_and_consume(&mut iter, TokenKind::Punct(Punct::Colon))?;

                    let rhs = self.parse_expr_bp(&mut iter, r_bp)?;
                    let kind = ASTKind::TertiaryOp(
                        TertiaryExpr{
                            lhs: Box::new(lhs), 
                            op: TertiaryOps::Conditional, 
                            mid: Box::new(mhs), 
                            rhs: Box::new(rhs)});
                    ASTNode {kind: kind, pos: tok.pos}
                } else { 
                    //otherwise, we have a binary expression.
                    let rhs = self.parse_expr_bp(iter, r_bp)?;
                    if let Some(node) = Parser::to_assign(lhs.clone(), &op, rhs.clone()) {
                         node
                    } else {
                        let binop = match op {
                            TokenKind::Punct(Punct::Add(_)) => BinaryOps::Add,
                            TokenKind::Punct(Punct::Sub(_)) => BinaryOps::Sub,
                            TokenKind::Punct(Punct::Div) => BinaryOps::Div,
                            TokenKind::Punct(Punct::Asterisk(_)) => BinaryOps::Mult,
                            TokenKind::Punct(Punct::Lt) => BinaryOps::Lt,
                            TokenKind::Punct(Punct::Le) => BinaryOps::Le,
                            TokenKind::Punct(Punct::Gt) => BinaryOps::Gt,
                            TokenKind::Punct(Punct::Ge) => BinaryOps::Ge,
                            TokenKind::Punct(Punct::Ne) => BinaryOps::Ne,
                            TokenKind::Punct(Punct::Eq) => BinaryOps::Eq,
                            TokenKind::Punct(Punct::Ampersand(_)) => BinaryOps::BAnd,
                            TokenKind::Punct(Punct::Or) => BinaryOps::BOr,
                            TokenKind::Punct(Punct::Xor) => BinaryOps::BXor,
                            TokenKind::Punct(Punct::LOr) => BinaryOps::LOr,
                            TokenKind::Punct(Punct::LAnd) => BinaryOps::LAnd,
                            TokenKind::Punct(Punct::Shr) => BinaryOps::Shr,
                            TokenKind::Punct(Punct::Shl) => BinaryOps::Shl,
                            TokenKind::Punct(Punct::Assign) => BinaryOps::Assign,
                            //TODO: assignment operators
                            _ => return Err(Parser::gen_parser_err(ParseErrMsg::Something, &tok)),
                        };
    
                        ASTNode {kind: ASTKind::BinaryOp(BinaryExpr{op: binop, rhs: Box::new(rhs), lhs: Box::new(lhs)}), pos: tok.pos}
                    }
                };
                continue;
            }

            break;
        }
        Ok(lhs)
    }
    
    ///
    /// `declaration ::= declaration-specifier+ init-declarator* ';'
    /// 
    /// `init-declarator ::= declarator ('=' initializer)?`
    fn parse_declaration(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<ASTNode> {
        let qty = self.parse_declaration_specifier(iter)?;
        let mut iter = iter.peekable();
        let token = match iter.peek() {
            Some(tok) => tok,
            None => return Parser::gen_eof_error(),
        };

        if consume_if_equal(&mut iter, TokenKind::Punct(Punct::Semicolon)) {
            //end of declaration.
            //in this case it does not declare anything, but it is syntactically valid.
            let node_kind = ASTKind::Noop;
            Ok(ASTNode {kind: node_kind, pos: token.pos})

        } else { //init-declarator, possibly multiple
            let declarator = self.parse_declarator(&mut iter, qty)?;
            let lhs = ASTNode::make_variable_declaration(declarator, &token);

            if consume_if_equal(&mut iter, TokenKind::Punct(Punct::Assign)) {
                let mut rhs = self.parse_initializer(&mut iter, &mut declarator.qty.ty)?;
                let mut new_rhs = rhs.clone();
                let mut node = ASTNode::make_assignment(lhs, rhs);
                
                while !consume_if_equal(&mut iter, TokenKind::Punct(Punct::Semicolon)) {
                    let declarator = self.parse_declarator(&mut iter, qty)?;
                    new_rhs = self.parse_initializer(&mut iter, &mut declarator.qty.ty)?; 
                    //TODO: this may necessitate multiple variable declarations being created instead of a make_assignement
                    rhs = ASTNode::make_assignment(rhs, new_rhs);
                    node.reassign_rhs(rhs);
                }
                Ok(node)
            } else {
                ensure_and_consume(&mut iter, TokenKind::Punct(Punct::Semicolon))?;
                Ok(lhs)
            }
        }
    }

    /// Converts assignment operators to assignment expressions. 
    /// 
    /// The non-assignment operator needs to be executed first, so it is 
    /// lower on the parse tree. 
    fn to_assign(lhs: ASTNode, op: &TokenKind, rhs: ASTNode) -> Option<ASTNode> {
        //convert `a += b` to `a = a + b`
        let binop = match op {
            TokenKind::Punct(Punct::AssignAdd) => BinaryOps::Add,
            TokenKind::Punct(Punct::AssignSub) => BinaryOps::Sub,
            TokenKind::Punct(Punct::AssignMul) => BinaryOps::Mult,
            TokenKind::Punct(Punct::AssignDiv) => BinaryOps::Div,
            TokenKind::Punct(Punct::AssignMod) => BinaryOps::Mod,
            TokenKind::Punct(Punct::AssignShl) => BinaryOps::Shl,
            TokenKind::Punct(Punct::AssignShr) => BinaryOps::Shr,
            TokenKind::Punct(Punct::AssignOr)  => BinaryOps::BOr,
            TokenKind::Punct(Punct::AssignAnd) => BinaryOps::BAnd,
            TokenKind::Punct(Punct::AssignXor) => BinaryOps::BXor,
            _ => return None,
        };

        let op_node = Box::new(ASTNode {
            kind: ASTKind::BinaryOp(BinaryExpr{lhs: Box::new(lhs), op: binop, rhs: Box::new(rhs) }), 
            pos: lhs.pos,
        });

        let assign_node = ASTNode {
            kind: ASTKind::BinaryOp(BinaryExpr {lhs: Box::new(lhs), op: BinaryOps::Assign, rhs: op_node}),
            pos: lhs.pos,
        };

        Some(assign_node)
    }

    fn parse_struct_member(&mut self, iter: &mut impl Iterator<Item = Token>, var: &Var, member: &Token) -> ParseRes<ASTNode> {
        todo!("Parse struct member");
    }

    /// A declarator declares an identifier and specifies its form and whether it is a function or object. 
    /// When an operand of the same form as the declarator appears in an expression, it designates the function
    /// or object with the scope, storage duration and type indicated by the declaration specifier, parsed before 
    /// the declarator.
    /// 
    /// 
    /// See 6.7.6.
    ///
    /// `declarator ::= pointer? (identifier| '(' declarator ')' | ) declarator-suffix`
    /// 
    /// `declarator-suffix ::= ('[' constant-expression ']' | '(' parameter-list')')*`
    fn parse_declarator(&mut self, iter: &mut impl Iterator<Item = Token>, mut qty: QualifiedTypeInfo) -> ParseRes<Declarator> {
        let mut qty = self.parse_pointer(iter, qty)?;
        
        if consume_if_equal(iter, TokenKind::Punct(Punct::OpenParen)) {
            //nested declarators.
            let mut declarator = self.parse_declarator(iter, qty)?;
            ensure_and_consume(iter, TokenKind::Punct(Punct::CloseParen))?;
            declarator = self.parse_declarator_suffix(iter, declarator)?;
            return Ok(declarator);
        }

        let token = iter.peekable().peek().unwrap(); //TODO: proper None handling here
        match token.token_type {
            TokenKind::Ident(id) => {
                let mut declarator = Declarator{ decl_type: DeclaratorTypes::Undetermined, name: id, qty: qty};
                self.parse_declarator_suffix(iter, declarator)
            }
            _ => {
                //no identifier??
                todo!("Determine what to do when declarator has no identifier");
                
                //self.parse_declarator_suffix(iter, qty)?;
            }
        }
    }

    /// Suffix for declarators. Depending on the suffix, the declared object is either:
    /// - A function
    /// - An array
    /// - A variable
    /// 
    /// If there are square braces `[]` then then an array is declared. 
    /// If there are parentheses `()` then a function is declared. 
    /// 
    /// `declarator-suffix ::= '(' func-params ')' | '[' const-expression ']' | ε`
    /// 
    /// See [`self.parse_declarator(self, iter, qty)`]
    fn parse_declarator_suffix(&mut self, iter: &mut impl Iterator<Item = Token>, decl: Declarator) -> ParseRes<Declarator> {
        let mut iter = iter.peekable();
        match iter.peek().unwrap().token_type {
            TokenKind::Punct(Punct::OpenParen) => self.parse_declarator_func_params(&mut iter, decl),
            TokenKind::Punct(Punct::OpenBoxBracket) => self.parse_declarator_array_dimensions(&mut iter, decl),
            _ => Ok(decl),
        }
    }
    
    /// Parsing function params is needed in the grammar for `declarator`. However, the parsing logic is separated
    /// out here for ease of reading.
    /// 
    /// `func-params ::= "void" | param ("," param)*`
    /// 
    /// `param ::= declaration-specifier declarator`
    fn parse_declarator_func_params(&mut self, iter: &mut impl Iterator<Item = Token>, mut decl: Declarator) -> ParseRes<Declarator> {
        let mut iter = iter.peekable_n();

        //if the function signature is `(void)`, then the function accepts no parameters.
        //TODO: determine whether an empty signature, e.g. `func()` also needs to be explicitly handled.
        //TODO: this seems like this path can be taken if the keyword is void without checking for the closeParen.
        //then we need to ensure that CloseParen is the next token regardless.
        if iter.peek().unwrap().token_type == TokenKind::Keyword(Keyword::Void) 
        && iter.peek_next().unwrap().token_type == TokenKind::Punct(Punct::CloseParen) {
            decl.qty.ty = decl.qty.ty.make_func(Vec::new(), false);
            iter.skip_to_cursor();
            iter.next();
            return Ok(decl);
        } else {
            //
            iter.reset_cursor();
        }
        
        let mut is_vararg = false; 
        let mut is_first_loop = true;
        let params = Vec::new();

        while let Some(token) = iter.next() {
            if token.token_type == TokenKind::Punct(Punct::CloseParen){
                break;
            }
            
            //all params must be followed by a ',' after parsing the first param.
            if !is_first_loop {
                ensure_and_consume(&mut iter, TokenKind::Punct(Punct::Comma))?;
            }
            
            //if the next token is `...` then the function accepts varargs. 
            //This should be the last argument of the function.
            if token.token_type == TokenKind::Punct(Punct::Vararg){
                is_vararg = true;
                iter.next();
                ensure_and_consume(&mut iter, TokenKind::Punct(Punct::CloseParen))?;
                break;
            }

            let qty = self.parse_declaration_specifier(&mut iter)?;
            let mut item_decl = self.parse_declarator(&mut iter, qty)?;
            match item_decl.qty.ty.kind {
                //if the type is an array or a function, convert to a pointer 
                TypeKind::Array(_, _) | TypeKind::Func(_, _, _) => item_decl.qty.ty = item_decl.qty.ty.make_ptr(),
                _ => (),
            }
            params.push(item_decl);
            is_first_loop = false; 
        }

        decl.decl_type = DeclaratorTypes::Function(params);
        //TODO: because of our schema, the same information (function parameter type infommation) is redundantly represented 
        //at different levels of the data structure. This should be revisited at a later point to see if this can be simplified. 
        decl.qty.ty = decl.qty.ty.make_func(params.into_iter().map(|d| d.qty.ty).collect(), is_vararg);

        Ok(decl)
    }
    
    /// Parse array dimensions.
    /// Specifiers for array dimensions can be `static` or `restrict`. 
    /// 
    fn parse_declarator_array_dimensions(&mut self, iter: &mut impl Iterator<Item = Token>, mut decl: Declarator) -> ParseRes<Declarator> {
        
        //skip all `static` and `restrict` keywords. 
        let mut iter = iter.skip_while(
            |t| t.token_type == TokenKind::Keyword(Keyword::Static) 
                || t.token_type == TokenKind::Keyword(Keyword::Restrict));
        
        if iter.next().unwrap().token_type == TokenKind::Punct(Punct::CloseBoxBracket) {
            decl = self.parse_declarator_suffix(&mut iter, decl)?;
            decl.qty.ty.make_array(-1);
            return Ok(decl);
        }
        
        let expr = self.parse_const_expression(&mut iter)?;
        ensure_and_consume(&mut iter, TokenKind::Punct(Punct::CloseBoxBracket));
        decl = self.parse_declarator_suffix(&mut iter, decl)?;
        if is_const_expr(&expr)? || decl.qty.ty.kind == TypeKind::VLA {
            decl.qty.ty = decl.qty.ty.make_vla_array(expr);
            return Ok(decl);
        }
        //evaluate expression to determine size of array.
        decl.qty.ty = decl.qty.ty.make_array(expr.eval_int()?);
        Ok(decl)
    }

    fn parse_const_expression(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<ASTNode> {
        let mut iter = iter.peekable();
        let token = iter.peek().unwrap();
        let node = self.parse_expr_bp(&mut iter, 30)?;
        if is_const_expr(&node)? {
            Ok(node)
        } else {
            Err(Parser::gen_parser_err(ParseErrMsg::NotCompileTimeConstant, &token))
        }
    }

    /// `struct-declaration ::= specifier-qualifier* struct declarator ( ',' struct-declarator )*`
    /// 
    /// `struct-declarator ::= declarator (':' constant-expression )? | ';' constant-expression`
    /// 
    /// The presence of colons indicates a bitfield which has unique alignment and size considerations. 
    /// 
    /// In theory, unions and structs are parsed the exact same way. In practice, they are represented differently
    /// at a semantic level, which means it is typically easier to parse them with separate functions.
    /// 
    /// `struct-union-specifier ::= struct-or-union ( identifier ('{' struct-declaration '}')? ) | '{' struct-declaration '}'`
    /// struct 
    fn parse_struct_union_specifier(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<TypeInfo> {
        let is_struct = match iter.next() {
            None => return Parser::gen_eof_error(),
            Some(token) => match token.token_type {
                TokenKind::Keyword(Keyword::Struct) => true,
                TokenKind::Keyword(Keyword::Union) => false,
                _ => return Parser::gen_expected_error(&token, TokenKind::Keyword(Keyword::Struct)),
            }
        };

        let mut iter = iter.peekable();
        let (name, members) = match iter.peek() {
            Some(token) => match token.token_type {
                TokenKind::Ident(s) => {
                    iter.next();
                    ensure_and_consume(&mut iter, TokenKind::Punct(Punct::OpenBrace));
                    (s, self.parse_record_members(&mut iter)?)
                },
                TokenKind::Punct(Punct::OpenBrace) => {
                    //if the record type has no name, generate a random name
                    //e.g. `typedef struct {int a; } S;`
                    let name: String = Alphanumeric.sample_string(&mut rand::thread_rng(), 16);
                    (name, self.parse_record_members(&mut iter)?)
                },
                _ => return Parser::gen_expected_error(token, TokenKind::Punct(Punct::OpenBrace)),
            },
            None => return Parser::gen_eof_error(),
        };

        //if there is no struct definition, then we merely have a struct declaration that should be added to 
        //the symbol table.


        //TODO: I am unsure how to add the struct as a symbol to the symbol table.
        //this needs to be done in order to reference it later when structs need to
        //be initialized. 

        Ok(if members.is_empty() {
            //if the struct/union has no members, then it is an empty type.
            //
            if is_struct {
                TypeInfo {
                    kind: TypeKind::Struct(name, Vec::new()),
                    align: 0,
                    size: 1,
                }
            } else {
                TypeInfo {
                    kind: TypeKind::Union(name, members, 0),
                    align: 0,
                    size: 1,
                }
            }
        } else {
            let mut align = 1;
            let mut size = 0;
            //TODO: handle bitfields
            if is_struct {
                for mem in members {
                    size += mem.ty.size;
                    align = std::cmp::max(align, mem.ty.align);
                }
                TypeInfo {
                    kind: TypeKind::Struct(name, members),
                    align: align,
                    size: size,
                }
            } else {
                let mut index = 0;
                for (n, mem) in members.into_iter().enumerate() {
                    let old = size;
                    size = std::cmp::max(mem.ty.size, size);
                    if old < size {
                        index = n;
                    }
                    align = std::cmp::max(mem.ty.align, align);
                }

                TypeInfo {
                    kind: TypeKind::Union(name, members, index),
                    align: align,
                    size: size,
                }
            }
        })
    }


    /// `struct-declaration ::= specifier-qualifier struct-declarator ( ',' struct-declarator )*`
    /// 
    /// `struct-declarator ::= declarator ':' constant-expression | ':' constant-expression
    fn parse_record_members(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<Vec<RecordMember>> {
        let members = Vec::new();

        while !consume_if_equal(iter, TokenKind::Punct(Punct::CloseBrace)) {
            let qty = self.parse_declaration_specifier(iter)?;
            let mut first = true;
            
            //anonymous struct member
            match qty.ty.kind {
                TypeKind::Struct(_, _) => todo!("Handle structs!"),
                TypeKind::Union(_, _, _) => todo!("Handle unions!"),
                _ => (),
            }

            //regular struct member
            while !consume_if_equal(iter, TokenKind::Punct(Punct::Semicolon)){
                if !first {
                    ensure_and_consume(iter, TokenKind::Punct(Punct::Comma))?;
                }
                first = false;
                let decl = self.parse_declarator(iter, qty)?;
                
                let mut member = RecordMember {name: decl.name, is_bitfield: false, ty: qty.ty};
                
                if consume_if_equal(iter, TokenKind::Punct(Punct::Colon)) {
                    member.is_bitfield = true;
                    let width = self.parse_const_expression(iter)?.eval_int()?;
                    member.ty = member.ty.make_bitfield(width)?;
                } 
                members.push(member);
            }
        }

        Ok(members)
    }

    /// `enum-specifier ::= ( identifier ( '{' enumerator-list '}')? | '{' enumerator-list '}' )
    /// 
    /// `enumerator-list ::= enumerator ( ',' enumerator )*`
    /// 
    /// If the enum does not have a tag, then the values in the enum are merely treated as compile-time 
    /// constants. Tagged enums can be passed as in as function types. 
    /// 
    /// TODO: handle programmer-specified types for enums. 
    /// This is not explicitly part of the C11 spec but is a very common feature. 
    fn parse_enum_specifier(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<TypeInfo> {
        let token = consume_token_or_eof(iter)?;
        let ty = TypeInfo::make_enum();

        match token.token_type {
            //anonymous enum
            TokenKind::Punct(Punct::OpenBrace) => (),
            TokenKind::Ident(name) => {
                // enum has a tag

                // this assumes that the enum tag is just a typedef from the underlying type, and that
                // an implicit conversion is always possible. So values not defined in the enum would be 
                // valid as long as the underlying type is the same.
                // I believe this is the case, but it would be good to double-check the spec. 
                let node = ASTNode::new(ASTKind::Typedef(ty, name.to_string()), token.pos);
                self.syms.typedefs.push_symbol(
                    Symbol::new_local(&name, node)
                )?;
                ensure_and_consume(iter, TokenKind::Punct(Punct::OpenBrace))?;
            }
            _ => return Parser::gen_expected_one_of_error(&token, vec![TokenKind::Punct(Punct::OpenBrace), TokenKind::Ident("()".to_string())]),
        };
        
        self.parse_enumerator(iter)?;
        Ok(ty)
    }

    /// `enumerator ::= identifier ( '=' constant-expression )?
    /// 
    /// 
    fn parse_enumerator(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<()> {
        let mut iter = iter.peekable();
        let mut val = 0; //default value of enum
        let mut enum_type = TypeInfo::make_uint();
        while consume_if_equal(&mut iter, TokenKind::Punct(Punct::CloseBrace)) {
            let enumerator_label = consume_token_or_eof(&mut iter)?;
            if let TokenKind::Ident(label) = enumerator_label.token_type {
                if consume_if_equal(&mut iter, TokenKind::Punct(Punct::Assign)) {
                    // enum value has a custom value
                    let constexpr = self.parse_const_expression(&mut iter)?;
                    val = constexpr.eval_int()?;
                } else {
                    //give it an automatic value
                }
                let node = ASTNode::new(ASTKind::Int(val, 4), enumerator_label.pos); //TODO: the int size is hardcoded
                let enum_symbol = Symbol::new_local(&label, node);
                self.syms.identifiers.push_symbol(enum_symbol)?;
                val += 1;
                
            } else {
                return Parser::gen_expected_error(&enumerator_label, TokenKind::Ident("()".to_string()));
            }
        }

        Ok(())
    }

    /// Parse type definition, and add it to the symbol table.
    /// TODO: this implementation is probably incorrect, please revise.
    fn parse_typedef(&mut self, iter: &mut impl Iterator<Item=Token>) -> ParseRes<()> {
        let typeinfo = self.parse_declaration_specifier(iter)?;
        let token = iter.next().unwrap(); //TODO: error handling
        let typename = match token.token_type {
            TokenKind::Ident(s) => s,
            _ => return Err(Parser::gen_parser_err(ParseErrMsg::Something, &token)),
        };
        let node = ASTNode::new(ASTKind::Typedef(typeinfo.ty, typename), token.pos);
        let symbol = Symbol::new_local(&typename, node);
        self.syms.typedefs.push_symbol(symbol);

        Ok(())
    }
    
    /// Parse function calls. 
    /// 
    /// There is technically no "function-call" symbol in the C grammar. Instead, function calls are 
    /// an implicit part of the `postfix-expression` grammar. However, having a separate parsing function for function 
    /// calls is useful as the logic is rather complex and function calls generally have a unique representation 
    /// in the IR. 
    fn parse_func_call(&mut self, iter: &mut impl Iterator<Item = Token>, f: &Function) -> ParseRes<ASTNode> {
        let pos = extract_position(iter)?;
        let mut args = Vec::new();
        let mut first = true;
        if let TypeKind::Func(rettype, argtypes, is_vararg) = f.declarator.qty.ty.kind {
            while !consume_if_equal(iter, TokenKind::Punct(Punct::CloseParen)) {
                if !first {
                    ensure_and_consume(iter, TokenKind::Punct(Punct::Comma))?;
                }
                let arg = self.parse_assignment_expression(iter)?;
                //TODO: typechecking
                args.push(arg);
                if args.len() > argtypes.len() && !is_vararg {
                    return Err(Parser::gen_parser_err_pos(ParseErrMsg::TooManyArguments, &pos));
                }
                first = false;
            }

            let kind = ASTKind::FuncCall(f.body, args);
            Ok(ASTNode::new(kind, pos))
        } else {
            return Parser::gen_internal_error_pos(&pos, line!());
        }
    }

    /// `type-specifier ::= restrict | const | volatile`
    fn parse_type_specifier(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<Qualifiers> {
        let quals = Qualifiers::new();
        let mut iter = iter.peekable();

        while let Some(token) = iter.peek() {
            match token.token_type {
                TokenKind::Keyword(kw) => match kw {
                    Keyword::Restrict => quals.is_restrict = true,
                    Keyword::Const => quals.is_const = true,
                    Keyword::Volatile => quals.is_volatile = true,
                    _ => return Err(Parser::gen_parser_err(ParseErrMsg::Something, token)),
                }
                _ => break,
            }
        }

        Ok(quals)
    }

    /// Pointers are specified in declarators. Pointers can stack, so for example `***T` is 
    /// a pointer-to-a-pointer-to-a-pointer to `T`.
    /// 
    /// `pointer ::= '*' type-qualifier* pointer?`
    fn parse_pointer(&mut self, iter: &mut impl Iterator<Item = Token>, mut qty: QualifiedTypeInfo) -> ParseRes<QualifiedTypeInfo> {
        let mut iter = iter.peekable();
        let mut quals = Qualifiers::new();
        while let Some(token) = iter.peek() {
            match token.token_type {
                TokenKind::Punct(Punct::Asterisk(_)) => {
                    qty.ty = qty.ty.make_ptr();
                    iter.next();
                    
                    //skip qualifiers for now
                    quals = self.parse_type_specifier(&mut iter)?;
                    //TODO: handle qualifiers somehow

                },
                _ => break,
            }
        }
        qty.qualifiers = quals;
        Ok(qty)
    }
    
    /// Generic selections are prefaced with the keyword `_Generic`. 
    fn parse_generic_selection(&mut self, iter: &mut impl Iterator<Item = Token>){

    }

    ///
    /// `statement ::= labeled-statement | expression-statement | compound-statement 
    /// | selection-statement | iteration-statement | jump-statement | asm-statement`
    ///  
    /// 
    /// TODO: complete this documentation
    fn parse_statement(&'a mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<ASTNode> {
        let mut iter = iter.peekable_n();
        let tok = iter.peek();
        
        match tok.and_then(|t| Some(t.token_type)).unwrap_or(TokenKind::EOF) {
            
            TokenKind::Keyword(Keyword::If) |
            TokenKind::Keyword(Keyword::Switch) => self.parse_selection_stmt(&mut iter),
            
            TokenKind::Keyword(Keyword::For) |
            TokenKind::Keyword(Keyword::While) |
            TokenKind::Keyword(Keyword::Do) => self.parse_iteration_stmt(&mut iter),

            TokenKind::Keyword(Keyword::Default) |
            TokenKind::Keyword(Keyword::Case) => self.parse_labeled_stmt(&mut iter),
            TokenKind::Ident(_) => {
                //an identifier can either mean an expression statement or a labeled statement
                if let Some(token) = iter.peek_next() {
                    match token.token_type {
                        TokenKind::Punct(Punct::Colon) => self.parse_labeled_stmt(&mut iter),
                        _ => self.parse_expr_stmt(&mut iter),
                    }
                } else {
                    return Parser::gen_eof_error();
                }
            }
            
            TokenKind::Keyword(Keyword::Return) |
            TokenKind::Keyword(Keyword::Goto) |
            TokenKind::Keyword(Keyword::Break) |
            TokenKind::Keyword(Keyword::Continue) => self.parse_jump_stmt(&mut iter),

            TokenKind::Punct(Punct::OpenBrace) => self.parse_compound_stmt(&mut iter, None),

            TokenKind::Keyword(Keyword::Asm) => self.parse_asm_stmt(&mut iter),

            _ => self.parse_expr_stmt(&mut iter),
        }
    }

    ///`labeled-statement ::= identifier | 'case' constant-expression | 'default') ':' statement`
    fn parse_labeled_stmt(&'a mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<ASTNode> {
        if let Some(token) = iter.next() {
            let kind = match token.token_type {
                TokenKind::Ident(label) => ASTKind::Label(label),
                TokenKind::Keyword(Keyword::Case) => {
                    let constexpr = self.parse_const_expression(iter)?;
                    ASTKind::Case(Box::new(constexpr))
                },
                TokenKind::Keyword(Keyword::Default) => {
                    ASTKind::Default
                },
                _ => return Parser::gen_internal_error(&token, line!()),
            };

            ensure_and_consume(iter, TokenKind::Punct(Punct::Colon))?;
            let stmt = self.parse_statement(iter)?;
            //TODO: unclear what kind of AST node needs to be creared. 
            Ok(ASTNode::new(kind, token.pos))

        } else {
            Parser::gen_eof_error()
        }
    }

    /// Iteration statements handle loops. 
    /// 
    /// `iteration-statement ::= 
    /// ( 'while' '(' 
    /// | 'for' '(' expression? ';' expression? ';' ) expression ')' statement 
    /// | 'do' statement 'while' '(' expression ')' ';'`
    fn parse_iteration_stmt(&'a mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<ASTNode> {
        let token = consume_token_or_eof(iter)?;
        let kind = match token.token_type {
            TokenKind::Keyword(Keyword::While) => {
                ensure_and_consume(iter, TokenKind::Punct(Punct::OpenParen))?;
                let expr = Box::new(self.parse_expression(iter)?);
                ensure_and_consume(iter, TokenKind::Punct(Punct::CloseParen))?;
                let loop_body = Box::new(self.parse_statement(iter)?);
                ASTKind::While{has_do: false, cond: expr, body: loop_body}
            },
            TokenKind::Keyword(Keyword::For) => {
                ensure_and_consume(iter, TokenKind::Punct(Punct::OpenParen))?;
                
                let init = if check_if_equal(iter, TokenKind::Punct(Punct::Semicolon)) {
                    Box::new(self.parse_expression(iter)?)
                } else {
                    let pos = extract_position(iter)?;
                    Box::new(ASTNode::new(ASTKind::Noop, pos))
                };

                ensure_and_consume(iter, TokenKind::Punct(Punct::Semicolon))?;

                let cond = if check_if_equal(iter, TokenKind::Punct(Punct::Semicolon)) {
                    Box::new(self.parse_expression(iter)?)
                } else {
                    //According to 6.8.5.3 the conditional expression is replaced by a nonzero constant if it is not 
                    //present.
                    let pos = extract_position(iter)?;
                    Box::new(ASTNode::new(ASTKind::Int(1, 1), pos))
                };

                ensure_and_consume(iter, TokenKind::Punct(Punct::Semicolon))?;
                let step = if check_if_equal(iter, TokenKind::Punct(Punct::CloseParen)) {
                    Box::new(self.parse_expression(iter)?)
                } else {
                    let pos = extract_position(iter)?;
                    Box::new(ASTNode::new(ASTKind::Noop, pos))
                };

                ensure_and_consume(iter, TokenKind::Punct(Punct::CloseParen))?;
                let body = Box::new(self.parse_statement(iter)?);
                ASTKind::For { init: init, cond: cond , step: step, body: body }
            },
            TokenKind::Keyword(Keyword::Do) => {
                let loop_body = Box::new(self.parse_statement(iter)?);
                ensure_and_consume(iter, TokenKind::Keyword(Keyword::While))?;
                ensure_and_consume(iter, TokenKind::Punct(Punct::OpenParen))?;
                let expr = Box::new(self.parse_expression(iter)?);
                ensure_and_consume(iter, TokenKind::Punct(Punct::CloseParen))?;
                ensure_and_consume(iter, TokenKind::Punct(Punct::Semicolon))?;
                ASTKind::While{has_do: true, cond: expr, body: loop_body}
            },
            _ => return Parser::gen_internal_error(&token, line!()),
        };

        Ok(ASTNode::new(kind, token.pos))
    }

    /// `selection-statement ::= ( 'if' '(' expression ')' ( statement 'else' )? | 'switch' '(' expression ')' ) statement
    /// Selection statements cover branching statements, specifically if-else and switch statements.
    fn parse_selection_stmt(&'a mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<ASTNode> {
        let token = consume_token_or_eof(iter)?;
        let kind = match token.token_type {
            TokenKind::Keyword(Keyword::If) => {
                ensure_and_consume(iter, TokenKind::Punct(Punct::OpenParen))?;
                let expr = self.parse_expression(iter)?;
                ensure_and_consume(iter, TokenKind::Punct(Punct::CloseParen))?;
                let then = self.parse_statement(iter)?;
                let mut els = None;
                //check if there is an else branch
                if consume_if_equal(iter, TokenKind::Keyword(Keyword::Else)) {
                    els = Some(Box::new(self.parse_statement(iter)?));
                }
                ASTKind::If { predicate: Box::new(expr), then: Box::new(then), els: els }
            },
            TokenKind::Keyword(Keyword::Switch) => {
                ensure_and_consume(iter, TokenKind::Punct(Punct::OpenParen))?;
                let expr = self.parse_expression(iter)?;
                ensure_and_consume(iter, TokenKind::Punct(Punct::CloseParen))?;
                let cases = self.parse_statement(iter)?;
                ASTKind::Switch { cond: Box::new(expr), cases: Box::new(cases) }
            },
            //this should never happen
            _ => return Parser::gen_internal_error(&token, line!())
        };

        Ok(ASTNode::new(kind, token.pos))
    }
    
    /// `expression-statement ::= expression? ';'
    /// Most lines in a well-formed C program will be expression statements. 
    fn parse_expr_stmt(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<ASTNode> {
        let pos = extract_position(iter)?;
        if consume_if_equal(iter,TokenKind::Punct(Punct::Semicolon)){
            Ok(ASTNode::new(ASTKind::Noop, pos))
        } else {
            let expr_res = self.parse_expression(iter)?;
            ensure_and_consume(iter, TokenKind::Punct(Punct::Semicolon))?;
            Ok(expr_res)
        }
    } 
    /// `jump-statement ::= ('goto' identifier | 'continue' | 'break' | 'return' expression+ ) ';'
    fn parse_jump_stmt(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<ASTNode> {
        //calling unwrap is safe because the iterator has already been peeked for nonemptiness
        let token = iter.next().unwrap();
        let kind = match token.token_type { 
            TokenKind::Keyword(Keyword::Continue) => ASTKind::Continue,
            TokenKind::Keyword(Keyword::Break) => ASTKind::Break,
            TokenKind::Keyword(Keyword::Return) => {
                let mut iter = iter.peekable();
                if let Some(tok) = iter.peek() {
                    if tok.token_type == TokenKind::Punct(Punct::Semicolon) {
                        let node = self.parse_expression(&mut iter)?;
                        ASTKind::Return(Some(Box::new(node)))
                    } else {
                        ASTKind::Return(None)
                    }
                } else {
                    return Parser::gen_eof_error();
                }
            },
            TokenKind::Keyword(Keyword::Goto) => {
                let token = consume_token_or_eof(iter)?;
                match token.token_type {
                    TokenKind::Ident(id) => ASTKind::Goto(id),
                    _ => return Parser::gen_expected_error(&token, TokenKind::Ident("".to_string())),
                }
            }
            _ => return Parser::gen_expected_one_of_error(
                &token, 
                vec![
                    TokenKind::Keyword(Keyword::Continue), 
                    TokenKind::Keyword(Keyword::Break),
                    TokenKind::Keyword(Keyword::Return),
                    TokenKind::Keyword(Keyword::Goto) ]
                ),
        };

        ensure_and_consume(iter, TokenKind::Punct(Punct::Semicolon))?;

        Ok(ASTNode::new(kind, token.pos))
    }


    /// Section J.5.10
    /// `asm-statement = "asm" ("volatile" | "inline")* "(" string-literal ")" 
    fn parse_asm_stmt(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<ASTNode> {
        ensure_and_consume(iter, TokenKind::Keyword(Keyword::Asm))?;
        let iter = iter.skip_while(
            |token| 
            token.token_type == TokenKind::Keyword(Keyword::Volatile) ||
            token.token_type == TokenKind::Keyword(Keyword::Inline)
        );
        
        ensure_and_consume(&mut iter, TokenKind::Punct(Punct::OpenParen))?;
        let asm = consume_token_or_eof(&mut iter)?;
        let asm_code = match asm.token_type {
            TokenKind::Str(s) => s,
            _ => return Parser::gen_expected_error(&asm, TokenKind::Str("".to_string())),
        };

        let node = ASTNode::new(ASTKind::Asm(asm_code), asm.pos);

        ensure_and_consume(&mut iter, TokenKind::Punct(Punct::CloseParen))?;

        Ok(node)
    }

    /// `compound-statement ::= '{' statement* '}'`
    /// 
    /// TODO: more sophisticated error handling. Currently if there is an error in any of the 
    /// statements/declarations then the parsing fails. Instead it should collect all of the errors and 
    /// display them afterwards. 
    fn parse_compound_stmt(&'a mut self, iter: &mut impl Iterator<Item = Token>, scope: Option<ActiveScope<'a>> ) -> ParseRes<ASTNode> {
        let pos = extract_position(iter)?;
        let mut iter = iter.peekable();
        let mut stmts = Vec::new();
        
        let scope = if scope.is_some() {
            scope.unwrap()
        } else {
            self.syms.identifiers.enter_scope()
        };

       
        loop {
            if consume_if_equal(&mut iter, TokenKind::Punct(Punct::CloseBrace)) {
                break;
            }

            let peek_token = iter.peek().unwrap(); //TODO: error handling in case of EOF
            if self.is_type_specifier(&peek_token.token_type) {
                stmts.push(self.parse_declaration(&mut iter)?);
            } else {
                stmts.push(self.parse_statement(&mut iter)?); //TODO: more sophisticated error handling. 
            }
        }

        Ok(ASTNode::new(ASTKind::Block(stmts), pos))
    }
    
    #[inline]
    fn gen_parser_err(msg: ParseErrMsg, token: &Token) -> ParseErr {
        ParseErr::new(token.pos, token.pos, msg)
    }
    
    #[inline]
    fn gen_eof_error<T>() -> ParseRes<T> {
        Err(Parser::gen_parser_err(ParseErrMsg::EOF,&Token::eof_token(None)))
    }

    #[inline]
    fn gen_expected_error<T>(tk: &Token, expected: TokenKind) -> ParseRes<T> {
        Err(Parser::gen_parser_err(ParseErrMsg::ExpectedSymbol(expected), tk))
    }

    #[inline]
    fn gen_internal_error<T>(tk: &Token, line: u32) -> ParseRes<T> {
        Err(Parser::gen_parser_err(ParseErrMsg::InternalError(line), tk))
    }

    #[inline]
    fn gen_internal_error_pos<T>(pos: &TextPosition, line: u32) -> ParseRes<T> {
        Err(Parser::gen_parser_err_pos(ParseErrMsg::InternalError(line), pos))
    }

    #[inline]
    fn gen_expected_one_of_error<T>(tk: &Token, expected: Vec<TokenKind>) -> ParseRes<T> {
        Err(Parser::gen_parser_err(ParseErrMsg::ExpectedOneOfSymbols(expected), tk))
    }

    #[inline]
    fn gen_parser_err_pos(msg: ParseErrMsg, pos: &TextPosition) -> ParseErr {
        ParseErr::new(pos.clone(), pos.clone(), msg)
    }

    /// Determine if a token stream is a function definition.
    /// 
    fn is_func_def(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<bool> {
        let mut is_funcdef = false;
        let mut iter = iter.peekable_n();
        let mut is_first_loop = true;
        while let Some(token) = iter.peek() {
            if !is_first_loop {
                iter.advance_cursor();
            }
            is_first_loop = false;

            if token.token_type == TokenKind::Punct(Punct::Semicolon) {
                break;
            }

            if self.is_type_specifier(&token.token_type){
                continue;
            }
            
            if !matches!(token.token_type, TokenKind::Ident(_)){
                continue;
            }
            
            if iter.peek().unwrap().token_type != TokenKind::Punct(Punct::OpenParen) {
                continue;
            }
            
            _ = iter.advance_cursor_while(|tok| tok.token_type != TokenKind::Punct(Punct::CloseParen));
            let tok = match iter.peek_next() {
                Some(tok) => tok,
                None => return Parser::gen_eof_error(),
            };

            is_funcdef = tok.token_type == TokenKind::Punct(Punct::OpenBrace);
            break;
        }

        iter.reset_cursor();
        Ok(is_funcdef)
    }

    /// Determine if a keyword denotes a typename, a type qualifier or a type storage specifier. This includes user-generated
    /// types.
    fn is_type_specifier(&self, token_type: &TokenKind) -> bool {
        match token_type  {
            TokenKind::Keyword(kw) => match kw {
                Keyword::Void   | Keyword::Bool     | Keyword::Const  | Keyword::ThreadLocal | 
                Keyword::Atomic | Keyword::Double   | Keyword::Static | Keyword::Short  | Keyword::Int |
                Keyword::Float  | Keyword::Volatile | Keyword::Typeof | Keyword::Inline |
                Keyword::Register | Keyword::Enum | Keyword::Struct | Keyword::Char | Keyword::Thread |
                Keyword::NoReturn | Keyword::Union | Keyword::Restrict | Keyword::Unsigned | Keyword::Signed |
                Keyword::Auto => true,
                _ => false,
            }
            TokenKind::Ident(name) => if self.syms.typedefs.get_symbol(&name).is_some() {
                true
            } else {
                false
            },
            _ => false
        }
    }

    /// `declaration-specifier ::= storage-class-specifier | type-specifier | type-qualifier`
    /// 
    /// The ordering of specifiers does not matter. For example, `static char inline` is the same as 
    /// `inline char static`. If `long` or `short` are used, then `int` can be omitted from the type specifier.
    /// However, some combinations are invalid, and those need to be tracked.
    /// 
    /// TODO: complete this 
    fn parse_declaration_specifier(&mut self, iter: &mut impl Iterator<Item = Token>) -> ParseRes<QualifiedTypeInfo> {
        //TODO: potentially rework this.
        //these constants are used in a trick found in chibicc in order to concisely determine
        //the builtin type based on counting the keywords. 
        //this is compact and easy but may not be suitable for semantic analysis and sophisticated error reporting.
        const VOID: i32 = 1<<0;
        const BOOL: i32 = 1<<2;
        const CHAR: i32 = 1<<4;
        const SHORT: i32 = 1<<6;
        const INT: i32 = 1<<8;
        const LONG: i32 = 1<<10;
        const FLOAT: i32 = 1<<12;
        const DOUBLE: i32 = 1<<14;
        const OTHER: i32  = 1<<16;
        const SIGNED: i32 = 1<<17;
        const UNSIGNED: i32 = 1<<18;
        
        let mut iter = iter.peekable_n();
        let mut counter = 0;
        // hand
        let mut qty = QualifiedTypeInfo::new();
        let mut quals = Qualifiers::new();
        let mut storage_class_defined = false;
        let mut ty = TypeInfo::new();
        let mut tc = TypeCounter::default();
        while let Some(token) = iter.peek() {

            if self.is_type_specifier(&token.token_type) {

                //handle storage class specifiers
                match Parser::read_storage_class_specifier(token, true) {
                    Some(storage) => {
                        //storage class specifiers are mutually exclusive. If more than one is defined, return an error. 
                        if qty.storage != StorageClass::UNUSED {
                            return Err(Parser::gen_parser_err(ParseErrMsg::Something, token))

                        } else {
                            qty.storage = storage;
                            iter.next();
                            continue;
                        }
                    }
                    None => ()
                }

                //handle type qualifiers
                let mut match_found = false;
                let set_flag = |b: &mut bool | {*b = true; match_found = true;};
                match token.token_type {
                    //TODO: may need to add a continue when a match is found
                    TokenKind::Keyword(kw) => match kw {
                        Keyword::Const => set_flag(&mut quals.is_const),
                        Keyword::NoReturn => set_flag(&mut quals.is_noreturn),
                        Keyword::Inline => set_flag(&mut quals.is_inline),
                        Keyword::Volatile => set_flag(&mut quals.is_volatile),
                        Keyword::Restrict => set_flag(&mut quals.is_restrict),
                        Keyword::ThreadLocal => set_flag(&mut quals.is_tls),
                        _ => (),
                    },
                    _ => (),
                }
                if match_found {
                    iter.next();
                    continue;
                }

                //handle user defined types
                todo!("Handle user defined types");
                match token.token_type {
                    TokenKind::Ident(name) => {
                        if let Some(symbol) = self.syms.typedefs.get_symbol(&name)  {
                            qty.ty.kind = symbol.typeinfo.kind;
                        } else {
                            ()
                        }
                    },
                    TokenKind::Keyword(kw) => match kw {
                        Keyword::Struct => self.parse_struct_union_specifier(&mut iter), //struct declaration
                        Keyword::Union => self.parse_struct_union_specifier(&mut iter),
                        Keyword::Enum => self.parse_enum_specifier(&mut iter),
                        Keyword::Typeof => (),
                        _ => (),
                    },
                    _ => (),
                }

                //handle builtin types
                match token.token_type {
                    TokenKind::Keyword(kw) => match kw {
                        Keyword::Void => counter += VOID,
                        Keyword::Bool => counter += BOOL,
                        Keyword::Char => counter += CHAR,
                        Keyword::Short => counter += SHORT,
                        Keyword::Int => counter += INT,
                        Keyword::Long => counter += LONG,
                        Keyword::Float => counter += FLOAT,
                        Keyword::Double => counter += DOUBLE,
                        Keyword::Signed => counter |= SIGNED,
                        Keyword::Unsigned => counter |= UNSIGNED,
                        _ => todo!("Handle all builtin types"),
                    },
                    _ => (),
                }

                match counter {
                    VOID => ty = TypeInfo::make_void(),
                    BOOL => ty = TypeInfo::make_bool(),
                    CHAR => ty = TypeInfo::make_char(),                    
                }


            } else {
                //we are no longer in the declaration specifier
                break;
            }
        }
        qty.qualifiers = quals;
        todo!("Determine return type of this function");

        Ok(qty)
    }

    /// Storage class specifiers are one of: `static`, `register`, `extern`, `auto`, `enum`, `typedef`. 
    /// 
    fn read_storage_class_specifier(token: &Token, is_funcdef: bool) -> Option<StorageClass> {
        let sc = match token.token_type {
            TokenKind::Keyword(Keyword::Static) => StorageClass::Static,
            TokenKind::Keyword(Keyword::Register) => StorageClass::Register,
            TokenKind::Keyword(Keyword::Extern) => StorageClass::Extern,
            TokenKind::Keyword(Keyword::Auto) => StorageClass::Auto,
            TokenKind::Keyword(Keyword::Enum) => StorageClass::Enum,
            TokenKind::Keyword(Keyword::Typedef) => StorageClass::Typedef,
            _ => return None,
        };
        
        Some(sc)

    }
    ///Parse a token stream, transforming it into an AST. 
    pub fn parse(&'a self, tokens: Vec<Token>) -> ParseRes<Vec<ASTNode>> {
        let iter = tokens.into_iter().peekable();
        let nodes = Vec::new();
        while let Some(tok) = iter.peek() {
            nodes.append(&mut self.parse_translation_unit(&mut iter)?);
        }

        Ok(nodes)
    }
}

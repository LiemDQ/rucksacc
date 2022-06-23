
use crate::ast::{ASTKind, ASTNode};
use crate::lex::token::{TokenKind, Keyword, Punct, PMKind, AstKind, AmpKind};
use crate::err::{ParseRes};

/// Determine whether an AST node is a constant expression or not. 
/// 
pub fn is_const_expr(node: &ASTNode) -> ParseRes<bool> {
    Ok(match &node.kind {
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
pub fn postfix_binding_power(tok: &TokenKind) -> Option<(u8, ())> {
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

pub fn prefix_binding_power(tok: &TokenKind) -> Option<((), u8)> {
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
pub fn infix_binding_power(tok: &TokenKind) -> Option<(u8, u8)> {
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
            Punct::AssignShl | Punct::AssignShr |
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

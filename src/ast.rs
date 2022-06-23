use crate::lex::{Token};
use crate::err::{ParseRes, ParseErr, ParseErrMsg};
use crate::utils::{TextPosition};
use crate::types::{TypeKind, TypeInfo, StorageClass};
use crate::types;
use crate::parse::parser::{Declarator};



pub struct Entity {
    name: String, 
    qualifiers: types::Qualifiers,
    storage: types::StorageClass,
    kind: EntityKind
}

pub enum EntityKind {
    LocalVar{offset: i32},
    Function{params: Box<Entity>, locals: Box<Entity>, stack_size: u32},
    GlobalVar{is_tentative: bool, is_tls: bool, init_data: String, },
    StaticInlineFunction{is_live: bool, is_root: bool, refs: String}, //TODO: handle this case
}

/// Global variables can be initialized either by a constant expression or by a pointer to
/// another global variable. This struct represents the latter case. 
struct Relocation {
    offset: u32,
    label: Vec<String>,
    addend: usize,
}

#[derive(Debug, Clone)]
pub struct ASTNode {
    pub kind: ASTKind,
    pub pos: TextPosition,
}

/// Internal AST representation for the compiler. 
/// This IR combines a low-level AST with a basic block CFG approach. 
#[derive(Debug, Clone)]
pub enum ASTKind {
    Typedef(TypeInfo, String), //(original type, new type name)
    Noop,
    Break,
    Continue,
    Default,
    Load(Box<ASTNode>), //represents memory jumps (e.g. struct member accesses, pointer derefs)
    InitArray(Vec<ASTNode>),
    InitStruct(Vec<ASTNode>),
    Switch{cond: Box<ASTNode>, cases: Box<ASTNode>},
    Case(Box<ASTNode>),
    UnaryOp(UnaryExpr),
    BinaryOp(BinaryExpr),
    TertiaryOp(TertiaryExpr),
    Block(ScopeBlock),
    Variable(Var),
    VariableDecl(Var, StorageClass, Option<Box<ASTNode>>), //variable, storage class, initial value
    Int(i64, i8), //val, size
    Float(f64, i8), //val, size
    String(String),
    Char(i16), //val
    Func(Function), //function definition
    FuncCall(Box<ASTNode>, Vec<ASTNode>), //(function definition, arguments)
    StructRef(Box<ASTNode>, String), //string is name of struct field
    If{predicate: Box<ASTNode>, then: Box<ASTNode>, els: Option<Box<ASTNode>>,},
    For{init: Box<ASTNode>,cond: Box<ASTNode>,step: Box<ASTNode>,body: Box<ASTNode>},
    While{has_do: bool, cond: Box<ASTNode>, body: Box<ASTNode>},
    Label(String), //label name
    Goto(String), //label
    Asm(String), //asm sequence
    Cast(Box<ASTNode>, TypeKind), //from, to. TODO: determine whether TypeInfo should be used instead. 
    CAS, //atomic compare-and-swap
    Exch, //atomic exchange
    Return(Option<Box<ASTNode>>), //return expression, if any
}

impl ASTNode {
    pub fn new(kind: ASTKind, pos: TextPosition) -> Self {
        Self { kind: kind, pos: pos }
    }

    pub fn make_variable_declaration(decl: Declarator, tok: &Token) -> Self {
        let var = Var {name: decl.name, ty: decl.qty.ty, offset: 0}; //TODO: handle offset
        Self {
            kind: ASTKind::VariableDecl(var, decl.qty.storage, None),
            pos: tok.pos.clone(),
        }
    }

    pub fn make_assignment(lhs: ASTNode, rhs: ASTNode) -> Self {
        let pos = lhs.pos.clone();
        let binexpr = BinaryExpr {lhs: Box::new(lhs), op: BinaryOps::Assign, rhs: Box::new(rhs)};
        let kind = ASTKind::BinaryOp(binexpr);
        Self { kind: kind, pos: pos }
    }

    /// When the node is a binary expression
    pub fn reassign_rhs(mut self, rhs: ASTNode) -> ParseRes<Self> {
        match &mut self.kind {
            ASTKind::BinaryOp(ref mut bexp) => match bexp.op {
                BinaryOps::Assign => {
                    bexp.rhs = Box::new(rhs);
                    Ok(self)
                },
                _ => Err(self.gen_err(ParseErrMsg::Something)),
            }
            _ => Err(self.gen_err(ParseErrMsg::Something))
        }
    }
    
    /// Evaluate a constant expression.
    /// 
    /// A constant expression resolves to a number or a ptr+n where ptr is
    /// a pinter to a global variable and n is an integer. The latter is only acceptable
    /// for an initialization expression for a global variable. 
    pub fn eval_int(&self) -> ParseRes<i64> {
        let result = match &self.kind {
            ASTKind::Float(_, _) => todo!("Error handling for eval"),//should never happen
            ASTKind::Int(n, _) => *n, //TODO: this doesn't handle type conversions properly
            ASTKind::Cast(node, _) => node.eval_int()?,
            ASTKind::UnaryOp(op) => self.eval_unary_op(&op)?,
            ASTKind::BinaryOp(op) => self.eval_binary_op_int(&op)?,
            ASTKind::TertiaryOp(op) => self.eval_tertiary_op_int(&op)?,
            _ => todo!("Error handling for eval"),
        };

        Ok(result)
    }

    fn eval_unary_op(&self, op_expr: &UnaryExpr) -> ParseRes<i64> {
        Ok(match op_expr.op {
            UnaryOps::LNot => (op_expr.id.eval_int()? == 0) as i64,
            UnaryOps::BNot => !op_expr.id.eval_int()?,
            UnaryOps::Neg => -op_expr.id.eval_int()?,
            UnaryOps::IncPostfix | UnaryOps::IncPrefix => op_expr.id.eval_int()? + 1,
            UnaryOps::DecPostfix | UnaryOps::DecPrefix => op_expr.id.eval_int()? - 1,
            UnaryOps::Deref => op_expr.id.eval_int()?, //TODO: not sure if deref is just a no-op in this context
            UnaryOps::Addr => op_expr.id.eval_int()?,
            UnaryOps::Alignof => op_expr.id.alignof()?,
            UnaryOps::Sizeof => op_expr.id.sizeof()?,
        })
    }

    fn eval_binary_op_int(&self, binop: &BinaryExpr) -> ParseRes<i64> {
        let n = match binop.op {
            BinaryOps::Add => binop.lhs.eval_int()? + binop.rhs.eval_int()?,
            BinaryOps::Sub => binop.lhs.eval_int()? - binop.rhs.eval_int()?,
            BinaryOps::Div => binop.lhs.eval_int()? / binop.rhs.eval_int()?,
            BinaryOps::Mult=> binop.lhs.eval_int()? * binop.rhs.eval_int()?,
            BinaryOps::Mod => binop.lhs.eval_int()? % binop.rhs.eval_int()?,
            BinaryOps::BAnd=> binop.lhs.eval_int()? & binop.rhs.eval_int()?,
            BinaryOps::BOr => binop.lhs.eval_int()? | binop.rhs.eval_int()?,
            BinaryOps::BXor=> binop.lhs.eval_int()? ^ binop.rhs.eval_int()?,
            BinaryOps::LOr => (binop.lhs.eval_int()? != 0 || binop.rhs.eval_int()? != 0) as i64,
            BinaryOps::LAnd=> (binop.lhs.eval_int()? != 0 && binop.rhs.eval_int()? != 0) as i64,
            BinaryOps::Eq  => (binop.lhs.eval_int()? == binop.rhs.eval_int()?) as i64,
            BinaryOps::Ne  => (binop.lhs.eval_int()? != binop.rhs.eval_int()?) as i64,
            BinaryOps::Lt  => (binop.lhs.eval_int()? < binop.rhs.eval_int()?) as i64,
            BinaryOps::Gt  => (binop.lhs.eval_int()? > binop.rhs.eval_int()?) as i64,
            BinaryOps::Le  => (binop.lhs.eval_int()? <= binop.rhs.eval_int()?) as i64,
            BinaryOps::Ge  => (binop.lhs.eval_int()? >= binop.rhs.eval_int()?) as i64,
            BinaryOps::Shl => binop.lhs.eval_int()? << binop.rhs.eval_int()?,
            BinaryOps::Shr => binop.lhs.eval_int()? >> binop.rhs.eval_int()?,
            BinaryOps::Comma => binop.lhs.eval_int()?, //according to the C spec, the first item in the argument should be evaluated. 
            BinaryOps::Assign => return Err(self.gen_err(ParseErrMsg::Something)), //cannot have assignment expression inside a constexpr    
        };

        Ok(n)
    }

    fn eval_tertiary_op_int(&self, op: &TertiaryExpr) -> ParseRes<i64> {
        match op.op {
            TertiaryOps::Conditional => if op.lhs.eval_int()? != 0 {
                op.mid.eval_int()
            } else {
                op.rhs.eval_int()
            }
        }        
    }
    
    //evaluate floating-point operations
    pub fn eval_float(&self) -> ParseRes<f64> {
        let result = match &self.kind {
            ASTKind::Float(n, _) => *n,
            ASTKind::Int(n, _) => *n as f64, 
            ASTKind::Cast(node, _) => node.eval_float()?,
            ASTKind::UnaryOp(op) => return Err(self.gen_err(ParseErrMsg::InvalidFloatOperation)),
            ASTKind::BinaryOp(op) => self.eval_binary_op_float(&op)?,
            _ => todo!("Error handling for eval"),
        };

        Ok(result)
    }

    pub fn eval_binary_op_float(&self, binop: &BinaryExpr) -> ParseRes<f64> {
        let n = match binop.op {
            BinaryOps::Add => binop.lhs.eval_float()? + binop.rhs.eval_float()?,
            BinaryOps::Sub => binop.lhs.eval_float()? - binop.rhs.eval_float()?,
            BinaryOps::Div => binop.lhs.eval_float()? / binop.rhs.eval_float()?,
            BinaryOps::Mult=> binop.lhs.eval_float()? * binop.rhs.eval_float()?,
            BinaryOps::Mod => binop.lhs.eval_float()? % binop.rhs.eval_float()?,
            //bitwise operations are invalid for floats. 
            BinaryOps::BAnd=> return Err(self.gen_err(ParseErrMsg::InvalidFloatOperation)),
            BinaryOps::BOr => return Err(self.gen_err(ParseErrMsg::InvalidFloatOperation)),
            BinaryOps::BXor=> return Err(self.gen_err(ParseErrMsg::InvalidFloatOperation)),
            //logical operations should be converted before making it to this step.
            BinaryOps::LOr => return Err(self.gen_err(ParseErrMsg::InvalidFloatOperation)),
            BinaryOps::LAnd=> return Err(self.gen_err(ParseErrMsg::InvalidFloatOperation)),
            BinaryOps::Eq  => return Err(self.gen_err(ParseErrMsg::InvalidFloatOperation)),
            BinaryOps::Ne  => return Err(self.gen_err(ParseErrMsg::InvalidFloatOperation)),
            BinaryOps::Lt  => return Err(self.gen_err(ParseErrMsg::InvalidFloatOperation)),
            BinaryOps::Gt  => return Err(self.gen_err(ParseErrMsg::InvalidFloatOperation)),
            BinaryOps::Le  => return Err(self.gen_err(ParseErrMsg::InvalidFloatOperation)),
            BinaryOps::Ge  => return Err(self.gen_err(ParseErrMsg::InvalidFloatOperation)),
            BinaryOps::Shl => return Err(self.gen_err(ParseErrMsg::InvalidFloatOperation)),
            BinaryOps::Shr => return Err(self.gen_err(ParseErrMsg::InvalidFloatOperation)),
            BinaryOps::Comma => binop.lhs.eval_float()?, //according to the C spec, the first item in the argument should be evaluated. 
            BinaryOps::Assign => return Err(self.gen_err(ParseErrMsg::Something)), //cannot have assignment expression inside a constexpr    
        };

        Ok(n)
    }

    pub fn eval_tertiary_op_float(&self, op: &TertiaryExpr) -> ParseRes<f64> {
        match op.op {
            TertiaryOps::Conditional => if op.lhs.eval_float()? != 0. {
                op.mid.eval_float()                
            } else {
                op.rhs.eval_float()
            }
        }
    }

    fn sizeof(&self) -> ParseRes<i64> {
        Ok(match &self.kind {
            ASTKind::Int(_, n) => *n as i64,
            ASTKind::Float(_, n) => *n as i64,
            ASTKind::Char(_) => 1,
            ASTKind::Variable(v) => v.ty.size,
            _ => return Err(self.gen_err(ParseErrMsg::Something)),
        })
    }

    fn alignof(&self) -> ParseRes<i64> {
        Ok(match &self.kind {
            ASTKind::Int(_, n) => *n as i64,
            ASTKind::Float(_, n) => *n as i64,
            ASTKind::Char(_) => 1,
            ASTKind::Variable(v) => v.ty.align,
            _ => return Err(self.gen_err(ParseErrMsg::Something)),
        })
    }

    fn gen_err(&self, msg: ParseErrMsg) -> ParseErr {
        ParseErr::new(self.pos.clone(), self.pos.clone(), msg)
    }
}

#[derive(Debug, Clone)]
pub struct IntExpr {
    pub val : i64,
    pub size: i8,
}

#[derive(Debug, Clone)]
pub struct FloatExpr {
    pub val: f64,
    pub size: i8,
}

#[derive(Debug, Clone)]
pub struct Var {
    pub name: String,
    pub ty: TypeInfo,
    pub offset: i32,

}

#[derive(Debug, Clone)]
pub struct SwitchBlock {
    cond: Box<ASTNode>,
    cases: Box<ASTNode>,
}

#[derive(Debug, Clone)]
pub enum UnaryOps {
    Sizeof,
    Alignof,
    Deref,
    IncPostfix, //a++
    DecPostfix, //a--
    IncPrefix, //++a
    DecPrefix, //--a
    Neg,
    LNot, //logical not
    BNot, //bitwise not
    Addr,
}

#[derive(Debug, Clone)]
pub enum BinaryOps {
    Add,
    Sub,
    Div,
    Mult,
    Mod,
    BAnd,
    BOr,
    BXor,
    LOr,
    LAnd,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Shl,
    Shr,
    Comma,
    Assign,    
}

#[derive(Debug, Clone)]
pub enum TertiaryOps {
    Conditional,
}

#[derive(Debug, Clone)]
pub struct UnaryExpr {
    pub id: Box<ASTNode>,
    pub op: UnaryOps,
}

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    pub lhs: Box<ASTNode>,
    pub rhs: Box<ASTNode>,
    pub op: BinaryOps,
}


#[derive(Debug, Clone)]
pub struct TertiaryExpr {
    pub lhs: Box<ASTNode>,
    pub mid: Box<ASTNode>,
    pub rhs: Box<ASTNode>,
    pub op: TertiaryOps, 
}

#[derive(Debug, Clone)]
pub struct ConditionalBlock {
    predicate: Box<ASTNode>,
    then: Box<ASTNode>,
    els: Option<Box<ASTNode>>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub declarator: Declarator, 
    pub body: Box<ASTNode>,
    pub stack_size: i32,
}

pub type ScopeBlock = Vec<ASTNode>;

use crate::lex::{AstKind, Token};
use crate::err::{ParseRes, ParseErr, ParseErrMsg};
use crate::utils::{TextPosition};
use crate::types::{TypeKind, TypeInfo, StorageClass, Qualifiers};
use crate::types;
use crate::parse::{Declarator, DeclaratorTypes};

use std::collections::{HashMap};
/// NOTE: It's a little unclear to me how this will interact with the actual AST nodes,
/// or whether it would be more appropriate to store a reference to those nodes instead.
/// This would require some restructuring, but as it stands, it's a little unclear to me
/// how they would interact and if there is overlapping data that is being stored in the nodes
/// vs stored in these symbols. 
#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub position: TextPosition,
    //pub scope: ScopeKind,
    pub sclass: StorageClass,
    pub node: ASTNode,
}

impl Symbol {
    pub fn new_local(name: &str, node: ASTNode) -> Self {
        Self { name: name.to_string(), position: node.pos.clone(), sclass: StorageClass::Auto, node: node }
    }
}

/// Table of symbols. This table is filled up during the parse step and is used for subsequent parsing operations, during code generation
/// and for resolving ambiguous aspects of the C grammar. 
/// 
/// Each translation unit has its own symbol table. 
/// 
/// C scopes are modeled using [parent pointer trees](https://en.wikipedia.org/wiki/Parent_pointer_tree) 
/// which allow higher scopes to be accessed from lower scopes. This is difficult to model idiomatically in Rust 
/// as it would lead to circular references. Instead, we use a basic arena-based approach, using a Vec instead, 
/// with the table owning all scopes. This leads to a relatively simple implementation, particularly since we do
/// not remove scopes once they are added to the table.  
///  
/// The global scope is always at index 0.
#[derive(Debug, Clone)]
pub struct SymbolTable {
    // pub global: Rc<Cell<SymbolScope>>,
    // pub active_sc: Rc<Cell<SymbolScope>>,
    scopes: Vec<SymbolScope>,
    active_scope_idx: usize,
}


impl SymbolTable {
    pub fn new() -> Self {
        Self {
            scopes: vec![SymbolScope::new(0)],
            active_scope_idx: 0
        }
    }

    pub fn global(&self) -> &SymbolScope {
        //we are guaranteed to have a scope at index 0 since it is created when the table is initialized, so unchecked unwrapping is fine
        self.scopes.get(0).unwrap()
    }

    pub fn global_mut(&mut self) -> &mut SymbolScope {
        self.scopes.get_mut(0).unwrap()
    }

    pub fn push_scope(&mut self, mut sc: SymbolScope){
        let idx = self.scopes.len();
        sc.index = idx;
        sc.depth = self.active_scope().depth + 1;
        sc.up = Some(self.active_scope_idx);
        self.scopes.push(sc);
        self.active_scope_idx = idx; 
    }

    fn active_scope(&self) -> &SymbolScope {
        self.scopes.get(self.active_scope_idx).unwrap()
    }

    fn active_scope_mut(&mut self) -> &mut SymbolScope {
        self.scopes.get_mut(self.active_scope_idx).unwrap()
    }

    /// Creates a new child scope and enters it. 
    pub fn enter_scope(&mut self) -> ActiveScope {
        let idx = self.scopes.len();
        let mut sc = SymbolScope::new(idx);
        sc.depth = self.active_scope().depth + 1;
        sc.up = Some(self.active_scope_idx);
        self.scopes.push(sc);

        self.active_scope_idx = idx; 

        ActiveScope::new(self, idx)
    }

    pub fn enter_scope_unmanaged(&mut self) -> usize {
        let idx = self.scopes.len();
        let mut sc = SymbolScope::new(idx);
        sc.depth = self.active_scope().depth + 1;
        sc.up = Some(self.active_scope_idx);
        self.scopes.push(sc);
        self.active_scope_idx = idx; 
        idx
    }

    fn get_scope(&self, idx: usize) -> Option<&SymbolScope> {
        self.scopes.get(idx)
    }

    fn get_scope_mut(&mut self, idx: usize) -> Option<&mut SymbolScope> {
        self.scopes.get_mut(idx)
    }

    /// Exit the current scope and switch the the higher enclosing scope.
    /// TODO: determine whether this should do nothing if we are already in the global scope, or if we should
    /// throw an error.
    pub fn exit_scope(&mut self) {
        self.active_scope_idx = self.active_scope().up.unwrap()
    }

    /// Add a symbol to the active scope. 
    /// 
    #[must_use]
    pub fn push_symbol(&mut self, sym: Symbol) -> ParseRes<()> {
        let position = sym.position.clone();
        let result = self.active_scope_mut().symbols.insert(sym.name.clone(), sym); 
        if result.is_some() {
            Err(ParseErr::new(position.clone(), position, ParseErrMsg::Something))
        } else {
            Ok(())
        }
    }

    ///Add a symbol to the highest level (global) scope.
    pub fn push_global_symbol(&mut self, sym: Symbol) -> ParseRes<()> {
        let scope = self.global_mut();
        let position = sym.position.clone();
        
        let result = scope.symbols.insert(sym.name.clone(), sym);
        if result.is_some() {
            Err(ParseErr::new(position.clone(), position, ParseErrMsg::Something))
        } else {
            Ok(())
        }
    }

    pub fn get_symbol(&self, name: &str) -> Option<&Symbol> {
        //search lowest scope first
        //if value is not found in the lowest scope, search all enclosing scopes
        //up until the global scope.
        let mut idx = self.active_scope_idx;
        while let Some(scope) = self.get_scope(idx) {
            let opt_symbol = scope.symbols.get(name);
            if opt_symbol.is_some() {
                return opt_symbol;
            }
            if let Some(up) = scope.up {
                idx = up
            } else {
                break;
            }
        }
        //no symbol is found
        None
    }

    pub fn get_mut_symbol(&mut self, name: &str) -> Option<&mut Symbol> {
        let mut idx = self.active_scope_idx;

        
        while let Some(scope) = self.get_scope(idx) {
            if scope.symbols.contains_key(name) {
                //a double lookup is required here due to a known limitation of the borrow checker. 
                //the workaround necessitates borrowing as immutable in the loop, and then performing a mutable
                //borrow once the correct symbol has been found.
                //see https://www.reddit.com/r/learnrust/comments/rmgif7/mutable_borrow_in_while_loop/

                return self.scopes.get_mut(idx).unwrap().symbols.get_mut(name);
            }
            if let Some(up) = scope.up {
                idx = up
            } else {
                break;
            }
        }
        None
    }

    pub fn dump(&self) {
        todo!("Dump symbol table contents based on scope");
    }
}


/// Represents a scope in C, typically indicated by enclosing `{}` brackets. 
/// 
#[derive(Debug, Clone)]
pub struct SymbolScope {
    pub symbols: HashMap<String, Symbol>,
    pub up: Option<usize>,
    pub index: usize,
    pub depth: usize,
}

impl SymbolScope {
    pub fn new(index: usize) -> Self {
        Self {
            symbols: HashMap::new(),
            up: None,
            index: index,
            depth: 0
        }
    }
}


#[derive(Debug, Clone, Copy)]
pub enum ScopeKind {
    Constants,
    Labels,
    Global,
    Param,
    Local(i32) //scope depth
}


/// Convenience struct that automatically closes the active C scope once the current scope is exited.
#[derive(Debug)]
pub struct ActiveScope<'a> {
    table: &'a mut SymbolTable,
    scope_idx: usize,
}

impl<'a> ActiveScope<'a> {
    pub fn new(table: &'a mut SymbolTable, scope: usize) -> Self {
        Self { table: table, scope_idx: scope }
    }

    pub fn exit_scope(self) {
        self.table.exit_scope();
    }

    pub fn push_symbol(&mut self, sym: Symbol) -> ParseRes<()> {
        self.table.push_symbol(sym)
    }

    pub fn index(&self) -> usize {
        self.scope_idx
    }

    /// Pushes the symbol onto the scope, but accepts the symbol constructor arguments and constructs it in place
    /// rather than accepting the symbol itself. This is a convenience method to reduce boilerplate.
    pub fn emplace_symbol(&mut self, name: String, ast: ASTNode, storage: StorageClass) -> ParseRes<()> {
        self.table.push_symbol(Symbol {name: name, position: ast.pos.clone(), node: ast, sclass: storage})
    }

    pub fn push_global_symbol(&mut self, sym: Symbol) -> ParseRes<()> {
        self.table.push_global_symbol(sym)
    }

    pub fn get_symbol(&self, name: &str) -> Option<&Symbol> {
        self.table.get_symbol(name)
    }

}


impl<'a> Drop for ActiveScope<'a> {
    fn drop(&mut self) {
        self.table.exit_scope();
    }
}

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

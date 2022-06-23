use crate::types::*;
use crate::ast::*;
use crate::err::*;
use crate::utils::*;
use std::collections::HashMap;

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

    pub fn active_scope(&self) -> &SymbolScope {
        self.scopes.get(self.active_scope_idx).unwrap()
    }

    pub fn active_scope_mut(&mut self) -> &mut SymbolScope {
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

    pub fn get_scope(&self, idx: usize) -> Option<&SymbolScope> {
        self.scopes.get(idx)
    }

    pub fn get_scope_mut(&mut self, idx: usize) -> Option<&mut SymbolScope> {
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
    
    #[must_use]
    pub fn emplace_symbol(&mut self, name: String, ast: ASTNode, storage: StorageClass) -> ParseRes<()> {
        self.push_symbol(Symbol {name: name, position: ast.pos.clone(), node: ast, sclass: storage})
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

pub struct Symbols {
    pub constants: SymbolTable,
    pub globals: SymbolTable,
    pub typedefs: SymbolTable,
    pub identifiers: SymbolTable,
    pub labels: SymbolTable,
    pub externals: SymbolTable,
}

impl Symbols {
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

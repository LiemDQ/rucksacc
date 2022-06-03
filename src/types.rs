use crate::ast::ASTNode;
use crate::err::ParseRes;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sign {
    Signed,
    Unsigned,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    Void,
    Bool,
    Char(Sign),
    Short(Sign),
    Int(Sign),
    Long(Sign),
    LLong(Sign),
    Float,
    Double,
    LDouble,
    Enum(Box<TypeInfo>), //underlying type
    Ptr(Box<TypeInfo>), //base type of pointer
    Func(Box<TypeInfo>, Vec<TypeInfo>, bool), //return type, param types, is vararg?
    Array(Box<TypeInfo>, i64), //element type, size. Negative means the size is automatically determined
    VLA,
    Struct(String, Vec<RecordMember>), //name, members
    Union(String, Vec<RecordMember>, usize), //name, members, index of largest member
    Bitfield(u32, u32), //bit offset, bit width
    UNUSED,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedTypeInfo {
    pub ty: TypeInfo,
    pub is_atomic: bool,
    pub storage: StorageClass,
    pub qualifiers: Qualifiers, 
}

impl QualifiedTypeInfo {
    pub fn new() -> Self {
        Self { 
            ty: TypeInfo::new(),
            is_atomic: false, 
            storage: StorageClass::Auto, //storage class specifier is automatic by default.
            qualifiers: Qualifiers::new(), 
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeInfo {
    pub kind : TypeKind,
    pub size : i64,
    pub align : i64,
}

impl TypeInfo {
    pub fn new() -> Self {
        Self { 
            kind: TypeKind::UNUSED,
            size: 0, 
            align: 0, 
        }
    }

    pub fn make_ptr(self) -> Self {
        //TODO: handle array aliasing to pointers. 
        Self { kind: TypeKind::Ptr(Box::new(self)), size: 8, align: 8}
    }

    pub fn make_void() -> Self {
        TypeInfo::make_new_info(TypeKind::Void, 1, 1)
    }
    
    pub fn make_char() -> Self {
        TypeInfo::make_new_info(TypeKind::Char(Sign::Signed), 1, 1)
    }

    pub fn make_short() -> Self {
        TypeInfo::make_new_info(TypeKind::Short(Sign::Signed), 2, 2)
    }

    pub fn make_int() -> Self {
        TypeInfo::make_new_info(TypeKind::Int(Sign::Signed), 4, 4)
    }

    pub fn make_long() -> Self {
        TypeInfo::make_new_info(TypeKind::Long(Sign::Signed), 8, 8)
    }

    pub fn make_uchar() -> Self {
        TypeInfo::make_new_info(TypeKind::Char(Sign::Unsigned), 1, 1)
    }
    
    pub fn make_ushort() -> Self {
        TypeInfo::make_new_info(TypeKind::Short(Sign::Unsigned), 2, 2)
    }

    pub fn make_uint() -> Self {
        TypeInfo::make_new_info(TypeKind::Int(Sign::Unsigned), 4, 4)
    }

    pub fn make_ulong() -> Self {
        TypeInfo::make_new_info(TypeKind::Long(Sign::Unsigned), 8, 8)
    }

    pub fn make_bool() -> Self {
        TypeInfo::make_new_info(TypeKind::Bool, 1, 1)
    }

    pub fn make_float() -> Self {
        TypeInfo::make_new_info(TypeKind::Float, 4, 4)
    }

    pub fn make_double() -> Self {
        TypeInfo::make_new_info(TypeKind::Double, 8, 8)
    }

    pub fn make_ldouble() -> Self {
        TypeInfo::make_new_info(TypeKind::LDouble, 16, 16)
    }

    /// TODO: allow programmer defined underlying types
    pub fn make_enum() -> Self {
        TypeInfo::make_new_info(TypeKind::Enum(Box::new(TypeInfo::make_int())), 1, 1)
    }

    /// Converts the `TypeInfo` object to a function type, with a return type 
    /// the same as the original `TypeInfo` object.
    /// 
    /// NOTE: The C spec forbids sizeof(<function-type>) but GCC allows it and evaluates
    /// it to 1.
    pub fn make_func(self, params: Vec<TypeInfo>, vararg: bool) -> Self {
        Self { kind: TypeKind::Func(Box::new(self), params, vararg), size: 1, align: 1 }
    }

    ///
    pub fn make_array(self, len: i64) -> Self {
        TypeInfo::make_new_info(TypeKind::Array(Box::new(self), len), self.size * len, self.align)
    }

    pub fn make_vla_array(self, len: ASTNode) -> Self {
        todo!("Create VLA array");
    }

    pub fn make_bitfield(self, size: i64) -> ParseRes<Self> {
        todo!("Create bitfield");
    }

    pub fn get_struct_member_types<'a>(&'a self) -> Option<Vec<&'a TypeInfo>> {
        match self.kind {
            TypeKind::Struct(_, ref members) | TypeKind::Union(_, ref members, _) => {
                let types = members
                    .iter()
                    .map(|member| &member.ty)
                    .collect();
                
                    Some(types)
            }
            _ => None,
        }
    }

    pub fn get_struct_member_names<'a>(&'a self) -> Option<Vec<&'a str>> {
        match self.kind {
            TypeKind::Struct(_, ref members) | TypeKind::Union(_, ref members, _) => {
                let names = members
                    .iter()
                    .map(|member| &*member.name)
                    .collect();
                
                    Some(names)
            }
            _ => None,
        }
    }

    fn make_new_info(kind: TypeKind, size: i64, align: i64) -> Self {
        Self { kind: kind, size: size, align: align }
    }
}


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StorageClass {
    Auto,
    Register,
    Static,
    Extern,
    Typedef,
    Enum,
    UNUSED,
}

/// Type qualifiers for a variable, that aren't mutually exclusive and don't fall neatly into a specific category. 
#[derive(Debug, Clone, PartialEq)]
pub struct Qualifiers {
    pub is_restrict: bool,
    pub is_const: bool,
    pub is_constexpr: bool,
    pub is_tls: bool,
    pub is_volatile: bool,
    pub is_inline: bool,
    pub is_noreturn: bool,
}

impl Qualifiers {
    pub fn new() -> Self {
        Self {
            is_restrict: false,
            is_const: false,
            is_constexpr: false,
            is_tls: false,
            is_volatile: false,
            is_inline: false,
            is_noreturn: false,
        }
    }
}

struct ArrayType {
    arr_len : i64,
    //TODO: elements for VLAs
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecordMember {
    pub name: String,
    pub is_bitfield: bool, 
    pub ty: TypeInfo
}


/// Determine whether two types can be implicitly converted to one another.
pub fn is_compatible( a: &TypeInfo, b: &TypeInfo) -> bool {
    todo!();
}

pub fn is_integer(ty: &TypeInfo) -> bool {
    let k = &ty.kind;
    
    match k {
        TypeKind::Bool | TypeKind::Char(_) | TypeKind::Short(_) |
        TypeKind::Int(_)  |  TypeKind::Long(_) |  TypeKind::Enum(_) => true,
        _ => false,

    }
}

pub fn is_float_num(ty: &TypeInfo) -> bool {
    let k = &ty.kind;
    match k {
        TypeKind::Double | TypeKind::LDouble | TypeKind::Float => true,
        _ => false,
    }
}

pub fn is_numeric(ty: &TypeInfo) -> bool {
    is_integer(ty) || is_float_num(ty)
}

/// Finds the common
/// Section 6.3 of the ISO specification covers type conversions.
/// All builtin integer types have a "rank" that determines what integers they can be converted to.
/// 
/// `long long int` > `long int` > `int` > `short int` > `char` > `_Bool`
/// 
/// Rankings are transitive. Unsigned integers have the same rank as the corresponding
/// signed integer type. 
pub fn get_common_type(rhs: &TypeInfo, lhs: &TypeInfo) -> TypeInfo {
    todo!("Get common type")
}

/// For many binary operators, we implicitly promote operands so that
/// both operands have the same type. Any integral value smaller than int
/// is always promoted to int. If the type of one operand is larger than 
/// the other's (e.g. `long` vs `int`), the smaller operand will be promoted
/// to match with the other.
/// 
/// This is known as "usual arithmetic conversion". 
/// 
/// See ISO 9899 specification 6.3.1.8.
pub fn convert_usual_arith() {

}
#![allow(dead_code)]
mod codegen;
mod lex; 
mod utils;
mod parse;
mod types;
mod err;
mod ast;

use clap::Parser;

#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    stmt: String
}

fn main() {
    let args = Args::parse();
    let val = args.stmt.parse::<i32>().unwrap();

    println!(
        "define i32 @main() {{"
    );
    println!("%1 = alloca i32, align 4");
    println!("store i32 {}, i32* %1, align 4", val);
    println!("}}");
}

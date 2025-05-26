//! Mutator that adds new, empty functions.

use std::ptr::read;

use super::Mutator;
use crate::info::ModuleInfo;
use crate::module::{PrimitiveTypeInfo, TypeInfo};
use crate::{Result, WasmMutate};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use wasm_encoder::{AbstractHeapType, HeapType, Instruction, MemArg, Module, SectionId};
use wasmparser::{KnownCustom, Parser, Payload};

/// Mutator that adds new, empty functions to a Wasm module.
#[derive(Clone, Copy)]
pub struct ZeroCopyFunctionMutator;

fn find_target_function_index(
    wasm_bytes: &[u8],
    target_function_name: String,
) -> Result<Option<u32>> {
    let parser = Parser::new(0);
    let mut function_index = 0u32;
    let mut export_section_found = false;

    for payload in parser.parse_all(wasm_bytes) {
        match payload? {
            Payload::ExportSection(reader) => {
                export_section_found = true;
                for export in reader {
                    let export = export?;
                    println!("1. {}", export.name);
                    if export.name == target_function_name {
                        if let wasmparser::ExternalKind::Func = export.kind {
                            return Ok(Some(export.index));
                        }
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                if !export_section_found {
                    function_index += reader.count();
                }
            }
            Payload::CustomSection(reader) => {
                if let KnownCustom::Name(namesection_reader) = reader.as_known() {
                    for names in namesection_reader {
                        if let Ok(names) = names {
                            if let wasmparser::Name::Function(namemap) = names {
                                for fname in namemap {
                                    if let Ok(naming) = fname {
                                        println!("{}: {}", naming.index, naming.name);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(None)
}

fn find_target_function_index_from_custom_section(
    wasm_bytes: &[u8],
    target_function_name: String,
) -> Result<Option<u32>> {
    let parser = Parser::new(0);

    for payload in parser.parse_all(wasm_bytes) {
        match payload? {
            Payload::CustomSection(reader) => {
                if let KnownCustom::Name(namesection_reader) = reader.as_known() {
                    for names in namesection_reader {
                        if let Ok(names) = names {
                            if let wasmparser::Name::Function(namemap) = names {
                                for fname in namemap {
                                    if let Ok(naming) = fname {
                                        if target_function_name == naming.name {
                                            return Ok(Some(naming.index));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(None)
}

fn find_ty_idx(wasm_bytes: &[u8], target_func_index: u32) -> Option<u32> {
    let mut info = ModuleInfo::default();
    info.input_wasm = wasm_bytes.clone();
    let parser = Parser::new(0);

    for payload in parser.parse_all(wasm_bytes) {
        if let Ok(Payload::FunctionSection(reader)) = payload {
            info.functions = Some(info.raw_sections.len());
            info.section(SectionId::Function.into(), reader.range(), wasm_bytes);

            for ty in reader {
                info.function_map.push(ty.unwrap());
            }

            println!("--- {:?}", info.function_map);

            return Some(info.function_map[target_func_index as usize]);
        }
    }
    None
}

impl Mutator for ZeroCopyFunctionMutator {
    fn mutate<'a>(
        &self,
        config: &'a mut WasmMutate,
    ) -> Result<Box<dyn Iterator<Item = Result<Module>> + 'a>> {
        let (_target_func_idx_fr_copy, ty_idx_fr_copy) = if let Some(info) = &config.info {
            let original_wat_bytes = info.input_wasm;
            let target_func_idx = find_target_function_index_from_custom_section(
                &original_wat_bytes,
                "Fr_copy".to_string(),
            )
            .unwrap();
            let ty_idx = find_ty_idx(&original_wat_bytes, target_func_idx.unwrap());
            (target_func_idx, ty_idx)
        } else {
            (None, None)
        };

        let ty_idx = ty_idx_fr_copy.unwrap();

        // (Re)encode the function section and add this new entry.
        let mut func_sec_enc = wasm_encoder::FunctionSection::new();
        if let Some(func_sec_idx) = config.info().functions {
            let reader = config.info().get_binary_reader(func_sec_idx);
            let reader = wasmparser::FunctionSectionReader::new(reader)?;
            for x in reader {
                func_sec_enc.function(x?);
            }
        }
        func_sec_enc.function(ty_idx);

        // Copy the existing code bodies over and then add a new dummy body for
        // this function.
        let mut code_sec_enc = wasm_encoder::CodeSection::new();
        if let Some(code_sec_idx) = config.info().code {
            let reader = config.info().get_binary_reader(code_sec_idx);
            let reader = wasmparser::CodeSectionReader::new(reader)?;
            for body in reader {
                let body = body?;
                code_sec_enc.raw(body.as_bytes());
            }
        }
        let func_ty = match &config.info().types_map[usize::try_from(ty_idx).unwrap()] {
            TypeInfo::Func(func_ty) => func_ty,
        };

        println!("{:?}", func_ty.params);
        println!("{:?}", func_ty.returns);

        let mut func = wasm_encoder::Function::new(vec![]);
        // Instructions for the new function:
        // local.get $pr (param index 0)
        func.instruction(&Instruction::LocalGet(0));
        // i64.const 0
        func.instruction(&Instruction::I64Const(0));
        // i64.store (align = 3 for i64, offset = 0)
        func.instruction(&Instruction::I64Store(MemArg {
            offset: 0,
            align: 3,
            memory_index: 0,
        }));
        // local.get $pr
        func.instruction(&Instruction::LocalGet(0));
        // i64.const 0
        func.instruction(&Instruction::I64Const(0));
        // i64.store offset=8 (align = 3 for i64)
        func.instruction(&Instruction::I64Store(MemArg {
            offset: 8,
            align: 3,
            memory_index: 0,
        }));
        // local.get $pr
        func.instruction(&Instruction::LocalGet(0));
        // i64.const 0
        func.instruction(&Instruction::I64Const(0));
        // i64.store offset=16 (align = 3 for i64)
        func.instruction(&Instruction::I64Store(MemArg {
            offset: 16,
            align: 3,
            memory_index: 0,
        }));
        // local.get $pr
        func.instruction(&Instruction::LocalGet(0));
        // i64.const 0
        func.instruction(&Instruction::I64Const(0));
        // i64.store offset=24 (align = 3 for i64)
        func.instruction(&Instruction::I64Store(MemArg {
            offset: 24,
            align: 3,
            memory_index: 0,
        }));
        // local.get $pr
        func.instruction(&Instruction::LocalGet(0));
        // i64.const 0
        func.instruction(&Instruction::I64Const(0));
        // i64.store offset=32 (align = 3 for i64)
        func.instruction(&Instruction::I64Store(MemArg {
            offset: 32,
            align: 3,
            memory_index: 0,
        }));

        func.instructions().end();
        code_sec_enc.function(&func);

        let module = if config.info().functions.is_some() {
            // Replace the old sections with the new ones.
            config
                .info()
                .replace_multiple_sections(|_, sec_id, module| match sec_id {
                    x if x == wasm_encoder::SectionId::Function as u8 => {
                        module.section(&func_sec_enc);
                        true
                    }
                    x if x == wasm_encoder::SectionId::Code as u8 => {
                        module.section(&code_sec_enc);
                        true
                    }

                    _ => false,
                })
        } else {
            // Insert the new sections in their respective places.
            let mut added_func = false;
            let mut added_code = false;
            let mut module = config
                .info()
                .replace_multiple_sections(|_, sec_id, module| {
                    if !added_func && sec_id >= wasm_encoder::SectionId::Function as u8 {
                        module.section(&func_sec_enc);
                        added_func = true;
                    }

                    if !added_code
                        && sec_id >= wasm_encoder::SectionId::Code as u8
                        && sec_id != wasm_encoder::SectionId::DataCount as u8
                    {
                        module.section(&code_sec_enc);
                        added_code = true;
                    }

                    sec_id == wasm_encoder::SectionId::Function as u8
                        || sec_id == wasm_encoder::SectionId::Code as u8
                });
            if !added_func {
                module.section(&func_sec_enc);
            }
            if !added_code {
                module.section(&code_sec_enc);
            }
            module
        };

        Ok(Box::new(std::iter::once(Ok(module))))
    }

    fn can_mutate<'a>(&self, config: &'a WasmMutate) -> bool {
        // Note: adding a new, never-called function preserves semantics so we
        // don't need to gate on whether `config.preserve_semantics` is set or
        // not.
        !config.reduce && config.info().num_types() > 0
    }
}

/// u32 を LEB128 エンコードするヘルパー
fn leb128_encode(mut value: u32) -> Vec<u8> {
    let mut buf = Vec::new();
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            buf.push(byte | 0x80);
        } else {
            buf.push(byte);
            break;
        }
    }
    buf
}

/// RAW バイト列中のすべての `call(old_id)` を探し、その中から
/// ランダムに１箇所だけを `call(new_id)` に置き換える。
///
/// - bytes: 元の WASM バイト列
/// - old_id: 置き換えたい関数番号
/// - new_id: 新しい関数番号
/// - seed: RNG のシード値
pub fn patch_calls_raw_random(
    bytes: &[u8],
    old_id: u32,
    new_id: u32,
    seed: u64,
) -> Result<Vec<u8>> {
    let mut out = bytes.to_vec();
    let old_leb = leb128_encode(old_id);
    let new_leb = leb128_encode(new_id);

    // すべての候補オフセットを収集
    let opcode_call = 0x10u8;
    let mut candidates = Vec::new();
    let len = out.len();
    let pattern_len = 1 + old_leb.len();

    let mut i = 0;
    while i + pattern_len <= len {
        if out[i] == opcode_call && out[i + 1..i + 1 + old_leb.len()] == old_leb[..] {
            candidates.push(i);
            i += pattern_len;
        } else {
            i += 1;
        }
    }

    // 候補がない場合はそのまま返却
    if candidates.is_empty() {
        return Ok(out);
    }

    // シード指定で RNG を初期化し、ランダムに１つ選択
    let mut rng = StdRng::seed_from_u64(seed);
    let idx = rng.gen_range(0..candidates.len());
    let pos = candidates[idx];

    // その１箇所だけを置き換え
    out[pos + 1..pos + 1 + old_leb.len()].copy_from_slice(&new_leb);

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutators::match_mutation;
    use wasm_encoder::{EntityType, ValType};

    #[test]
    fn test_patch_calls_raw_random() {
        // simple モジュール: func0=$mul, func1=$add, func2=$use_mul が２回呼び出す
        let wat = r#"
            (module
                (func $mul (export "mul") (param i32 i32) (result i32)
                    local.get 0 
                    local.get 1 
                    i32.mul
                )
                (func $use (export "use") (param i32 i32) (result i32)
                    local.get 0 
                    local.get 1 
                    call $mul
                    local.get 0 
                    local.get 1 
                    call $mul
                )
            )
        "#;
        let wasm = wat::parse_str(wat).unwrap();
        let patched = patch_calls_raw_random(&wasm, 0, 1, 42).unwrap();
        let text = wasmprinter::print_bytes(patched).unwrap();
        println!("{}", text);
        assert!(false);
    }

    #[test]
    fn test_add_specific_function() {
        // Initial WAT: a module with a memory and a type (i32, i32) -> ()
        // We'll ensure type 0 has the signature (param i32 i32) (result)
        // and a memory exists.
        println!("222222");
        let original_wat = r#"
            (module
                (func $mul (export "mul") (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.mul
                )
                (func $add (export "add") (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.add
                )
                (func $Fr_copy (param $pr i32) (param $px i32)
                    local.get $pr
                    local.get $px
                    i64.load
                    i64.store
                    local.get $pr
                    local.get $px
                    i64.load offset=8
                    i64.store offset=8
                    local.get $pr
                    local.get $px
                    i64.load offset=16
                    i64.store offset=16
                    local.get $pr
                    local.get $px
                    i64.load offset=24
                    i64.store offset=24
                    local.get $pr
                    local.get $px
                    i64.load offset=32
                    i64.store offset=32
                )
                (memory (;0;) 11)
                (export "memory" (memory 0))
            )
        "#;
        let expected_wat_regex = r#"
            (module
                (type (;0;) (func (param i32 i32) (result i32)))
                (type (;1;) (func (param i32 i32)))
                (memory (;0;) 11)
                (export "mul" (func $mul))
                (export "add" (func $add))
                (export "memory" (memory 0))
                (func $mul (;0;) (type 0) (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.mul
                )
                (func $add (;1;) (type 0) (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.add
                )
                (func $Fr_copy (;2;) (type 1) (param $pr i32) (param $px i32)
                    local.get $pr
                    local.get $px
                    i64.load
                    i64.store
                    local.get $pr
                    local.get $px
                    i64.load offset=8
                    i64.store offset=8
                    local.get $pr
                    local.get $px
                    i64.load offset=16
                    i64.store offset=16
                    local.get $pr
                    local.get $px
                    i64.load offset=24
                    i64.store offset=24
                    local.get $pr
                    local.get $px
                    i64.load offset=32
                    i64.store offset=32
                )
                (func (;3;) (type 1) (param i32 i32)
                    local.get 0
                    i64.const 0
                    i64.store
                    local.get 0
                    i64.const 0
                    i64.store offset=8
                    local.get 0
                    i64.const 0
                    i64.store offset=16
                    local.get 0
                    i64.const 0
                    i64.store offset=24
                    local.get 0
                    i64.const 0
                    i64.store offset=32
                )
            )
        "#;

        //let original_wat_bytes = wat::parse_str(original_wat).unwrap();

        /*
        //let target_func_idx =
        //    find_target_function_index(&original_wat_bytes, "Fr_copy".to_string()).unwrap();
        let target_func_idx = find_target_function_index_from_custom_section(
            &original_wat_bytes,
            "Fr_copy".to_string(),
        )
        .unwrap();
        let ty_idx = find_ty_idx(&original_wat_bytes, target_func_idx.unwrap());

        println!("5555");
        */

        println!("333333");
        match_mutation(original_wat, ZeroCopyFunctionMutator, expected_wat_regex);
        println!("444444");

        //assert!(false);
        println!("444444");
        assert!(false);
    }
}

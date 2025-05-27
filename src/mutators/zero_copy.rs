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

pub fn find_target_function_index_from_custom_section(
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

pub fn find_ty_idx(wasm_bytes: &[u8], target_func_index: u32) -> Option<u32> {
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

            return Some(info.function_map[target_func_index as usize]);
        }
    }
    None
}

pub fn find_function_body_range(
    wasm_bytes: &[u8],
    target_func_index: u32,
) -> Option<std::ops::Range<usize>> {
    let parser = Parser::new(0);
    let mut current_func_index = 0u32;

    for payload in parser.parse_all(wasm_bytes) {
        if let Ok(Payload::CodeSectionEntry(reader)) = payload {
            println!("c: {}", current_func_index);
            if current_func_index == target_func_index {
                return Some(reader.range().start..reader.range().end);
            }
            current_func_index += 1;
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

        // (Re)encode the name section
        let mut name_sec_enc = wasm_encoder::NameSection::new();
        let mut function_names = wasm_encoder::NameMap::new();

        let original_wat_bytes = config.info.clone().unwrap().input_wasm;
        let parser = Parser::new(0);
        let mut max_idx = 0;
        for payload in parser.parse_all(original_wat_bytes) {
            match payload? {
                Payload::CustomSection(reader) => {
                    if let KnownCustom::Name(namesection_reader) = reader.as_known() {
                        for names in namesection_reader {
                            if let Ok(names) = names {
                                if let wasmparser::Name::Function(namemap) = names {
                                    for fname in namemap {
                                        if let Ok(naming) = fname {
                                            max_idx = std::cmp::max(max_idx, naming.index);
                                            function_names.append(naming.index, naming.name);
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
        function_names.append(max_idx + 1, "Fr_copy_zero");
        name_sec_enc.functions(&function_names);

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
                    x if x == wasm_encoder::SectionId::Custom as u8 => {
                        module.section(&name_sec_enc);
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

pub fn patch_calls_raw_random(
    bytes: &[u8],
    old_id: u32,
    new_id: u32,
    start_pos: usize,
    end_pos: usize,
    seed: u64,
) -> Result<Vec<u8>> {
    let mut out = bytes.to_vec();
    let old_leb = leb128_encode(old_id);
    let new_leb = leb128_encode(new_id);

    let opcode_call = 0x10u8;
    let mut candidates = Vec::new();
    let pattern_len = 1 + old_leb.len();

    let mut i = start_pos;
    while i + pattern_len <= end_pos {
        if out[i] == opcode_call && out[i + 1..i + 1 + old_leb.len()] == old_leb[..] {
            candidates.push(i);
            i += pattern_len;
        } else {
            i += 1;
        }
    }

    if candidates.is_empty() {
        return Ok(out);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let idx = rng.gen_range(0..candidates.len());
    let pos = candidates[idx];

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
        let wat = r#"
            (module
                (func $add (export "add") (param i32 i32) (result i32)
                    local.get 0 
                    local.get 1 
                    i32.add
                )
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
        let expected_wat = r#"
            (module
                (func $add (export "add") (param i32 i32) (result i32)
                    local.get 0 
                    local.get 1 
                    i32.add
                )
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
                    call $add
                )
            )
        "#;

        let wasm = wat::parse_str(wat).unwrap();
        let expected_wasm = wat::parse_str(expected_wat).unwrap();

        let func_idx =
            find_target_function_index_from_custom_section(&wasm, "use".to_string()).unwrap();
        let func_range = find_function_body_range(&wasm, func_idx.unwrap()).unwrap();
        let patched =
            patch_calls_raw_random(&wasm, 1, 0, func_range.start, func_range.end, 42).unwrap();

        let expected_text = wasmprinter::print_bytes(expected_wasm).unwrap();
        let patched_text = wasmprinter::print_bytes(patched).unwrap();

        assert_eq!(expected_text.trim(), patched_text.trim());
    }

    #[test]
    fn test_add_specific_function() {
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
                (func $Fr_copy (;2;) (type 1) (param i32 i32)
                    local.get 0
                    local.get 1
                    i64.load
                    i64.store
                    local.get 0
                    local.get 1
                    i64.load offset=8
                    i64.store offset=8
                    local.get 0
                    local.get 1
                    i64.load offset=16
                    i64.store offset=16
                    local.get 0
                    local.get 1
                    i64.load offset=24
                    i64.store offset=24
                    local.get 0
                    local.get 1
                    i64.load offset=32
                    i64.store offset=32
                )
                (func $Fr_copy_zero (;3;) (type 1) (param i32 i32)
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

        match_mutation(original_wat, ZeroCopyFunctionMutator, expected_wat_regex);
    }
}

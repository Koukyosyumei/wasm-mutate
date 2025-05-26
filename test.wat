(module
  (type (;0;) (func (param i32 i32) (result i32)))
  (type (;1;) (func (param i32 i32)))
  (memory (;0;) 11)
  (export "mul" (func $mul))
  (export "mul" (func $add))
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

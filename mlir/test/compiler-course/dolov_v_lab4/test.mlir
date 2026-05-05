// RUN: mlir-opt -load-pass-plugin="%mlir_lib_dir/dolov_v_lab4_MLIR%shlibext" --pass-pipeline="builtin.module(dolov-loop-fusion)" %s | FileCheck %s

func.func @test_primary_fusion(%arg0: memref<2048xf32>, %arg1: memref<2048xf32>) {
  %c32 = arith.constant 32 : index
  %c1024 = arith.constant 1024 : index
  %c16 = arith.constant 16 : index

  // CHECK: scf.for %[[IV:.*]] = %c32 to %c1024 step %c16
  // CHECK-NEXT: memref.load %arg0[%[[IV]]]
  // CHECK-NEXT: memref.store %{{.*}}, %arg1[%[[IV]]]
  // CHECK-NEXT: memref.load %arg1[%[[IV]]]
  // CHECK-NEXT: arith.mulf
  scf.for %i = %c32 to %c1024 step %c16 {
    %0 = memref.load %arg0[%i] : memref<2048xf32>
    memref.store %0, %arg1[%i] : memref<2048xf32>
  }
  scf.for %j = %c32 to %c1024 step %c16 {
    %1 = memref.load %arg1[%j] : memref<2048xf32>
    %2 = arith.mulf %1, %1 : f32
    memref.store %2, %arg0[%j] : memref<2048xf32>
  }
  return
}

func.func @test_triple_sequential_fusion(%arg0: memref<1000xi32>) {
  %c3 = arith.constant 3 : index
  %c333 = arith.constant 333 : index
  %c6 = arith.constant 6 : index
  %v = arith.constant 42 : i32

  // CHECK: scf.for
  // CHECK-NOT: scf.for
  scf.for %i = %c3 to %c333 step %c6 {
    memref.store %v, %arg0[%i] : memref<1000xi32>
  }
  scf.for %j = %c3 to %c333 step %c6 {
    memref.store %v, %arg0[%j] : memref<1000xi32>
  }
  scf.for %k = %c3 to %c333 step %c6 {
    memref.store %v, %arg0[%k] : memref<1000xi32>
  }
  return
}

func.func @test_nested_fusion(%arg0: memref<20x20xf32>) {
  %c0 = arith.constant 0 : index
  %c20 = arith.constant 20 : index
  %c1 = arith.constant 1 : index
  %f1 = arith.constant 1.0 : f32

  // CHECK: scf.for %[[I:.*]] = %c0 to %c20 step %c1
  // CHECK:   scf.for %[[J:.*]] = %c0 to %c20 step %c1
  // CHECK-NEXT: memref.store
  // CHECK-NEXT: memref.load
  scf.for %i = %c0 to %c20 step %c1 {
    scf.for %j1 = %c0 to %c20 step %c1 {
      memref.store %f1, %arg0[%i, %j1] : memref<20x20xf32>
    }
    scf.for %j2 = %c0 to %c20 step %c1 {
      %v = memref.load %arg0[%i, %j2] : memref<20x20xf32>
    }
  }
  return
}

func.func @test_complex_iter_args(%arg0: f32, %arg1: f32) -> (f32, f32) {
  %c11 = arith.constant 11 : index
  %c77 = arith.constant 77 : index
  %c3 = arith.constant 3 : index

  // CHECK: %[[R:.*]]:2 = scf.for %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[A1:.*]] = %arg0, %[[A2:.*]] = %arg1)
  // CHECK:   arith.addf %[[A1]]
  // CHECK:   arith.subf %[[A2]]
  // CHECK:   scf.yield %{{.*}}, %{{.*}}
  %0 = scf.for %i = %c11 to %c77 step %c3 iter_args(%acc1 = %arg0) -> f32 {
    %t1 = arith.addf %acc1, %acc1 : f32
    scf.yield %t1 : f32
  }
  %1 = scf.for %j = %c11 to %c77 step %c3 iter_args(%acc2 = %arg1) -> f32 {
    %t2 = arith.subf %acc2, %acc2 : f32
    scf.yield %t2 : f32
  }
  return %0, %1 : f32, f32
}

func.func @test_negative_step_mismatch(%arg0: memref<200xf32>) {
  %c0 = arith.constant 0 : index
  %c200 = arith.constant 200 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  // CHECK: scf.for
  // CHECK: scf.for
  scf.for %i = %c0 to %c200 step %c2 {
    memref.load %arg0[%i] : memref<200xf32>
  }
  scf.for %j = %c0 to %c200 step %c4 {
    memref.load %arg0[%j] : memref<200xf32>
  }
  return
}

func.func @test_negative_data_dependency(%arg0: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c80 = arith.constant 80 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[RES1:.*]] = scf.for
  // CHECK: scf.for %{{.*}} iter_args(%{{.*}} = %[[RES1]])
  %0 = scf.for %i = %c0 to %c80 step %c1 iter_args(%acc1 = %arg0) -> f32 {
    %v1 = arith.addf %acc1, %acc1 : f32
    scf.yield %v1 : f32
  }
  %1 = scf.for %j = %c0 to %c80 step %c1 iter_args(%acc2 = %0) -> f32 {
    %v2 = arith.mulf %acc2, %acc2 : f32
    scf.yield %v2 : f32
  }
  return %1 : f32
}

func.func @test_dynamic_fusion(%begin: index, %end: index, %stride: index, %buf: memref<?xf32>) {
  // CHECK: scf.for %[[IDX:.*]] = %arg0 to %arg1 step %arg2
  // CHECK-NEXT: memref.load %arg3[%[[IDX]]]
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: memref.load %arg3[%[[IDX]]]
  scf.for %i = %begin to %end step %stride {
    %v1 = memref.load %buf[%i] : memref<?xf32>
    %v2 = arith.addf %v1, %v1 : f32
  }
  scf.for %j = %begin to %end step %stride {
    %v3 = memref.load %buf[%j] : memref<?xf32>
  }
  return
}
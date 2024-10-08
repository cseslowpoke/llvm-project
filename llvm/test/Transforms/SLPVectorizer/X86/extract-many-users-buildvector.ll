; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 2
; RUN: opt -S -passes=slp-vectorizer -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

define i1 @test(float %0, double %1) {
; CHECK-LABEL: define i1 @test
; CHECK-SAME: (float [[TMP0:%.*]], double [[TMP1:%.*]]) {
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float poison>, float [[TMP0]], i32 3
; CHECK-NEXT:    [[TMP4:%.*]] = fpext <4 x float> [[TMP3]] to <4 x double>
; CHECK-NEXT:    [[TMP5:%.*]] = insertelement <2 x double> <double poison, double 0.000000e+00>, double [[TMP1]], i32 0
; CHECK-NEXT:    [[TMP6:%.*]] = fmul <2 x double> zeroinitializer, [[TMP5]]
; CHECK-NEXT:    [[TMP7:%.*]] = shufflevector <2 x double> [[TMP5]], <2 x double> [[TMP6]], <4 x i32> <i32 poison, i32 0, i32 3, i32 3>
; CHECK-NEXT:    [[TMP8:%.*]] = shufflevector <4 x double> [[TMP7]], <4 x double> <double 0.000000e+00, double poison, double poison, double poison>, <4 x i32> <i32 4, i32 1, i32 2, i32 3>
; CHECK-NEXT:    [[TMP9:%.*]] = shufflevector <4 x double> [[TMP4]], <4 x double> <double poison, double poison, double poison, double 0.000000e+00>, <4 x i32> <i32 2, i32 0, i32 1, i32 7>
; CHECK-NEXT:    [[TMP10:%.*]] = fmul <4 x double> [[TMP8]], [[TMP9]]
; CHECK-NEXT:    [[TMP11:%.*]] = fmul <4 x double> zeroinitializer, [[TMP4]]
; CHECK-NEXT:    [[TMP12:%.*]] = call <8 x double> @llvm.vector.insert.v8f64.v4f64(<8 x double> <double poison, double poison, double poison, double poison, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00>, <4 x double> [[TMP10]], i64 0)
; CHECK-NEXT:    [[TMP13:%.*]] = call <8 x double> @llvm.vector.insert.v8f64.v4f64(<8 x double> <double poison, double poison, double poison, double poison, double poison, double poison, double 0.000000e+00, double 0.000000e+00>, <4 x double> [[TMP11]], i64 0)
; CHECK-NEXT:    [[TMP14:%.*]] = call <8 x double> @llvm.vector.insert.v8f64.v2f64(<8 x double> [[TMP13]], <2 x double> [[TMP6]], i64 4)
; CHECK-NEXT:    [[TMP15:%.*]] = fsub <8 x double> [[TMP12]], [[TMP14]]
; CHECK-NEXT:    [[TMP16:%.*]] = fmul <8 x double> [[TMP12]], [[TMP14]]
; CHECK-NEXT:    [[TMP17:%.*]] = shufflevector <8 x double> [[TMP15]], <8 x double> [[TMP16]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 13, i32 14, i32 15>
; CHECK-NEXT:    [[TMP18:%.*]] = fptrunc <8 x double> [[TMP17]] to <8 x float>
; CHECK-NEXT:    [[TMP19:%.*]] = fmul <8 x float> [[TMP18]], zeroinitializer
; CHECK-NEXT:    [[TMP20:%.*]] = fcmp oeq <8 x float> [[TMP19]], zeroinitializer
; CHECK-NEXT:    [[TMP21:%.*]] = freeze <8 x i1> [[TMP20]]
; CHECK-NEXT:    [[TMP22:%.*]] = call i1 @llvm.vector.reduce.and.v8i1(<8 x i1> [[TMP21]])
; CHECK-NEXT:    ret i1 [[TMP22]]
;
  %3 = fpext float %0 to double
  %4 = fpext float 0.000000e+00 to double
  %5 = fpext float 0.000000e+00 to double
  %6 = fpext float 0.000000e+00 to double
  %7 = fmul double 0.000000e+00, 0.000000e+00
  %8 = fmul double 0.000000e+00, %1
  %9 = fmul double 0.000000e+00, 0.000000e+00
  %10 = fmul double 0.000000e+00, %5
  %11 = fmul double 0.000000e+00, %6
  %12 = fsub double %10, %11
  %13 = fptrunc double %12 to float
  %14 = fmul double %9, 0.000000e+00
  %15 = fmul double 0.000000e+00, %3
  %16 = fsub double %14, %15
  %17 = fptrunc double %16 to float
  %18 = fptrunc double %7 to float
  %19 = fmul double %1, %6
  %20 = fmul double 0.000000e+00, %4
  %21 = fsub double %19, %20
  %22 = fptrunc double %21 to float
  %23 = fsub double 0.000000e+00, %8
  %24 = fptrunc double %23 to float
  %25 = fmul double 0.000000e+00, 0.000000e+00
  %26 = fptrunc double %25 to float
  %27 = fmul double %9, %4
  %28 = fmul double 0.000000e+00, %5
  %29 = fsub double %27, %28
  %30 = fptrunc double %29 to float
  %31 = fmul double %9, 0.000000e+00
  %32 = fptrunc double %31 to float
  %33 = fmul float %13, 0.000000e+00
  %34 = fcmp oeq float %33, 0.000000e+00
  %35 = fmul float %22, 0.000000e+00
  %36 = fcmp oeq float %35, 0.000000e+00
  %37 = select i1 %34, i1 %36, i1 false
  %38 = fmul float %30, 0.000000e+00
  %39 = fcmp oeq float %38, 0.000000e+00
  %40 = select i1 %37, i1 %39, i1 false
  %41 = fmul float %17, 0.000000e+00
  %42 = fcmp oeq float %41, 0.000000e+00
  %43 = select i1 %40, i1 %42, i1 false
  %44 = fmul float %24, 0.000000e+00
  %45 = fcmp oeq float %44, 0.000000e+00
  %46 = select i1 %43, i1 %45, i1 false
  %47 = fmul float %32, 0.000000e+00
  %48 = fcmp oeq float %47, 0.000000e+00
  %49 = select i1 %46, i1 %48, i1 false
  %50 = fmul float %18, 0.000000e+00
  %51 = fcmp oeq float %50, 0.000000e+00
  %52 = select i1 %49, i1 %51, i1 false
  %53 = fmul float %26, 0.000000e+00
  %54 = fcmp oeq float %53, 0.000000e+00
  %55 = select i1 %52, i1 %54, i1 false
  ret i1 %55
}

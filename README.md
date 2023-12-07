# CompNMF

[![Build Status](https://github.com/kdw503/CompNMF.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/kdw503/CompNMF.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Compressed NMF

[Mariano Tepper and Guillermo Sapiro. "Compressed nonnegative matrix factorization
is fast and accurate." IEEE Transactions on Signal Processing, 64(9):2269–2283, May
2016.]

# Usage example

```jl
#===== Generate test image (if you already have data, you don't need this step) ======#
using LinearAlgebra

noc = 3
Q = qr(randn(8, 8))
Q = Q.Q
Q = Q[:,1:noc]
D = Diagonal([10, 1, 0.1])
X = Q*D*Q'

#==== Perform CompNMF  ============================================#
using NMF, CompNMF

Wcn, Hcn = NMF.nndsvd(X, noc, variant=:ar);
result = CompNMF.solve!(CompNMF.CompressedNMF{Float64}(maxiter=100, tol=1e-7), X, Wcn, Hcn)
```

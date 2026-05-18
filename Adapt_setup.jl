using Adapt
using SciMLOperators
using QuantumToolbox
using CUDA
using CUDA.CUSPARSE

# Adapt.adapt_structure(::Type{T}, x::AbstractMatrix) where T = T(x)
Adapt.adapt_storage(::Type{<:CuSparseMatrixCSC}, xs::AbstractSparseMatrix) = CuSparseMatrixCSC(xs)
Adapt.adapt_storage(::Type{<:CuSparseMatrixCSR}, xs::AbstractSparseMatrix) = CuSparseMatrixCSR(xs)
Adapt.adapt_storage(::Type{<:CuSparseMatrixCOO}, xs::AbstractSparseMatrix) = CuSparseMatrixCOO(xs)
Adapt.adapt_structure(to, x::QuantumObject) = QuantumObject(Adapt.adapt_structure(to, x.data), x.type, x.dimensions)
Adapt.adapt_structure(to, x::QuantumObjectEvolution) = QuantumObjectEvolution(Adapt.adapt_structure(to, x.data), x.type, x.dimensions)
Adapt.adapt_structure(to, x::SciMLOperators.AddedOperator) = SciMLOperators.AddedOperator(Adapt.adapt_structure(to, x.ops))
Adapt.adapt_structure(to, x::SciMLOperators.MatrixOperator) = SciMLOperators.MatrixOperator(to(x.A))
Adapt.adapt_structure(to, x::QuantumToolbox.SpostSuperOperator) = QuantumToolbox.SpostSuperOperator(to(x.R))
Adapt.adapt_structure(to, x::QuantumToolbox.SprePostSuperOperator) = QuantumToolbox.SprePostSuperOperator(to(x.L), to(x.R))

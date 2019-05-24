@inline unpack_args(i, args::Tuple) = (unpack(args[1], i), unpack_args(i, Base.tail(args))...)
unpack_args(i, args::Tuple{Any}) = (unpack(args[1], i),)
unpack_args(::Any, args::Tuple{}) = ()

## utils
common_number(a, b) =
    a == 0 ? b :
    (b == 0 ? a :
     (a == b ? a :
      throw(DimensionMismatch("number of partitions must be equal"))))

npartitions(A, Bs...) = common_number(npartitions(A), _npartitions(Bs))
npartitions(A) = 0
npartitions(bc::Broadcast.Broadcasted) = _npartitions(bc.args)

@inline _npartitions(args::Tuple) = common_number(npartitions(args[1]), _npartitions(Base.tail(args)))
_npartitions(args::Tuple{Any}) = npartitions(args[1])
_npartitions(args::Tuple{}) = 0

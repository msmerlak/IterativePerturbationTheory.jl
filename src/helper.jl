
function had!(A::AbstractMatrix, B::AbstractMatrix)
  m,n = size(A)
  @assert (m,n) == size(B)
  for j in 1:n
     for i in 1:m
       @inbounds A[i,j] *= B[i,j]
     end
  end
  return A
end

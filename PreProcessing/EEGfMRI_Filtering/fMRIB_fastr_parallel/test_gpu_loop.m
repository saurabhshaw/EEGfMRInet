function A_out = test_gpu_loop(A_in)

count = 1;
A_out = gpuArray.zeros(1,size(A_in,2));
while (count <= size(A_in,2))
    A_out(count) = A_in(count)/mean(A_in);
    count = count + 1;
end

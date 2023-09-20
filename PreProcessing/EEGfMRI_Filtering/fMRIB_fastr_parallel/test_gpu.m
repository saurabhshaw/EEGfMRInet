A = gpuArray.rand(64,250000);

A_out = arrayfun( @test_gpu_loop, ...
                  A );

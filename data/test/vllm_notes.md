# vLLM Notes

## Overview

vLLM is a high-throughput inference and serving engine for large language models. It focuses on efficient memory management and continuous batching so that many requests can share the same GPU without wasting memory. The project started at UC Berkeley and is now widely used in production.

## PagedAttention

PagedAttention is the core idea behind vLLM. Instead of storing the key-value cache of each sequence in one contiguous block of GPU memory, it splits the cache into fixed-size blocks called pages. A block table maps each logical block of a sequence to a physical block in memory. This removes most external fragmentation and allows blocks to be shared between sequences, for example when several outputs are sampled from the same prompt.

## Continuous Batching

Traditional static batching waits for every request in a batch to finish before starting new ones. Continuous batching schedules work at the iteration level: when a sequence finishes, its slot is immediately given to a waiting request. This keeps the GPU busy and greatly improves throughput, especially when output lengths vary a lot.

## KV Cache Management

The KV cache stores the attention keys and values of already processed tokens so they do not need to be recomputed at every decoding step. Its size grows linearly with sequence length and batch size, which makes it the main memory bottleneck during inference. The block manager allocates, frees and swaps cache blocks. When memory runs out, the scheduler can preempt a sequence by swapping its blocks to CPU memory or by recomputing them later.

## Prefix Caching

Automatic prefix caching reuses KV cache blocks across requests that share the same prompt prefix, such as a long system prompt or a few-shot template. Each block is hashed based on its tokens and the tokens before it. If a new request produces a matching hash, the cached block is reused and the prefill computation for that part is skipped.

## Scheduler

The scheduler decides which requests run in each step. It keeps three queues: waiting, running and swapped. At every iteration it tries to admit waiting requests while respecting limits such as max_num_seqs and max_num_batched_tokens. Chunked prefill can split a long prompt into smaller pieces so that decoding requests are not stalled by a single huge prefill.

## Quantization

vLLM supports several quantization methods to reduce memory use and speed up inference, including GPTQ, AWQ, FP8 and INT8. Quantized weights take less space, which leaves more room for the KV cache and therefore allows larger batches. The trade-off is a small potential loss in model quality, which depends on the method and the model.

## Tensor Parallelism

For models too large for a single GPU, vLLM can shard the weights across several GPUs using tensor parallelism. Each layer is split so that every GPU computes a part of the matrix multiplications, and results are combined with all-reduce operations. The option tensor_parallel_size controls how many GPUs are used. Pipeline parallelism is also available for multi-node setups.

## OpenAI-Compatible Server

vLLM ships with a server that implements the OpenAI API, including the completions and chat completions endpoints. You can start it with the vllm serve command and point existing OpenAI clients to it by changing the base URL. Streaming responses, logprobs and sampling parameters such as temperature, top_p and max_tokens are supported.

## Speculative Decoding

Speculative decoding uses a small draft model to propose several tokens at once, which the large target model then verifies in a single forward pass. When the draft is accurate, multiple tokens are accepted per step, reducing latency without changing the output distribution.

## Installation

Install vLLM with pip install vllm in a Python environment that has a compatible CUDA version. Building from source is possible for custom kernels or unreleased features. Make sure the PyTorch version matches the one vLLM was built against to avoid import errors.

## Common Errors

An out of memory error usually means gpu_memory_utilization is too high or max_model_len is too large for the available GPU. Lowering max_model_len, using a quantized model or enabling tensor parallelism are the usual fixes. A CUDA version mismatch often shows up as an undefined symbol error at import time.

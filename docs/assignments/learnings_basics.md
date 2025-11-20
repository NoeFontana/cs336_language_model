# Learnings

My learnings and answers from the assignment.

Note: Timings are done on a laptop. While the statistics are gathered over 5+ runs, this remains non-reliable due to hardware limitations.

## Personal Learnings

### Optimizing BPE

#### Pre-Tokenization

Already "TinyStories Validation set", the most performance could be gained through speeding-up tokenization.
As suggested in the assignment, the optimizations was conducted via chunking the file and distributed the chunks over multiple processes.
A benchmark was ran with "TinyStories Validation set" with different number of chunks/process. The number of chunks/processes, **N**, is denoted by [**N**]

| Name (Chunks)                                | Min (s) | Max (s) | Mean (s) | StdDev (s) | Median (s) | OPS    |
| -------------------------------------------- | ------- | ------- | -------- | ---------- | ---------- | ------ |
| `test_chunked_pretokenization_benchmark[10]` | 1.1641  | 1.2785  | 1.2177   | 0.0407     | 1.2169     | 0.8213 |
| `test_chunked_pretokenization_benchmark[6]`  | 1.2424  | 1.6195  | 1.4866   | 0.1679     | 1.5820     | 0.6727 |
| `test_chunked_pretokenization_benchmark[2]`  | 2.3072  | 2.4787  | 2.4158   | 0.0658     | 2.4415     | 0.4139 |
| `test_chunked_pretokenization_benchmark[1]`  | 4.0109  | 4.2727  | 4.1388   | 0.0944     | 4.1427     | 0.2416 |

To speed things further, a BYTE_CACHE mapping integer indices from [0, 255] to the corresponding bytestring is added. For a single threaded implementation, this leads to about 20-25% speedup.

| Name (Chunks)                                | Min (s) | Max (s) | Mean (s) | StdDev (s) | Median (s) | OPS    |
| -------------------------------------------- | ------- | ------- | -------- | ---------- | ---------- | ------ |
| `test_chunked_pretokenization_benchmark[6]`  | 1.0108  | 1.1132  | 1.0657   | 0.0373     | 1.0721     | 0.9384 |
| `test_chunked_pretokenization_benchmark[10]` | 1.1152  | 1.1915  | 1.1517   | 0.0336     | 1.1555     | 0.8683 |
| `test_chunked_pretokenization_benchmark[2]`  | 1.8768  | 2.3527  | 2.1129   | 0.1736     | 2.0872     | 0.4733 |
| `test_chunked_pretokenization_benchmark[1]`  | 3.1773  | 3.2989  | 3.2483   | 0.0509     | 3.2655     | 0.3078 |

Interestingly, with this additional caching, 6 chunks provide slightly better performance than 10 chunks. This may be due to interference of other processes.

Running on the larger TinyStories train set, using 10 chunks and processes is about 15% faster so we'll use 10 chunks for the rest of the assignment.
With these optimizations, pretokenization takes about 92s.

With further micro-optimization, namely reducing chunked results asynchronously with Counter, we can cut tokenization time by almost 2 for small datasets:

| Name (Chunks)                                                                   | Min (s) | Max (s) | Mean (s) | StdDev (s) | Median (s) | OPS    |
| ------------------------------------------------------------------------------- | ------- | ------- | -------- | ---------- | ---------- | ------ |
| `test_chunked_pretokenization_benchmark[10-~/.../TinyStoriesV2-GPT4-valid.txt]` | 0.5710  | 0.6019  | 0.5857   | 0.0114     | 0.5872     | 1.7074 |

This doesn't help for larger datasets.

Fixing the core implementation to operate on single bytes instead of utf-8 multi-bytes drastically speeds up the whole ordeal.

| Name (Chunks)                                                                   | Min (s) | Max (s) | Mean (s) | StdDev (s) | Median (s) | OPS    |
| ------------------------------------------------------------------------------- | ------- | ------- | -------- | ---------- | ---------- | ------ |
| `test_chunked_pretokenization_benchmark[10-~/.../TinyStoriesV2-GPT4-valid.txt]` | 0.2698  | 0.4124  | 0.3128   | 0.5936     | 0.2860     | 3.1967 |

#### Merging

After optimizing tokenization, we take a look at merging.
We start with the naive implementation.
I tried to optimize in Python, over TinyStoriesV2-GPT4-valid.txt, with no success. My Python implementations of more advanced strategies with incremental updates proved to have worse performance.
I then moved to Rust. The direct transcription of the naive Python implementation had similar performance. Removing unrequired byte copies, a 41% speedup was achieved. It's now time to go into the incremental updates.

In the table, timings are conducted on /home/noe/datasets/cs336/ with vocab size 10000

| Implementation                                      | Time  |
| --------------------------------------------------- | ----- |
| Naive Python (as per assignment)                    | 1m37s |
| Direct Python-to-Rust transcription (via Gemini)    | 1m30s |
| Rust with reference-based pair counting (`PairRef`) | 0m57s |
| Rust with caching + incremental updates             | 0m40s |
| Bunch of micro-optimizations                        | 0m23s |

#### Final Optimizations

With all that, we get to 6min on TinyStoriesV2-GPT4-train.txt.
66% of the time is being spent on pretokenization.

The bottleneck remains the tokenization implementation.

Optimizations applied from there:

- Matching the regex on bytes
- Representing pretoken as single byte (storing utf-8 token as multi-bytes was a mistake)
- Changing the regex matcher to concurrent=False to limit conflicts with multi-processing
- Token interning + HashMap for merging

Now sub 1 min for TinyStoriesV2-GPT4-train.txt on a laptop.
But this is too expensive in terms of memory on my laptop.
Moving to mmap+memory_view so the required RAM remains below my laptop's spec. With current tuning this results in a 2x slowdown for pretokenization on TinyStoriesV2-GPT4-train. We remain sub 2min for tokenization.
Using mmap+memory_view, we can now pre-tokenize owt_train without going out-of-memory.

Training time on owt_train.txt is about 12h so we still need to optimize that.

Back to optimizing, now with merging, the main data structures optimizations are using linked-list for O(1) pre-token merging, BTreeSet for Log(n_pretoken) max-heap lookup and arena for storing nodes.

With these optimizations, merging is about 0.2s on TinyStoriesV2-GPT4-train and 35s on owt-train.

## Assignment Questions

### Unicode

<!-- prettier-ignore-start -->
??? question "What Unicode character does `chr(0)` return?"
    `'\x00'`

??? question "How does this character’s string representation (`repr()`) differ from its printed representation?"
    The printed representation is non-visible, whereas its string representation is `'\x00'`.

??? question "What happens when this character occurs in text?"
    When added to a string, its string representation is stored as is. When printed, this `NULL` character is non-visible.

??? question "What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32?"
    - The longer sequence of byte values (resp. 2x and 4x) lead to worse compression ratio.
    - All default ascii characters, which are likely the most common characters in standard text, can be represented using a single byte.

??? question "Why is the decoding function incorrect?"
    This function decodes the bytestring byte-by-byte. It will thus fail for characters that are encoded on multiple bytes.
    An example input bytestring that yields incorrect results is the encoding for é `b'\xc3\xa9'`

??? question "Give a two byte sequence that does not decode to any Unicode character(s)."
    `b'\xc3\x41'` is such a sequence. While the first byte is valid, the second byte is invalid as its binary form highest order bits don't match `10`.

<!-- prettier-ignore-end -->

### tokenizer_experiments

<!-- prettier-ignore-start -->
??? question "What is each tokenizer’s compression ratio (bytes/token)?"
    - TinyStories tokenizer on TinyStories data: 4.11 bytes/token
    - OpenWebText tokenizer on OpenWebText data: 4.34 bytes/token

??? question "What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens."
    When tokenizing the OpenWebText sample with the TinyStories tokenizer, the compression ratio is 3.26 bytes/token which is lower than the 4.34 bytes/token of the OpenWebText tokenizer on its own data, indicating less efficient encoding for out-of-domain text. The TinyStories tokenizer, trained on a a smaller and different corpus, and with a smaller vocabulary size, breaks down the more diverse OpenWebText into more numerous, smaller tokens.

??? question "Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?"
    Measuring the performance of our owt trained tokenizer on owt_valid.txt, we obtain a throughput of 1.2 MB/s.
    Our current (slow) implementation complexity is linear with regards to the dataset size, we can thus extrapolate it would take 8.24 days to tokenize the Pile dataset.

??? question "We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is uint16 an appropriate choice?"
    Our largest vocabulary contains 32000 positive token ids and uint16 allows to represent up to 65536 positive integers.

<!-- prettier-ignore-end -->

### transformer_accounting

<!-- prettier-ignore-start -->

??? question "Consider GPT-2 XL, which has the following configuration: vocab_size : 50,257, context_length : 1,024, num_layers : 48, d_model : 1,600, num_heads : 25, d_ff : 6,400. Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?"
    Our model has 2,127,057,600 trainable parameters.
    This is about 7.92 GiB of memory to load.

??? question "Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens."
    We have matrix multiplies in the ffn, the mhsa projections, the sdpa and the final layer. In total this sums up to about 4,513,336,524,800 FLOPs.

??? question "Based on your analysis above, which parts of the model require the most FLOPs?"
    3,019,898,880,000 (66.91%) of the matrix multiplies happen in the ffn.

??? question "Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?"
    As the model size increase, the importances of the ffn and projections increase while the importance of the final layer decreases a lot and the importance of the attetion slightly decreases.


??? question "Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change?"
    The total FLOPs grow drastically to 149,522,795,724,800 (33x).
    The attention FLOPs blow up to 55% and cause the other contributions to decrease relatively.



<!-- prettier-ignore-end -->

### learning_rate_tuning

<!-- prettier-ignore-start -->

??? question "Run the SGD example above with three other values for the learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of training)?"
    Increasing the learning rate from 1e-1 to 1e2, the loss converges faster. At 1e3, the loss diverges.

<!-- prettier-ignore-end -->

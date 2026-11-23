# Third-party notices

The extension's original source is MIT licensed ([LICENSE](LICENSE)); this does
not relicense the code it adapts or any model weights.

| Project | Use | License | Revision | Source and license |
| --- | --- | --- | --- | --- |
| Crucible | Adapted EXL3 trellis decode, lane window, n-gram row codec and GEMV structure (`exl3/format.hpp`, `exl3/rocm/exl3.hpp` front tile order, `exl3/rocm/exl3.hip.cpp`) | `Apache-2.0` | `6c8b36f16a3dedfc759ce0eadae60eea437db459` | [jaedync/crucible-strix-halo](https://github.com/jaedync/crucible-strix-halo); [license and notice](licenses/crucible.txt) |
| ExLlamaV3 | EXL3 and n-gram trellis format reference; no source import | `MIT` | `8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0` | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3); [license](licenses/exllamav3.txt) |

The build installs these texts with `LICENSE` under
`share/licenses/gufo/extensions/qwen38_flash_next_exl3`. Preserve the
copyright notices embedded in adapted files when you change them.

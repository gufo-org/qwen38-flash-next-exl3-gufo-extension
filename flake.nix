{
  description = "Gufo with the Qwen3.8 Flash-Next EXL3 extension";
  inputs.gufo.url = "github:gufo-org/gufo/d3a7bab1563e7266c52c426a149db00841b6d4d6";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self ];
    };
  };
}

{
  description = "Gufo with the Qwen3.8 Flash-Next EXL3 extension";
  inputs.gufo.url = "github:gufo-org/gufo/784891ede060aebe5e71554310b71c14c096d491";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self ];
    };
  };
}

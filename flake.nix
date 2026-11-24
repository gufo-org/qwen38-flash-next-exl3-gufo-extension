{
  description = "Gufo with the Qwen3.8 Flash-Next EXL3 extension";
  inputs.gufo.url = "github:gufo-org/gufo/7c9fb4aefe6fa84bd582d481ac5819a0e3dfec9b";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self ];
    };
  };
}

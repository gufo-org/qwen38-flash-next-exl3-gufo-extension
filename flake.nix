{
  description = "Gufo with the Qwen3.8 Flash-Next EXL3 extension";
  inputs.gufo.url = "github:gufo-org/gufo/8c646451d91f1dd2234c9a763a2c68715f32f506";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self ];
    };
  };
}

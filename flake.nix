{
  description = "Gufo with the Qwen3.8 Flash-Next EXL3 extension";
  inputs.gufo.url = "github:gufo-org/gufo/cab68390c0ddb7e944cd363287b836a54237c48a";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self ];
    };
  };
}

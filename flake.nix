{
  description = "Binary search speed comparison: AVX2 brute vs branchless vs branchy (u8/u16/u32)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux.default = let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in pkgs.mkShell {
      packages = with pkgs; [
        gcc
        rustc
        cargo
        python3
        python3Packages.matplotlib
        python3Packages.numpy
        python3Packages.pandas
      ];
    };
  };
}

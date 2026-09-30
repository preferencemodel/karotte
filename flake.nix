{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
  };

  outputs = {nixpkgs, ...}: let
    systems = ["aarch64-darwin" "x86_64-linux" "aarch64-linux"];
    linuxSystems = ["x86_64-linux" "aarch64-linux"];
    forAllSystems = nixpkgs.lib.genAttrs systems;
    isLinux = system: builtins.elem system linuxSystems;

    devPkgs = pkgs:
      with pkgs; [
        just
        uv
        git
        alejandra
        podman
      ];

    mkShell = pkgs:
      pkgs.mkShell {
        packages = devPkgs pkgs;
        shellHook = "unset PYTHONPATH";
      };
  in {
    devShells = forAllSystems (system: let
      pkgs = import nixpkgs {inherit system;};
    in {
      default = mkShell pkgs;
    });
  };
}

{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-23.11";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, ... }@inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
        };

        poetry2nix = inputs.poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };
      in
      {
        packages.env = poetry2nix.mkPoetryEnv {
          projectDir = ./.;
          preferWheels = true;
          # voici currently not in use,
          # add it to pyproject.toml and uncomment the following to install:
          # overrides = poetry2nix.overrides.withDefaults (self: super: {
          #   voici-core = super.voici-core.overridePythonAttrs (old: {
          #     nativeBuildInputs = [ self.hatch-jupyter-builder ];
          #     patches = [ ./voici-core.patch ];
          #     pyproject = true;
          #     format = null;
          #   });
          # });
        };

        packages.dashboards = pkgs.stdenv.mkDerivation {
          name = "dashboards";
          src = ./code;
          phases = [ "installPhase" ];
          installPhase = ''
            mkdir -p $out/var/lib
            cp -a $src $out/var/lib/dashboards
          '';
        };

        packages.data = pkgs.stdenv.mkDerivation {
          name = "data";
          src = ./data;
          phases = [ "installPhase" ];
          installPhase = ''
            mkdir -p $out/var/lib
            cp -a $src $out/var/lib/data
          '';
        };

        packages.templates = pkgs.stdenv.mkDerivation {
          name = "share";
          src = ./share;
          phases = [ "installPhase" ];
          installPhase = ''
            mkdir -p $out/usr/share
            cp -a $src/jupyter $out/usr/share/jupyter
          '';
        };

        packages.image = pkgs.dockerTools.streamLayeredImage {
          name = "service/multioptforest";
          tag = "latest";
          created = "now";
          contents = [
            (pkgs.buildEnv {
              name = "image-contents";
              paths = [
                pkgs.busybox
                pkgs.dockerTools.fakeNss
                pkgs.dockerTools.usrBinEnv
                pkgs.tini
                self.packages.x86_64-linux.env
                self.packages.x86_64-linux.dashboards
                self.packages.x86_64-linux.data
                self.packages.x86_64-linux.templates
              ];
              pathsToLink = [ "/etc" "/sbin" "/bin" "/var/lib/dashboards" "/var/lib/data" "/usr/share" ];
            })
          ];
          extraCommands = ''
            mkdir -p usr/bin && ln -s /sbin/env usr/bin/env
            mkdir -p tmp && chmod a+rxwt tmp
          '';
          config = {
            Entrypoint = [
              "${pkgs.tini}/bin/tini"
              "--"
              "${self.packages.x86_64-linux.env}/bin/voila"
            ];
            Env = [
              "TMPDIR=/tmp"
              "HOME=/tmp"
            ];
            Labels = pkgs.lib.importJSON ./Labels.json;
            User = "nobody";
          };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.poetry
            pkgs.micromamba
            self.packages.${system}.env
            # build dependencies
            pkgs.jfrog-cli
            pkgs.jq
            pkgs.python311Packages.xarray
          ];

          # expose the locally modified theme for jupyter to find
          shellHook = ''
            export JUPYTER_PATH=$(pwd)/share/jupyter
          '';
        };
      });
}

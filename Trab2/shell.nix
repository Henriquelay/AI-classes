let
  # We pin to a specific nixpkgs commit for reproducibility.
  # Check for new commits at https://status.nixos.org.
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/0069cacc45352ad3f6846e002d20b58c04d6034f.tar.gz") {};
in
  pkgs.mkShell {
    packages = with pkgs; [
      (python312.withPackages (python-pkgs:
        with python-pkgs; [
          numpy
          scipy
          pygame
          # jupyter
          # pandas
          matplotlib
          seaborn
          scikit-learn
        ]))
    ];
    shellHook = ''
      python3 Trab2_Henrique_Layber.py
      exit
    '';
  }

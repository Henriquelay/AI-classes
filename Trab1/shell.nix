let
  # We pin to a specific nixpkgs commit for reproducibility.
  # Check for new commits at https://status.nixos.org.
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/3bcedce9f4de37570242faf16e1e143583407eab.tar.gz") {};
in
  pkgs.mkShell {
    packages = with pkgs; [
      (python3.withPackages (python-pkgs:
        with python-pkgs; [
          jupyter
          numpy
          pandas
          matplotlib
          seaborn
          scikit-learn
          scipy
        ]))
    ];
    shellHook = "jupyter notebook --NotebookApp.token='' --NotebookApp.password=''";
  }

(lang dune 3.14)

(name funocaml-raven-rl)

(version 0.1)

(generate_opam_files true)

(source
 (github raven-ml/funocaml-2025-rl))

(authors "Lukasz Stafiniak")

(maintainers "Lukasz Stafiniak <lukstafi@gmail.com>")

(license "BSD-2-Clause")

(documentation https://github.com/raven-ml/funocaml-2025-rl/blob/main/docs/)

(pin
 (url "git+https://github.com/raven-ml/raven/")
 (package
  (name nx)
  (version dev)))

(pin
 (url "git+https://github.com/raven-ml/raven/")
 (package
  (name rune)
  (version dev)))

(pin
 (url "git+https://github.com/raven-ml/raven/")
 (package
  (name kaun)
  (version dev)))

(package
 (name funocaml-raven-rl)
 (depends (dune ()) rune kaun))

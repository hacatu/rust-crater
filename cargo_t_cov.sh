# Run `cargo t` and `grcov` to generate a coverage report
# See https://doc.rust-lang.org/rustc/instrument-coverage.html
# and https://github.com/mozilla/grcov

export RUSTFLAGS="-Z unstable-options -C instrument-coverage=except-unused-generics"
export RUSTDOCFLAGS="-Z unstable-options -C instrument-coverage=except-unused-generics --persist-doctests target/debug/doctestbins"

rm *.profraw

cargo t

$HOME/.cargo/bin/grcov . -s . --binary-path target/debug/ -t html --branch --ignore-not-existing -o target/debug/cov

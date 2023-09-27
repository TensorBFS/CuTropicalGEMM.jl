const script_dir = dirname(@__FILE__)
const src_dir = joinpath(script_dir)
const build_dir = joinpath(script_dir, "build")

mkpath(build_dir)

cd(build_dir) do
    run(`cmake $src_dir`)
    run(`make clean`)
    run(`make`)
end